"""
Dataset/DataLoader helpers for 100STYLE 375-dim memmap databases.

Required memmap layout:
  - meta.json
  - X.f32
  - Labels.i32
Optional:
  - StyleNames.json
  - segments.json

Normalization:
  - optional z-score normalization on 375 dims
  - stats are estimated from train sample windows and shared with val/test

If `segments.json` contains `split=train|val|test`, those offline splits are honored.
Otherwise the loader falls back to the legacy random split over sample starts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


MEMMAP_META_FILE = "meta.json"
MEMMAP_X_FILE = "X.f32"
MEMMAP_LABELS_FILE = "Labels.i32"
MEMMAP_STYLES_FILE = "StyleNames.json"
MEMMAP_SEGMENTS_FILE = "segments.json"


def _load_memmap_meta(memmap_dir: Path) -> dict:
    meta_path = memmap_dir / MEMMAP_META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"Memmap meta not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_memmap_segments(memmap_dir: Path, *, num_frames: int) -> list[dict[str, object]] | None:
    segments_path = memmap_dir / MEMMAP_SEGMENTS_FILE
    if not segments_path.exists():
        return None

    with segments_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"{segments_path} must contain a JSON list")

    segments: list[dict[str, object]] = []
    for idx, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid segment record at index {idx}: expected object, got {type(raw)}")
        if "frame_start" not in raw or "num_frames" not in raw or "label" not in raw:
            raise KeyError(f"Segment record {idx} must contain frame_start, num_frames, label")

        frame_start = int(raw["frame_start"])
        num_segment_frames = int(raw["num_frames"])
        label = int(raw["label"])
        if frame_start < 0:
            raise ValueError(f"Segment record {idx} has negative frame_start: {frame_start}")
        if num_segment_frames < 0:
            raise ValueError(f"Segment record {idx} has negative num_frames: {num_segment_frames}")
        if frame_start + num_segment_frames > num_frames:
            raise ValueError(
                f"Segment record {idx} exceeds DB bounds: start={frame_start} len={num_segment_frames} "
                f"num_frames={num_frames}"
            )

        record = {
            "frame_start": frame_start,
            "num_frames": num_segment_frames,
            "label": label,
        }
        for key in (
            "split",
            "style_name",
            "source_entry_id",
            "source_path",
            "suffix",
            "mirror",
            "source_start",
            "source_stop",
            "source_global_label",
        ):
            if key in raw:
                record[key] = raw[key]
        segments.append(record)

    segments.sort(key=lambda item: (int(item["frame_start"]), int(item["num_frames"])))
    prev_end = 0
    for idx, record in enumerate(segments):
        frame_start = int(record["frame_start"])
        num_segment_frames = int(record["num_frames"])
        if idx > 0 and frame_start < prev_end:
            raise ValueError(f"Overlapping segments detected near frame {frame_start} in {segments_path}")
        prev_end = frame_start + num_segment_frames
    return segments


def _build_sample_starts(
    labels: np.ndarray,
    seq_len: int,
    stride: int,
    enforce_same_label: bool,
) -> np.ndarray:
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    n = int(labels.shape[0])
    if n < seq_len:
        return np.empty((0,), dtype=np.int64)

    if seq_len == 1:
        return np.arange(0, n, stride, dtype=np.int64)

    if not enforce_same_label:
        return np.arange(0, n - seq_len + 1, stride, dtype=np.int64)

    starts: list[int] = []
    run_start = 0
    while run_start < n:
        run_label = labels[run_start]
        run_end = run_start + 1
        while run_end < n and labels[run_end] == run_label:
            run_end += 1

        run_len = run_end - run_start
        if run_len >= seq_len:
            starts.extend(range(run_start, run_end - seq_len + 1, stride))

        run_start = run_end

    return np.asarray(starts, dtype=np.int64)


def _build_sample_starts_from_segments(
    segments: list[dict[str, object]],
    *,
    seq_len: int,
    stride: int,
) -> np.ndarray:
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    starts: list[int] = []
    for segment in segments:
        frame_start = int(segment["frame_start"])
        num_frames = int(segment["num_frames"])
        if num_frames < seq_len:
            continue
        starts.extend(range(frame_start, frame_start + num_frames - seq_len + 1, stride))
    return np.asarray(starts, dtype=np.int64)


def _validate_norm_stats(norm_mean: np.ndarray, norm_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(norm_mean, dtype=np.float32).reshape(-1)
    std = np.asarray(norm_std, dtype=np.float32).reshape(-1)
    if mean.shape != (375,):
        raise ValueError(f"norm_mean must have shape (375,), got {mean.shape}")
    if std.shape != (375,):
        raise ValueError(f"norm_std must have shape (375,), got {std.shape}")
    if np.any(std <= 0):
        raise ValueError("norm_std must be positive for all dims")
    return mean, std


def _compute_zscore_stats_from_starts(
    sample_starts: np.ndarray,
    seq_len: int,
    norm_eps: float,
    max_windows_for_stats: int = 0,
    seed: int = 42,
    x_source=None,
    db_path: Optional[Path] = None,
    windows_chunk_size: int = 4096,
    h5_cache_mb: int = 256,
    h5_cache_slots: int = 1_000_003,
) -> tuple[np.ndarray, np.ndarray]:
    del db_path, h5_cache_mb, h5_cache_slots

    starts = np.asarray(sample_starts, dtype=np.int64).reshape(-1)
    if starts.size == 0:
        raise ValueError("Cannot compute normalization stats from empty sample_starts")
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if norm_eps <= 0:
        raise ValueError(f"norm_eps must be > 0, got {norm_eps}")
    if windows_chunk_size < 1:
        raise ValueError(f"windows_chunk_size must be >= 1, got {windows_chunk_size}")
    if x_source is None:
        raise ValueError("x_source is required for memmap normalization-stat computation")

    if max_windows_for_stats > 0 and starts.size > max_windows_for_stats:
        rng = np.random.default_rng(seed)
        keep = rng.choice(starts.size, size=max_windows_for_stats, replace=False)
        starts = starts[keep]
    starts = np.sort(starts)

    sum_x = np.zeros((375,), dtype=np.float64)
    sum_x2 = np.zeros((375,), dtype=np.float64)
    count = 0

    def _read_block(source, begin: int, end: int) -> np.ndarray:
        return np.asarray(source[begin:end], dtype=np.float64).reshape(-1, 375)

    def _accumulate_contiguous_windows(source, start_begin: int, start_end: int) -> None:
        nonlocal sum_x, sum_x2, count
        s = int(start_begin)
        last = int(start_end)
        while s <= last:
            n_win = min(windows_chunk_size, last - s + 1)
            read_begin = s
            read_end = s + n_win + seq_len - 1
            block = _read_block(source, read_begin, read_end)

            prefix = np.zeros((block.shape[0] + 1, 375), dtype=np.float64)
            prefix2 = np.zeros((block.shape[0] + 1, 375), dtype=np.float64)
            np.cumsum(block, axis=0, out=prefix[1:])
            np.cumsum(np.square(block), axis=0, out=prefix2[1:])

            win_sum = prefix[seq_len : seq_len + n_win] - prefix[:n_win]
            win_sum2 = prefix2[seq_len : seq_len + n_win] - prefix2[:n_win]

            sum_x += win_sum.sum(axis=0)
            sum_x2 += win_sum2.sum(axis=0)
            count += n_win * seq_len
            s += n_win

    i = 0
    while i < starts.size:
        j = i + 1
        while j < starts.size and starts[j] == starts[j - 1] + 1:
            j += 1
        _accumulate_contiguous_windows(x_source, int(starts[i]), int(starts[j - 1]))
        i = j

    if count == 0:
        raise ValueError("No samples found while computing normalization stats")

    mean = sum_x / count
    var = sum_x2 / count - np.square(mean)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    std = np.maximum(std, norm_eps)
    return mean.astype(np.float32), std.astype(np.float32)


class Motion375Dataset(Dataset):
    """
    Read 375-dim features from memmap and emit frame/sequence samples.

    Returned item fields:
      - x:      [375] or [T, 375]
      - traj:   [84]  or [T, 84]
      - motion: [291] or [T, 291]
      - label:  scalar int64 style id
      - start:  scalar int64 start frame index in original database
    """

    def __init__(
        self,
        db_path: str | Path,
        seq_len: int = 1,
        stride: int = 1,
        enforce_same_label: bool = True,
        sample_starts: Optional[np.ndarray] = None,
        load_into_memory: bool = False,
        normalize: bool = False,
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
        norm_eps: float = 1e-8,
        max_windows_for_stats: int = 0,
        stats_seed: int = 42,
        h5_cache_mb: int = 256,
        h5_cache_slots: int = 1_000_003,
    ):
        del h5_cache_mb, h5_cache_slots

        super().__init__()
        self.db_path = Path(db_path).expanduser().resolve()
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.enforce_same_label = bool(enforce_same_label)
        self.load_into_memory = bool(load_into_memory)
        self.normalize = bool(normalize)
        self.norm_eps = float(norm_eps)
        self.max_windows_for_stats = int(max_windows_for_stats)
        self.stats_seed = int(stats_seed)

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database path not found: {self.db_path}")
        if not self.db_path.is_dir():
            raise ValueError(f"Unsupported db path: {self.db_path}. Expected memmap directory.")

        meta = _load_memmap_meta(self.db_path)
        num_frames = int(meta.get("num_frames", -1))
        feature_dim = int(meta.get("feature_dim", -1))
        if num_frames < 0:
            raise ValueError(f"Invalid num_frames in {self.db_path / MEMMAP_META_FILE}: {num_frames}")
        if feature_dim != 375:
            raise ValueError(
                f"Expected feature_dim=375 in {self.db_path / MEMMAP_META_FILE}, got {feature_dim}"
            )

        x_path = self.db_path / MEMMAP_X_FILE
        labels_path = self.db_path / MEMMAP_LABELS_FILE
        if not x_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"Memmap files missing under {self.db_path}: need {MEMMAP_X_FILE} and {MEMMAP_LABELS_FILE}"
            )

        self.num_frames = num_frames
        labels_mm = np.memmap(labels_path, dtype=np.int32, mode="r", shape=(self.num_frames,))
        self.labels = np.asarray(labels_mm, dtype=np.int64)

        styles_path = self.db_path / MEMMAP_STYLES_FILE
        if styles_path.exists():
            with styles_path.open("r", encoding="utf-8") as f:
                self.style_names = list(json.load(f))
        else:
            max_label = int(self.labels.max()) if self.labels.size else -1
            self.style_names = [f"label_{i}" for i in range(max_label + 1)]

        self.segments = _load_memmap_segments(self.db_path, num_frames=self.num_frames)
        self.has_precomputed_splits = False
        if self.segments is not None:
            self.has_precomputed_splits = any(
                str(segment.get("split", "")) in {"train", "val", "test"}
                for segment in self.segments
            )

        if self.load_into_memory:
            x_mm = np.memmap(x_path, dtype=np.float32, mode="r", shape=(self.num_frames, 375))
            self._x_mem = np.asarray(x_mm, dtype=np.float32).copy()
        else:
            self._x_mem = None

        if sample_starts is None:
            if self.segments is not None:
                self.sample_starts = _build_sample_starts_from_segments(
                    self.segments,
                    seq_len=self.seq_len,
                    stride=self.stride,
                )
            else:
                self.sample_starts = _build_sample_starts(
                    labels=self.labels,
                    seq_len=self.seq_len,
                    stride=self.stride,
                    enforce_same_label=self.enforce_same_label,
                )
        else:
            starts = np.asarray(sample_starts, dtype=np.int64).reshape(-1)
            if starts.size:
                if starts.min() < 0:
                    raise ValueError("sample_starts contains negative index")
                max_start = self.num_frames - self.seq_len
                if starts.max() > max_start:
                    raise ValueError(
                        f"sample_starts out of range: max start is {max_start}, got {int(starts.max())}"
                    )
            self.sample_starts = starts

        self._x_ds = None

        self.norm_mean: Optional[np.ndarray] = None
        self.norm_std: Optional[np.ndarray] = None
        if self.normalize:
            if (norm_mean is None) ^ (norm_std is None):
                raise ValueError("norm_mean and norm_std must be both provided or both omitted")
            if norm_mean is None:
                self._ensure_open()
                stats_source = self._x_mem if self._x_mem is not None else self._x_ds
                mean, std = _compute_zscore_stats_from_starts(
                    sample_starts=self.sample_starts,
                    seq_len=self.seq_len,
                    norm_eps=self.norm_eps,
                    max_windows_for_stats=self.max_windows_for_stats,
                    seed=self.stats_seed,
                    x_source=stats_source,
                )
            else:
                mean, std = _validate_norm_stats(norm_mean, norm_std)
            self.norm_mean = mean
            self.norm_std = std

    def __len__(self) -> int:
        return int(self.sample_starts.shape[0])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_x_ds"] = None
        return state

    def __del__(self):
        self._x_ds = None

    def _ensure_open(self):
        if self._x_mem is not None:
            return
        if self._x_ds is None:
            self._x_ds = self._open_memmap()

    def _open_memmap(self):
        x_path = self.db_path / MEMMAP_X_FILE
        return np.memmap(x_path, dtype=np.float32, mode="r", shape=(self.num_frames, 375))

    def _read_x(self, start: int, end: int) -> np.ndarray:
        if self._x_mem is not None:
            x = self._x_mem[start:end]
        else:
            self._ensure_open()
            x = self._x_ds[start:end]
        return np.asarray(x, dtype=np.float32)

    def __getitem__(self, idx: int):
        start = int(self.sample_starts[idx])
        end = start + self.seq_len
        x = self._read_x(start, end)
        if self.seq_len == 1:
            x = x[0]

        if self.normalize:
            x = (x - self.norm_mean) / self.norm_std

        label = int(self.labels[start])
        x_t = torch.from_numpy(x)
        traj_t = x_t[..., :84]
        motion_t = x_t[..., 84:]

        return {
            "x": x_t,
            "traj": traj_t,
            "motion": motion_t,
            "label": torch.tensor(label, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
        }

    def get_norm_stats(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.norm_mean is None or self.norm_std is None:
            return None, None
        return self.norm_mean.copy(), self.norm_std.copy()


@dataclass
class DataBundle:
    train_dataset: Motion375Dataset
    val_dataset: Optional[Motion375Dataset]
    test_dataset: Optional[Motion375Dataset]
    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    test_loader: Optional[DataLoader]
    norm_mean: Optional[np.ndarray]
    norm_std: Optional[np.ndarray]


def _validate_split_ratios(val_ratio: float, test_ratio: float) -> None:
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in [0, 1), got {test_ratio}")
    if (val_ratio + test_ratio) >= 1.0:
        raise ValueError(f"val_ratio + test_ratio must be < 1, got {val_ratio + test_ratio}")


def _resolve_split_counts(
    num_samples: int,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int]:
    if num_samples < 1:
        return 0, 0

    n_val = int(round(num_samples * val_ratio)) if val_ratio > 0.0 else 0
    n_test = int(round(num_samples * test_ratio)) if test_ratio > 0.0 else 0

    if val_ratio > 0.0:
        n_val = max(1, n_val)
    if test_ratio > 0.0:
        n_test = max(1, n_test)

    max_heldout = max(num_samples - 1, 0)
    while (n_val + n_test) > max_heldout:
        if n_test > 0 and (n_val == 0 or test_ratio <= val_ratio):
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break
    return n_val, n_test


def _split_sample_starts(
    sample_starts: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    starts = np.asarray(sample_starts, dtype=np.int64).reshape(-1)
    if starts.size == 0:
        return starts.copy(), None, None

    n_val, n_test = _resolve_split_counts(starts.size, val_ratio, test_ratio)
    if n_val == 0 and n_test == 0:
        return starts.copy(), None, None

    rng = np.random.default_rng(seed)
    perm = rng.permutation(starts.size)
    test_ids = perm[:n_test]
    val_ids = perm[n_test : n_test + n_val]
    train_ids = perm[n_test + n_val :]

    train_starts = starts[train_ids]
    val_starts = starts[val_ids] if n_val > 0 else None
    test_starts = starts[test_ids] if n_test > 0 else None
    return train_starts, val_starts, test_starts


def _precomputed_split_segments(
    segments: list[dict[str, object]] | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]] | None:
    if not segments:
        return None

    split_map = {"train": [], "val": [], "test": []}
    saw_precomputed = False
    for segment in segments:
        split = segment.get("split")
        if split in split_map:
            saw_precomputed = True
            split_map[str(split)].append(segment)
    if not saw_precomputed:
        return None
    return split_map["train"], split_map["val"], split_map["test"]


def _build_split_dataset(
    *,
    db_path: str | Path,
    seq_len: int,
    stride: int,
    enforce_same_label: bool,
    sample_starts: Optional[np.ndarray],
    load_into_memory: bool,
    normalize: bool,
    norm_mean: Optional[np.ndarray],
    norm_std: Optional[np.ndarray],
    norm_eps: float,
    max_windows_for_stats: int,
    seed: int,
    h5_cache_mb: int,
    h5_cache_slots: int,
) -> Optional[Motion375Dataset]:
    if sample_starts is None:
        return None
    return Motion375Dataset(
        db_path=db_path,
        seq_len=seq_len,
        stride=stride,
        enforce_same_label=enforce_same_label,
        sample_starts=sample_starts,
        load_into_memory=load_into_memory,
        normalize=normalize,
        norm_mean=norm_mean,
        norm_std=norm_std,
        norm_eps=norm_eps,
        max_windows_for_stats=max_windows_for_stats,
        stats_seed=seed,
        h5_cache_mb=h5_cache_mb,
        h5_cache_slots=h5_cache_slots,
    )


def _build_loader_for_split(
    dataset: Optional[Motion375Dataset],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    shuffle: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> Optional[DataLoader]:
    if dataset is None:
        return None
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **loader_kwargs)


def build_train_val_test_datasets(
    db_path: str | Path,
    seq_len: int = 1,
    stride: int = 1,
    enforce_same_label: bool = True,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    load_into_memory: bool = False,
    normalize: bool = True,
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
    norm_eps: float = 1e-8,
    max_windows_for_stats: int = 0,
    stats_windows_chunk_size: int = 4096,
    h5_cache_mb: int = 256,
    h5_cache_slots: int = 1_000_003,
) -> tuple[Motion375Dataset, Optional[Motion375Dataset], Optional[Motion375Dataset]]:
    _validate_split_ratios(val_ratio=val_ratio, test_ratio=test_ratio)

    base = Motion375Dataset(
        db_path=db_path,
        seq_len=seq_len,
        stride=stride,
        enforce_same_label=enforce_same_label,
        sample_starts=None,
        load_into_memory=load_into_memory,
        normalize=False,
        h5_cache_mb=h5_cache_mb,
        h5_cache_slots=h5_cache_slots,
    )

    resolved_norm_mean = None if norm_mean is None else np.asarray(norm_mean, dtype=np.float32)
    resolved_norm_std = None if norm_std is None else np.asarray(norm_std, dtype=np.float32)
    if (resolved_norm_mean is None) ^ (resolved_norm_std is None):
        raise ValueError("norm_mean and norm_std must be both provided or both omitted")

    if len(base) == 0:
        if normalize:
            raise RuntimeError("Cannot compute normalization stats: dataset has zero samples")
        return base, None, None

    precomputed = _precomputed_split_segments(base.segments)
    if precomputed is not None:
        train_segments, val_segments, test_segments = precomputed
        train_starts = _build_sample_starts_from_segments(train_segments, seq_len=seq_len, stride=stride)
        val_starts = _build_sample_starts_from_segments(val_segments, seq_len=seq_len, stride=stride)
        test_starts = _build_sample_starts_from_segments(test_segments, seq_len=seq_len, stride=stride)
        val_starts = val_starts if val_starts.size > 0 else None
        test_starts = test_starts if test_starts.size > 0 else None
        print(f"[Info] using offline split metadata from {Path(db_path).expanduser().resolve() / MEMMAP_SEGMENTS_FILE}")
    else:
        train_starts, val_starts, test_starts = _split_sample_starts(
            sample_starts=base.sample_starts,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    if train_starts.size == 0:
        raise RuntimeError("Training split is empty after applying sample-window construction")

    if normalize:
        if resolved_norm_mean is None:
            base._ensure_open()
            stats_source = base._x_mem if base._x_mem is not None else base._x_ds
            resolved_norm_mean, resolved_norm_std = _compute_zscore_stats_from_starts(
                sample_starts=train_starts,
                seq_len=seq_len,
                norm_eps=norm_eps,
                max_windows_for_stats=max_windows_for_stats,
                seed=seed,
                x_source=stats_source,
                windows_chunk_size=stats_windows_chunk_size,
                h5_cache_mb=h5_cache_mb,
                h5_cache_slots=h5_cache_slots,
            )
        else:
            resolved_norm_mean, resolved_norm_std = _validate_norm_stats(
                resolved_norm_mean,
                resolved_norm_std,
            )

    train_dataset = _build_split_dataset(
        db_path=db_path,
        seq_len=seq_len,
        stride=stride,
        enforce_same_label=enforce_same_label,
        sample_starts=train_starts,
        load_into_memory=load_into_memory,
        normalize=normalize,
        norm_mean=resolved_norm_mean,
        norm_std=resolved_norm_std,
        norm_eps=norm_eps,
        max_windows_for_stats=max_windows_for_stats,
        seed=seed,
        h5_cache_mb=h5_cache_mb,
        h5_cache_slots=h5_cache_slots,
    )
    if train_dataset is None:
        raise RuntimeError("train split is unexpectedly empty")

    val_dataset = _build_split_dataset(
        db_path=db_path,
        seq_len=seq_len,
        stride=stride,
        enforce_same_label=enforce_same_label,
        sample_starts=val_starts,
        load_into_memory=load_into_memory,
        normalize=normalize,
        norm_mean=resolved_norm_mean,
        norm_std=resolved_norm_std,
        norm_eps=norm_eps,
        max_windows_for_stats=max_windows_for_stats,
        seed=seed,
        h5_cache_mb=h5_cache_mb,
        h5_cache_slots=h5_cache_slots,
    )
    test_dataset = _build_split_dataset(
        db_path=db_path,
        seq_len=seq_len,
        stride=stride,
        enforce_same_label=enforce_same_label,
        sample_starts=test_starts,
        load_into_memory=load_into_memory,
        normalize=normalize,
        norm_mean=resolved_norm_mean,
        norm_std=resolved_norm_std,
        norm_eps=norm_eps,
        max_windows_for_stats=max_windows_for_stats,
        seed=seed,
        h5_cache_mb=h5_cache_mb,
        h5_cache_slots=h5_cache_slots,
    )

    base._x_ds = None
    return train_dataset, val_dataset, test_dataset


def build_train_val_datasets(
    db_path: str | Path,
    seq_len: int = 1,
    stride: int = 1,
    enforce_same_label: bool = True,
    val_ratio: float = 0.1,
    seed: int = 42,
    load_into_memory: bool = False,
    normalize: bool = True,
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
    norm_eps: float = 1e-8,
    max_windows_for_stats: int = 0,
    stats_windows_chunk_size: int = 4096,
    h5_cache_mb: int = 256,
    h5_cache_slots: int = 1_000_003,
) -> tuple[Motion375Dataset, Optional[Motion375Dataset]]:
    train_dataset, val_dataset, _ = build_train_val_test_datasets(
        db_path=db_path,
        seq_len=seq_len,
        stride=stride,
        enforce_same_label=enforce_same_label,
        val_ratio=val_ratio,
        test_ratio=0.0,
        seed=seed,
        load_into_memory=load_into_memory,
        normalize=normalize,
        norm_mean=norm_mean,
        norm_std=norm_std,
        norm_eps=norm_eps,
        max_windows_for_stats=max_windows_for_stats,
        stats_windows_chunk_size=stats_windows_chunk_size,
        h5_cache_mb=h5_cache_mb,
        h5_cache_slots=h5_cache_slots,
    )
    return train_dataset, val_dataset


def build_train_val_test_dataloaders(
    db_path: str | Path,
    seq_len: int = 1,
    stride: int = 1,
    enforce_same_label: bool = True,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    load_into_memory: bool = False,
    normalize: bool = True,
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
    norm_eps: float = 1e-8,
    max_windows_for_stats: int = 0,
    stats_windows_chunk_size: int = 4096,
    h5_cache_mb: int = 256,
    h5_cache_slots: int = 1_000_003,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
) -> DataBundle:
    if num_workers > 0 and prefetch_factor < 1:
        raise ValueError(f"prefetch_factor must be >= 1 when num_workers > 0, got {prefetch_factor}")

    train_dataset, val_dataset, test_dataset = build_train_val_test_datasets(
        db_path=db_path,
        seq_len=seq_len,
        stride=stride,
        enforce_same_label=enforce_same_label,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        load_into_memory=load_into_memory,
        normalize=normalize,
        norm_mean=norm_mean,
        norm_std=norm_std,
        norm_eps=norm_eps,
        max_windows_for_stats=max_windows_for_stats,
        stats_windows_chunk_size=stats_windows_chunk_size,
        h5_cache_mb=h5_cache_mb,
        h5_cache_slots=h5_cache_slots,
    )
    norm_mean, norm_std = train_dataset.get_norm_stats()

    train_loader = _build_loader_for_split(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        shuffle=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = _build_loader_for_split(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        shuffle=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = _build_loader_for_split(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        shuffle=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    if train_loader is None:
        raise RuntimeError("train loader is unexpectedly None")

    return DataBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )


def build_train_val_dataloaders(
    db_path: str | Path,
    seq_len: int = 1,
    stride: int = 1,
    enforce_same_label: bool = True,
    val_ratio: float = 0.1,
    seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    load_into_memory: bool = False,
    normalize: bool = True,
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
    norm_eps: float = 1e-8,
    max_windows_for_stats: int = 0,
    stats_windows_chunk_size: int = 4096,
    h5_cache_mb: int = 256,
    h5_cache_slots: int = 1_000_003,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
) -> DataBundle:
    return build_train_val_test_dataloaders(
        db_path=db_path,
        seq_len=seq_len,
        stride=stride,
        enforce_same_label=enforce_same_label,
        val_ratio=val_ratio,
        test_ratio=0.0,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        load_into_memory=load_into_memory,
        normalize=normalize,
        norm_mean=norm_mean,
        norm_std=norm_std,
        norm_eps=norm_eps,
        max_windows_for_stats=max_windows_for_stats,
        stats_windows_chunk_size=stats_windows_chunk_size,
        h5_cache_mb=h5_cache_mb,
        h5_cache_slots=h5_cache_slots,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


__all__ = [
    "Motion375Dataset",
    "DataBundle",
    "build_train_val_test_datasets",
    "build_train_val_test_dataloaders",
    "build_train_val_datasets",
    "build_train_val_dataloaders",
]
