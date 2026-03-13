from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np

from runtime_utils import (
    MEMMAP_LABELS_FILE,
    MEMMAP_META_FILE,
    MEMMAP_STYLES_FILE,
    MEMMAP_X_FILE,
)


def decode_style_names(raw: np.ndarray) -> list[str]:
    names: list[str] = []
    for item in raw:
        if isinstance(item, (bytes, bytearray)):
            names.append(item.decode("utf-8"))
        else:
            names.append(str(item))
    return names


def parse_style_selector(selector: Optional[str], style_names: Sequence[str]) -> Optional[int]:
    if selector is None:
        return None
    s = selector.strip()
    if s == "":
        return None
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        label = int(s)
        if not (0 <= label < len(style_names)):
            raise ValueError(f"Style label out of range: {label}, valid [0, {len(style_names)-1}]")
        return label
    if s in style_names:
        return style_names.index(s)
    lower_map = {name.lower(): idx for idx, name in enumerate(style_names)}
    key = s.lower()
    if key in lower_map:
        return lower_map[key]
    raise ValueError(f"Style '{selector}' not found in style names")


def contiguous_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    if labels.size == 0:
        return runs
    start = 0
    curr = int(labels[0])
    for i in range(1, labels.shape[0]):
        v = int(labels[i])
        if v != curr:
            runs.append((start, i, curr))
            start = i
            curr = v
    runs.append((start, labels.shape[0], curr))
    return runs


def choose_segment(
    labels: np.ndarray,
    num_frames: int,
    seed: int,
    target_label: Optional[int] = None,
    start_index: Optional[int] = None,
) -> tuple[int, int, int, tuple[int, int, int]]:
    runs = contiguous_runs(labels)
    if start_index is not None:
        start = int(start_index)
        end = start + num_frames
        if start < 0 or end > labels.shape[0]:
            raise ValueError(
                f"--start-index out of range: [{start}, {end}) with total frames {labels.shape[0]}"
            )
        lb0 = int(labels[start])
        if np.any(labels[start:end] != lb0):
            raise ValueError(
                f"Segment [{start},{end}) crosses style label boundary. "
                "Use a different --start-index."
            )
        if target_label is not None and lb0 != target_label:
            raise ValueError(
                f"--start-index segment label {lb0} does not match requested style {target_label}"
            )
        return start, end, lb0, (start, end, lb0)

    rng = random.Random(seed)
    candidates: list[tuple[int, int, int]] = []
    for rs, re, lb in runs:
        if re - rs < num_frames:
            continue
        if target_label is not None and lb != target_label:
            continue
        candidates.append((rs, re, lb))
    if not candidates:
        max_len = max((re - rs for rs, re, _ in runs), default=0)
        raise RuntimeError(
            f"No contiguous run can provide {num_frames} frames for style {target_label}. "
            f"Max run length={max_len}"
        )
    rs, re, lb = rng.choice(candidates)
    start = rng.randint(rs, re - num_frames)
    end = start + num_frames
    return start, end, lb, (rs, re, lb)


def load_db_arrays(db_path: Path) -> tuple[object, np.ndarray, list[str], str]:
    if db_path.is_file() and db_path.suffix.lower() in {".h5", ".hdf5"}:
        f = h5py.File(db_path, "r")
        if "X" not in f or "Labels" not in f:
            f.close()
            raise KeyError("HDF5 DB must contain datasets 'X' and 'Labels'")
        x_source = f["X"]
        labels = np.asarray(f["Labels"][:], dtype=np.int64)
        if x_source.ndim != 2 or x_source.shape[1] != 375:
            f.close()
            raise ValueError(f"Expected X shape (N, 375), got {x_source.shape}")
        if labels.ndim != 1 or labels.shape[0] != x_source.shape[0]:
            f.close()
            raise ValueError(f"Labels shape {labels.shape} incompatible with X shape {x_source.shape}")
        if "StyleNames" in f:
            style_names = decode_style_names(f["StyleNames"][:])
        else:
            max_label = int(labels.max()) if labels.size else -1
            style_names = [f"label_{i}" for i in range(max_label + 1)]
        return x_source, labels, style_names, "hdf5"

    if db_path.is_dir():
        meta_path = db_path / MEMMAP_META_FILE
        x_path = db_path / MEMMAP_X_FILE
        labels_path = db_path / MEMMAP_LABELS_FILE
        if not (meta_path.exists() and x_path.exists() and labels_path.exists()):
            raise FileNotFoundError(
                f"Memmap DB must include {MEMMAP_META_FILE}, {MEMMAP_X_FILE}, {MEMMAP_LABELS_FILE}"
            )
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        num_frames = int(meta.get("num_frames", -1))
        feature_dim = int(meta.get("feature_dim", -1))
        if num_frames <= 0:
            raise ValueError(f"Invalid num_frames in {meta_path}: {num_frames}")
        if feature_dim != 375:
            raise ValueError(f"Invalid feature_dim in {meta_path}: {feature_dim}, expected 375")
        x_source = np.memmap(x_path, dtype=np.float32, mode="r", shape=(num_frames, 375))
        labels_mm = np.memmap(labels_path, dtype=np.int32, mode="r", shape=(num_frames,))
        labels = np.asarray(labels_mm, dtype=np.int64)
        styles_path = db_path / MEMMAP_STYLES_FILE
        if styles_path.exists():
            with styles_path.open("r", encoding="utf-8") as f:
                style_names = list(json.load(f))
        else:
            max_label = int(labels.max()) if labels.size else -1
            style_names = [f"label_{i}" for i in range(max_label + 1)]
        return x_source, labels, style_names, "memmap"

    raise ValueError(f"Unsupported --db path: {db_path}")


def load_norm_stats(norm_stats_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"norm stats not found: {norm_stats_path}")
    data = np.load(norm_stats_path)
    if "mean" not in data or "std" not in data:
        raise KeyError(f"{norm_stats_path} must contain keys: mean, std")
    mean = np.asarray(data["mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(data["std"], dtype=np.float32).reshape(-1)
    if mean.shape != (375,) or std.shape != (375,):
        raise ValueError(f"norm stats must have shape (375,), got mean={mean.shape}, std={std.shape}")
    if np.any(std <= 0):
        raise ValueError("norm std must be > 0 for all dims")
    return mean, std

