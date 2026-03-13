"""
Generate memmap-only 100STYLE databases with 375-dim trajectory+motion features.

Outputs:
  - in-distribution memmap DB:
      meta.json
      X.f32
      Labels.i32
      StyleNames.json
      segments.json
  - optional zero-shot memmap DB for the held-out tail styles:
      meta.json
      X.f32
      Labels.i32
      StyleNames.json
      segments.json

Offline split policy:
  - the last K styles are peeled out as a zero-shot database
  - the remaining styles are split into train/val/test at clip level
  - split assignment is stratified per style and approximately balanced by frame count

Default dataset root:
  /home/shinn/桌面/revision/data/100sty
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import scipy.signal as signal


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CODE_DIR = REPO_ROOT / "Code"

MEMMAP_META_FILE = "meta.json"
MEMMAP_X_FILE = "X.f32"
MEMMAP_LABELS_FILE = "Labels.i32"
MEMMAP_STYLES_FILE = "StyleNames.json"
MEMMAP_SEGMENTS_FILE = "segments.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:  # noqa: E402
    import Utils.bvh as bvh  # noqa: E402
    from Utils import quat  # noqa: E402
except ModuleNotFoundError:  # noqa: E402
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    import Utils.bvh as bvh  # noqa: E402
    from Utils import quat  # noqa: E402


@dataclass(frozen=True)
class SegmentSummary:
    frames: int
    segments: int


class MemmapWriter:
    def __init__(
        self,
        *,
        out_dir: Path,
        style_names: list[str],
        overwrite: bool,
    ) -> None:
        self.out_dir = out_dir
        self.style_names = list(style_names)
        self.overwrite = bool(overwrite)
        self.num_frames = 0
        self.segments: list[dict[str, object]] = []

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.out_dir / MEMMAP_META_FILE
        self.x_path = self.out_dir / MEMMAP_X_FILE
        self.labels_path = self.out_dir / MEMMAP_LABELS_FILE
        self.styles_path = self.out_dir / MEMMAP_STYLES_FILE
        self.segments_path = self.out_dir / MEMMAP_SEGMENTS_FILE

        if not self.overwrite:
            for path in (self.meta_path, self.x_path, self.labels_path, self.styles_path, self.segments_path):
                if path.exists():
                    raise FileExistsError(f"{path} already exists. Use --overwrite to replace.")
        else:
            for path in (self.meta_path, self.x_path, self.labels_path, self.styles_path, self.segments_path):
                if path.exists():
                    path.unlink()

        self._x_file = self.x_path.open("wb")
        self._labels_file = self.labels_path.open("wb")

    def write_segment(
        self,
        *,
        x: np.ndarray,
        label: int,
        style_name: str,
        split: Optional[str],
        source_entry_id: int,
        source_path: Path,
        suffix: str,
        mirror: bool,
        source_start: int,
        source_stop: int,
        source_global_label: int,
    ) -> int:
        x_np = np.asarray(x, dtype=np.float32)
        if x_np.ndim != 2 or x_np.shape[1] != 375:
            raise ValueError(f"Expected x shape (N, 375), got {x_np.shape}")
        if x_np.shape[0] == 0:
            return 0

        frame_start = int(self.num_frames)
        labels_np = np.full((x_np.shape[0],), int(label), dtype=np.int32)
        x_np.tofile(self._x_file)
        labels_np.tofile(self._labels_file)

        record = {
            "frame_start": frame_start,
            "num_frames": int(x_np.shape[0]),
            "label": int(label),
            "style_name": str(style_name),
            "source_entry_id": int(source_entry_id),
            "source_path": str(source_path),
            "suffix": str(suffix),
            "mirror": bool(mirror),
            "source_start": int(source_start),
            "source_stop": int(source_stop),
            "source_global_label": int(source_global_label),
        }
        if split is not None:
            record["split"] = str(split)
        self.segments.append(record)
        self.num_frames += int(x_np.shape[0])
        return int(x_np.shape[0])

    def summarize_by_split(self) -> dict[str, SegmentSummary]:
        frames: dict[str, int] = defaultdict(int)
        segments: dict[str, int] = defaultdict(int)
        for record in self.segments:
            split = str(record.get("split", "all"))
            frames[split] += int(record["num_frames"])
            segments[split] += 1
        return {
            split: SegmentSummary(frames=frames[split], segments=segments[split])
            for split in sorted(frames.keys())
        }

    def finalize(self, *, extra_meta: dict[str, object]) -> None:
        self._x_file.close()
        self._labels_file.close()

        if self.num_frames == 0:
            raise RuntimeError(f"No samples were written to {self.out_dir}")

        meta = {
            "format": "motion375_memmap_v2",
            "num_frames": int(self.num_frames),
            "feature_dim": 375,
            "x_dtype": "float32",
            "labels_dtype": "int32",
            "has_style_names": True,
            "has_segments": True,
        }
        meta.update(extra_meta)

        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with self.styles_path.open("w", encoding="utf-8") as f:
            json.dump(self.style_names, f, ensure_ascii=False, indent=2)
        with self.segments_path.open("w", encoding="utf-8") as f:
            json.dump(self.segments, f, ensure_ascii=False, indent=2)


def animation_mirror(lrot: np.ndarray, lpos: np.ndarray, names: list[str], parents: np.ndarray):
    joints_mirror = np.array([
        (
            names.index("Left" + n[5:])
            if n.startswith("Right")
            else (
                names.index("Right" + n[4:])
                if n.startswith("Left")
                else names.index(n)
            )
        )
        for n in names
    ])

    mirror_pos = np.array([-1, 1, 1], dtype=np.float32)
    mirror_rot = np.array([[-1, -1, 1], [1, 1, -1], [1, 1, -1]], dtype=np.float32)

    grot, gpos = quat.fk(lrot, lpos, parents)
    gpos_mirror = mirror_pos * gpos[:, joints_mirror]
    grot_mirror = quat.from_xform(mirror_rot * quat.to_xform(grot[:, joints_mirror]))
    return quat.ik(grot_mirror, gpos_mirror, parents)


def convert_zeroeggs_coordinate_system(
    positions: np.ndarray,
    euler_rot: np.ndarray,
    order: str,
) -> tuple[np.ndarray, np.ndarray]:
    positions_out = np.asarray(positions, dtype=np.float32).copy()
    positions_out[..., 0] = -positions_out[..., 0]

    rotations = quat.from_euler(np.radians(euler_rot), order=order)
    xform = quat.to_xform(rotations)
    xform[..., 0, 1] = -xform[..., 0, 1]
    xform[..., 0, 2] = -xform[..., 0, 2]
    xform[..., 1, 0] = -xform[..., 1, 0]
    xform[..., 2, 0] = -xform[..., 2, 0]
    rotations_out = quat.from_xform(xform)
    rotations_out = quat.unroll(rotations_out)

    positions_out *= 0.01
    rotations_out = quat.normalize(rotations_out)
    return rotations_out, positions_out


def extract_features_375(
    rotations: np.ndarray,
    positions: np.ndarray,
    parents: np.ndarray,
    selected: list[int],
    window: int = 60,
) -> np.ndarray:
    if len(positions) < (2 * window + 2):
        return np.empty((0, 375), dtype=np.float32)

    velocities = np.empty_like(positions)
    velocities[1:-1] = (
        0.5 * (positions[2:] - positions[1:-1]) * 60.0
        + 0.5 * (positions[1:-1] - positions[:-2]) * 60.0
    )
    velocities[0] = velocities[1] - (velocities[3] - velocities[2])
    velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])

    angular_velocities = np.zeros_like(positions)
    angular_velocities[1:-1] = (
        0.5
        * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[2:], rotations[1:-1])))
        * 60.0
        + 0.5
        * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[1:-1], rotations[:-2])))
        * 60.0
    )
    angular_velocities[0] = angular_velocities[1] - (angular_velocities[3] - angular_velocities[2])
    angular_velocities[-1] = angular_velocities[-2] + (angular_velocities[-2] - angular_velocities[-3])

    global_rotations, global_positions, global_velocities, global_angular_velocities = quat.fk_vel(
        rotations, positions, velocities, angular_velocities, parents
    )

    direction = np.array([1, 0, 1], dtype=np.float32) * quat.mul_vec(
        global_rotations[:, 0:1], np.array([0.0, 0.0, 1.0], dtype=np.float32)
    )
    direction = direction / np.sqrt(np.sum(np.square(direction), axis=-1))[..., np.newaxis]
    direction = signal.savgol_filter(direction, 61, 3, axis=0, mode="interp")
    direction = direction / np.sqrt(np.sum(np.square(direction), axis=-1))[..., np.newaxis]

    root_rotation = quat.normalize(quat.between(np.array([0, 0, 1], dtype=np.float32), direction))

    root_velocity = np.array([1, 0, 1], dtype=np.float32) * quat.inv_mul_vec(
        root_rotation, global_velocities[:, 0:1]
    )
    root_angular_velocity = np.degrees(quat.inv_mul_vec(root_rotation, global_angular_velocities[:, 0:1]))

    local_positions = global_positions.copy()
    local_positions[:, :, 0] = local_positions[:, :, 0] - local_positions[:, 0:1, 0]
    local_positions[:, :, 2] = local_positions[:, :, 2] - local_positions[:, 0:1, 2]
    local_positions = quat.inv_mul_vec(root_rotation, local_positions)

    local_rotations = quat.inv_mul(root_rotation, global_rotations)
    local_xy = quat.to_xform_xy(local_rotations)
    local_velocities = quat.inv_mul_vec(root_rotation, global_velocities)

    x_chunks = []
    for i in range(window, len(positions) - window - 1):
        root_position = np.array([1, 0, 1], dtype=np.float32) * quat.inv_mul_vec(
            root_rotation[i : i + 1],
            global_positions[i - window : i + window : 10, 0:1] - global_positions[i : i + 1, 0:1],
        )
        root_direction = quat.inv_mul_vec(root_rotation[i : i + 1], direction[i - window : i + window : 10])
        root_vel = quat.inv_mul_vec(
            root_rotation[i : i + 1], global_velocities[i - window : i + window : 10, 0:1]
        )
        speed = np.sqrt(np.sum(np.square(root_vel), axis=-1))

        feat = np.hstack(
            [
                root_position[:, :, 0].ravel(),
                root_position[:, :, 2].ravel(),
                root_direction[:, :, 0].ravel(),
                root_direction[:, :, 2].ravel(),
                root_vel[:, :, 0].ravel(),
                root_vel[:, :, 2].ravel(),
                speed.ravel(),
                root_velocity[i - 1, :, 0].ravel(),
                root_angular_velocity[i - 1, :, 1].ravel(),
                root_velocity[i - 1, :, 2].ravel(),
                local_positions[i - 1, selected].ravel(),
                local_xy[i - 1, selected, :, 0].ravel(),
                local_xy[i - 1, selected, :, 1].ravel(),
                local_velocities[i - 1, selected].ravel(),
            ]
        )
        x_chunks.append(feat)

    if not x_chunks:
        return np.empty((0, 375), dtype=np.float32)

    x = np.asarray(x_chunks, dtype=np.float32)
    if x.shape[1] != 375:
        raise ValueError(
            f"Expected 375 feature dims, got {x.shape[1]}. Check selected joints or skeleton format."
        )
    return x


def infer_suffixes(columns: Iterable[str]) -> list[str]:
    preferred = ["BR", "BW", "FR", "FW", "ID", "SR", "SW", "TR1", "TR2", "TR3"]
    found = []
    col_set = set(columns)
    for col in columns:
        if col.endswith("_START"):
            suffix = col[:-6]
            if f"{suffix}_STOP" in col_set:
                found.append(suffix)

    found_unique = []
    seen = set()
    for suffix in found:
        if suffix not in seen:
            found_unique.append(suffix)
            seen.add(suffix)

    ordered = [suffix for suffix in preferred if suffix in seen]
    ordered.extend(sorted([suffix for suffix in found_unique if suffix not in preferred]))
    return ordered


def resolve_bvh_path(data_root: Path, style_name: str, suffix: str) -> Path | None:
    candidates = [
        data_root / style_name / f"{style_name}_{suffix}.bvh",
        data_root / f"{style_name}_{suffix}.bvh",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _estimate_num_rows(start: int, stop: int, window: int) -> int:
    clip_len = max(int(stop) - int(start), 0)
    return max(clip_len - (2 * int(window) + 1), 0)


def build_entries(frame_cuts_csv: Path, data_root: Path, suffixes: list[str] | None, window: int):
    df = pd.read_csv(frame_cuts_csv)

    if "STYLE_NAME" not in df.columns:
        raise ValueError(f"STYLE_NAME column not found in {frame_cuts_csv}")

    if suffixes is None:
        suffixes = infer_suffixes(df.columns)

    style_names = df["STYLE_NAME"].dropna().astype(str).str.strip().tolist()
    style_names = list(dict.fromkeys(style_names))
    style_to_global_label = {name: idx for idx, name in enumerate(style_names)}

    entries = []
    missing = []
    entry_id = 0

    for _, row in df.iterrows():
        style_name = str(row["STYLE_NAME"]).strip()
        if not style_name:
            continue

        for suffix in suffixes:
            s_col = f"{suffix}_START"
            e_col = f"{suffix}_STOP"
            if s_col not in df.columns or e_col not in df.columns:
                continue

            start = row[s_col]
            stop = row[e_col]
            if pd.isna(start) or pd.isna(stop):
                continue

            bvh_path = resolve_bvh_path(data_root, style_name, suffix)
            if bvh_path is None:
                missing.append(f"{style_name}_{suffix}.bvh")
                continue

            start_i = int(start)
            stop_i = int(stop)
            entries.append(
                {
                    "entry_id": entry_id,
                    "path": bvh_path,
                    "style": style_name,
                    "global_label": style_to_global_label[style_name],
                    "suffix": suffix,
                    "start": start_i,
                    "stop": stop_i,
                    "estimated_rows": _estimate_num_rows(start_i, stop_i, window),
                }
            )
            entry_id += 1

    return entries, style_names, missing, suffixes


def _validate_split_ratios(val_ratio: float, test_ratio: float) -> None:
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0,1), got {val_ratio}")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in [0,1), got {test_ratio}")
    if (val_ratio + test_ratio) >= 1.0:
        raise ValueError(f"val_ratio + test_ratio must be < 1, got {val_ratio + test_ratio}")


def _resolve_split_counts(num_entries: int, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    if num_entries < 1:
        return 0, 0

    n_val = int(round(num_entries * val_ratio)) if val_ratio > 0.0 else 0
    n_test = int(round(num_entries * test_ratio)) if test_ratio > 0.0 else 0

    if val_ratio > 0.0:
        n_val = max(1, n_val)
    if test_ratio > 0.0:
        n_test = max(1, n_test)

    max_heldout = max(num_entries - 1, 0)
    while (n_val + n_test) > max_heldout:
        if n_test > 0 and (n_val == 0 or test_ratio <= val_ratio):
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break
    return n_val, n_test


def _assign_style_splits(
    entries: list[dict[str, object]],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[int, str]:
    if not entries:
        return {}

    n_val, n_test = _resolve_split_counts(len(entries), val_ratio, test_ratio)
    remaining = {
        "train": len(entries) - n_val - n_test,
        "val": n_val,
        "test": n_test,
    }

    total_frames = float(sum(int(entry["estimated_rows"]) for entry in entries))
    targets = {
        "val": total_frames * float(val_ratio) if n_val > 0 else 0.0,
        "test": total_frames * float(test_ratio) if n_test > 0 else 0.0,
    }
    targets["train"] = total_frames - targets["val"] - targets["test"]
    assigned_frames = {"train": 0.0, "val": 0.0, "test": 0.0}

    rng = np.random.default_rng(seed)
    randomized = list(entries)
    rng.shuffle(randomized)
    randomized.sort(key=lambda item: int(item["estimated_rows"]), reverse=True)

    split_priority = {"val": 2, "test": 1, "train": 0}
    assignment: dict[int, str] = {}
    for entry in randomized:
        eligible = [split for split in ("val", "test", "train") if remaining[split] > 0]
        if not eligible:
            raise RuntimeError("No eligible split remained during stratified assignment")

        best_split = max(
            eligible,
            key=lambda split: (
                targets[split] - assigned_frames[split],
                split_priority[split],
                remaining[split],
                -assigned_frames[split],
            ),
        )
        assignment[int(entry["entry_id"])] = best_split
        remaining[best_split] -= 1
        assigned_frames[best_split] += float(int(entry["estimated_rows"]))

    return assignment


def assign_entry_splits(
    entries: list[dict[str, object]],
    *,
    style_names: list[str],
    zero_shot_style_count: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[dict[int, str], list[str], list[str]]:
    if zero_shot_style_count < 0:
        raise ValueError(f"zero_shot_style_count must be >= 0, got {zero_shot_style_count}")
    if zero_shot_style_count >= len(style_names):
        raise ValueError(
            f"zero_shot_style_count must be smaller than number of styles ({len(style_names)}), "
            f"got {zero_shot_style_count}"
        )

    zero_shot_style_names = style_names[-zero_shot_style_count:] if zero_shot_style_count > 0 else []
    id_style_names = style_names[:-zero_shot_style_count] if zero_shot_style_count > 0 else list(style_names)
    zero_shot_style_set = set(zero_shot_style_names)

    id_entries_by_style: dict[str, list[dict[str, object]]] = defaultdict(list)
    for entry in entries:
        style_name = str(entry["style"])
        if style_name in zero_shot_style_set:
            continue
        if int(entry["estimated_rows"]) <= 0:
            continue
        id_entries_by_style[style_name].append(entry)

    assignments: dict[int, str] = {}
    for style_index, style_name in enumerate(id_style_names):
        style_entries = id_entries_by_style.get(style_name, [])
        style_seed = int(seed) + style_index * 1009
        assignments.update(
            _assign_style_splits(
                style_entries,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=style_seed,
            )
        )

    return assignments, id_style_names, zero_shot_style_names


def parse_args() -> argparse.Namespace:
    default_data_root = REPO_ROOT / "data" / "100sty"

    parser = argparse.ArgumentParser(description="Generate memmap-only 100STYLE 375-dim databases")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help="100STYLE root directory (default: repo/data/100sty)",
    )
    parser.add_argument(
        "--frame-cuts",
        type=Path,
        default=None,
        help="Frame_Cuts.csv path (default: <data-root>/Frame_Cuts.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "database_100sty_375_memmap",
        help="Output memmap directory for in-distribution styles",
    )
    parser.add_argument(
        "--zero-shot-output",
        type=Path,
        default=None,
        help="Output memmap directory for held-out zero-shot styles (default: <output>_zeroshot)",
    )
    parser.add_argument(
        "--zero-shot-style-count",
        type=int,
        default=10,
        help="Hold out the last K styles as zero-shot data",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--suffixes",
        type=str,
        default="",
        help="Comma-separated motion suffixes, e.g. BR,BW,FR,FW,ID,SR,SW,TR1,TR2,TR3",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable mirrored augmentation for the in-distribution database",
    )
    parser.add_argument(
        "--mirror-zero-shot",
        action="store_true",
        help="Also mirror the held-out zero-shot database",
    )
    parser.add_argument(
        "--disable-coordinate-convert",
        action="store_true",
        help="Skip the ZeroEGGS coordinate-system conversion before feature extraction",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Context window size for trajectory feature extraction",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=0,
        help="Only process first N valid entries (0 means all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing memmap outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_split_ratios(val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    data_root = args.data_root.expanduser().resolve()
    frame_cuts = args.frame_cuts.expanduser().resolve() if args.frame_cuts else (data_root / "Frame_Cuts.csv")
    output = args.output.expanduser().resolve()
    zero_shot_output = (
        args.zero_shot_output.expanduser().resolve()
        if args.zero_shot_output is not None
        else output.parent / f"{output.name}_zeroshot"
    )

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not frame_cuts.exists():
        raise FileNotFoundError(f"Frame_Cuts.csv not found: {frame_cuts}")
    if args.window < 1:
        raise ValueError(f"window must be >= 1, got {args.window}")
    if zero_shot_output == output:
        raise ValueError("--zero-shot-output must be different from --output")

    suffixes = [s.strip() for s in args.suffixes.split(",") if s.strip()] if args.suffixes else None
    entries, style_names, missing_files, used_suffixes = build_entries(frame_cuts, data_root, suffixes, args.window)
    if args.max_entries and args.max_entries > 0:
        entries = entries[: args.max_entries]

    split_assignment, id_style_names, zero_shot_style_names = assign_entry_splits(
        entries,
        style_names=style_names,
        zero_shot_style_count=args.zero_shot_style_count,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
    )
    zero_shot_style_set = set(zero_shot_style_names)
    id_style_to_label = {name: idx for idx, name in enumerate(id_style_names)}
    zero_shot_style_to_label = {name: idx for idx, name in enumerate(zero_shot_style_names)}

    print(f"[Info] Data root         : {data_root}")
    print(f"[Info] Frame cuts        : {frame_cuts}")
    print(f"[Info] In-dist output    : {output}")
    print(f"[Info] Zero-shot output  : {zero_shot_output}")
    print(f"[Info] Styles total      : {len(style_names)}")
    print(f"[Info] In-dist styles    : {len(id_style_names)}")
    print(f"[Info] Zero-shot styles  : {len(zero_shot_style_names)}")
    print(f"[Info] Zero-shot tail    : {zero_shot_style_names}")
    print(f"[Info] Suffixes          : {used_suffixes}")
    print(f"[Info] Valid entries     : {len(entries)}")
    print(f"[Info] Split ratio       : val={args.val_ratio} test={args.test_ratio}")
    print(f"[Info] Coord convert     : {not args.disable_coordinate_convert}")
    if missing_files:
        print(f"[Warn] Missing BVH files referenced by CSV: {len(missing_files)}")
        print("[Warn] First 10 missing:", missing_files[:10])

    selected = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 35, 36, 37, 38, 61, 62, 63, 64, 68, 69, 70, 71]
    id_writer = MemmapWriter(out_dir=output, style_names=id_style_names, overwrite=args.overwrite)
    zero_writer = (
        MemmapWriter(out_dir=zero_shot_output, style_names=zero_shot_style_names, overwrite=args.overwrite)
        if zero_shot_style_names
        else None
    )

    id_mirror_flags = [False] if args.no_mirror else [False, True]
    zero_mirror_flags = [False, True] if args.mirror_zero_shot else [False]

    skipped_short = 0
    skipped_zero_features = 0

    for idx, entry in enumerate(entries, 1):
        path = Path(entry["path"])
        style_name = str(entry["style"])
        global_label = int(entry["global_label"])
        start = int(entry["start"])
        stop = int(entry["stop"])
        suffix = str(entry["suffix"])
        entry_id = int(entry["entry_id"])

        is_zero_shot = style_name in zero_shot_style_set
        split = None if is_zero_shot else split_assignment.get(entry_id)
        if not is_zero_shot and split is None:
            continue

        tag = "[ZeroShot]" if is_zero_shot else f"[{split}]"
        print(f"[{idx}/{len(entries)}] Loading {path.name} {tag}")

        bvh_data = bvh.load_zeroeggs(str(path))
        parents = bvh_data["parents"]
        positions = bvh_data["positions"][start:stop].copy()
        euler_rot = bvh_data["rotations"][start:stop]

        if len(positions) < (2 * args.window + 2):
            skipped_short += 1
            print(
                f"  [Skip] Too short after cut (len={len(positions)}), requires >= {2 * args.window + 2}"
            )
            continue

        if args.disable_coordinate_convert:
            rotations = quat.from_euler(np.radians(euler_rot), order=bvh_data["order"])
            rotations = quat.unroll(rotations)
            positions *= 0.01
            rotations = quat.normalize(rotations)
        else:
            rotations, positions = convert_zeroeggs_coordinate_system(
                positions=positions,
                euler_rot=euler_rot,
                order=bvh_data["order"],
            )

        writer = zero_writer if is_zero_shot else id_writer
        if writer is None:
            continue
        local_label = zero_shot_style_to_label[style_name] if is_zero_shot else id_style_to_label[style_name]
        mirror_flags = zero_mirror_flags if is_zero_shot else id_mirror_flags

        for mirror in mirror_flags:
            if mirror:
                rotations_cur, positions_cur = animation_mirror(
                    rotations,
                    positions,
                    bvh_data["names"],
                    bvh_data["parents"],
                )
                rotations_cur = quat.unroll(rotations_cur)
                rotations_cur = quat.normalize(rotations_cur)
            else:
                rotations_cur = rotations
                positions_cur = positions

            x = extract_features_375(rotations_cur, positions_cur, parents, selected, window=args.window)
            if x.shape[0] == 0:
                skipped_zero_features += 1
                print("  [Skip] No samples generated from this clip.")
                continue

            written = writer.write_segment(
                x=x,
                label=local_label,
                style_name=style_name,
                split=split,
                source_entry_id=entry_id,
                source_path=path,
                suffix=suffix,
                mirror=mirror,
                source_start=start,
                source_stop=stop,
                source_global_label=global_label,
            )
            mirror_tag = " (Mirrored)" if mirror else ""
            print(f"  [Write] rows={written}{mirror_tag}")

    id_summary = {
        split: {
            "frames": summary.frames,
            "segments": summary.segments,
        }
        for split, summary in id_writer.summarize_by_split().items()
    }
    zero_summary = (
        {
            split: {
                "frames": summary.frames,
                "segments": summary.segments,
            }
            for split, summary in zero_writer.summarize_by_split().items()
        }
        if zero_writer is not None and zero_writer.num_frames > 0
        else {}
    )

    id_writer.finalize(
        extra_meta={
            "dataset_role": "in_distribution",
            "source": "generate_database_100sty.py",
            "frame_cuts": str(frame_cuts),
            "data_root": str(data_root),
            "window": int(args.window),
            "split_seed": int(args.split_seed),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(args.test_ratio),
            "zero_shot_style_count": int(args.zero_shot_style_count),
            "held_out_style_names": zero_shot_style_names,
            "split_strategy": "clip_level_stratified_by_style_and_frame_count",
            "split_summary": id_summary,
            "mirrored": not args.no_mirror,
        }
    )
    if zero_writer is not None and zero_writer.num_frames > 0:
        zero_writer.finalize(
            extra_meta={
                "dataset_role": "zero_shot_tail_styles",
                "source": "generate_database_100sty.py",
                "frame_cuts": str(frame_cuts),
                "data_root": str(data_root),
                "window": int(args.window),
                "zero_shot_style_count": int(args.zero_shot_style_count),
                "source_in_distribution_db": str(output),
                "mirrored": bool(args.mirror_zero_shot),
                "split_summary": zero_summary,
            }
        )

    print(f"[Info] Skipped short clips : {skipped_short}")
    print(f"[Info] Empty feature clips : {skipped_zero_features}")
    print(f"[Done] In-dist memmap      : {output}")
    if zero_writer is not None and zero_writer.num_frames > 0:
        print(f"[Done] Zero-shot memmap    : {zero_shot_output}")
    else:
        print("[Done] Zero-shot memmap    : skipped (no held-out frames written)")
    print(f"[Done] In-dist summary     : {id_summary}")
    print(f"[Done] Zero-shot summary   : {zero_summary}")


if __name__ == "__main__":
    main()
