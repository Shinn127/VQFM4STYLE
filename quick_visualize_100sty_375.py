"""
Quick sanity-check visualization for 100STYLE 375-dim database.

This script randomly samples a contiguous 300-frame segment (default)
from the same style label, reconstructs motion from 375-dim traj+motion
features, and exports:
  - one GIF animation
  - one key-frame PNG
  - one JSON metadata file
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Prefer NewCode/Utils.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from Utils import bvh  # noqa: E402
from Utils.AnimationPlot import animation_plot_from_375  # noqa: E402


# Must match selected joints used in generate_database_100sty.py
SELECTED_24 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 35, 36, 37, 38, 61, 62, 63, 64, 68, 69, 70, 71]


def decode_style_names(raw: np.ndarray) -> List[str]:
    names: List[str] = []
    for item in raw:
        if isinstance(item, (bytes, bytearray)):
            names.append(item.decode("utf-8"))
        else:
            names.append(str(item))
    return names


def contiguous_runs(labels: np.ndarray) -> List[Tuple[int, int, int]]:
    """Return contiguous same-label runs as (start, end_exclusive, label)."""
    runs: List[Tuple[int, int, int]] = []
    if labels.size == 0:
        return runs

    start = 0
    curr = int(labels[0])
    for i in range(1, len(labels)):
        v = int(labels[i])
        if v != curr:
            runs.append((start, i, curr))
            start = i
            curr = v
    runs.append((start, len(labels), curr))
    return runs


def parse_style_selector(
    selector: Optional[str],
    style_names: Sequence[str],
) -> Optional[int]:
    """
    Parse style selector.
    - None -> no filtering
    - integer string -> label id
    - style name -> matched by exact name
    """
    if selector is None or selector == "":
        return None

    s = selector.strip()
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        label = int(s)
        if not (0 <= label < len(style_names)):
            raise ValueError(f"Style label out of range: {label}, valid [0, {len(style_names)-1}]")
        return label

    # exact name
    if s in style_names:
        return style_names.index(s)

    # case-insensitive fallback
    lower_map = {name.lower(): idx for idx, name in enumerate(style_names)}
    key = s.lower()
    if key in lower_map:
        return lower_map[key]

    raise ValueError(f"Style '{selector}' not found in StyleNames")


def choose_segment(
    labels: np.ndarray,
    num_frames: int,
    seed: int,
    target_label: Optional[int] = None,
) -> Tuple[int, int, int, Tuple[int, int, int]]:
    """
    Choose [start, end) segment with length=num_frames.

    Returns:
      start, end, label, source_run
    """
    rng = random.Random(seed)
    runs = contiguous_runs(labels)

    candidates = []
    for run in runs:
        rs, re, lb = run
        if re - rs < num_frames:
            continue
        if target_label is not None and lb != target_label:
            continue
        candidates.append(run)

    if not candidates:
        if target_label is None:
            detail = "no run long enough"
        else:
            detail = f"no run long enough for label {target_label}"
        max_len = max((re - rs for rs, re, _ in runs), default=0)
        raise RuntimeError(f"Cannot sample {num_frames} frames: {detail}. Max run length={max_len}")

    rs, re, lb = rng.choice(candidates)
    start = rng.randint(rs, re - num_frames)
    end = start + num_frames
    return start, end, lb, (rs, re, lb)


def first_bvh_under(root: Path) -> Path:
    files = sorted(root.rglob("*.bvh"))
    if not files:
        raise FileNotFoundError(f"No .bvh files found under: {root}")
    return files[0]


def build_reduced_parents(full_parents: np.ndarray, selected: List[int]) -> List[int]:
    selected_set = set(selected)
    remap = {j: i for i, j in enumerate(selected)}

    reduced: List[int] = []
    for j in selected:
        p = int(full_parents[j])
        while p != -1 and p not in selected_set:
            p = int(full_parents[p])
        reduced.append(-1 if p == -1 else remap[p])
    return reduced


def ensure_output_paths(out_dir: Path, prefix: str) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / f"{prefix}.gif"
    png_path = out_dir / f"{prefix}.png"
    json_path = out_dir / f"{prefix}.json"
    return gif_path, png_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Random 300-frame quick visualization for 100STYLE 375-dim DB")
    parser.add_argument(
        "--db",
        type=Path,
        default=SCRIPT_DIR / "database_100sty_375_with_labels.hdf5",
        help="Path to database_100sty_375_with_labels.hdf5",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=300,
        help="Number of contiguous frames to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=242,
        help="Random seed",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="",
        help="Optional style selector: style name (e.g., Zombie) or label id (e.g., 99)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Output GIF fps",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0 / 60.0,
        help="Frame dt for root integration",
    )
    parser.add_argument(
        "--root-wy-deg",
        action="store_true",
        default=True,
        help="Treat wy as degrees/sec when reconstructing root yaw (default: true)",
    )
    parser.add_argument(
        "--root-wy-rad",
        action="store_true",
        help="Treat wy as radians/sec when reconstructing root yaw",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Visualize local positions only (skip global root reconstruction)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "100sty",
        help="Root path to 100STYLE BVH files",
    )
    parser.add_argument(
        "--bvh-ref",
        type=Path,
        default=None,
        help="Optional reference BVH for deriving 24-joint parent topology",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Output directory for gif/png/json",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="quick_vis_100sty_300",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show animation directly in Matplotlib window (no save required)",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Do not export GIF",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Do not export key-frame PNG",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not export metadata JSON",
    )
    args = parser.parse_args()

    if args.num_frames <= 1:
        raise ValueError("--num-frames must be > 1")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    db_path = args.db.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    need_gif = not args.no_gif
    need_png = not args.no_png
    need_json = not args.no_json
    need_show = args.show

    if not (need_gif or need_png or need_json or need_show):
        raise ValueError("Nothing to do. Enable at least one of: GIF/PNG/JSON output or --show")

    gif_path = png_path = json_path = None
    if need_gif or need_png or need_json:
        gif_path, png_path, json_path = ensure_output_paths(out_dir, args.prefix)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with h5py.File(db_path, "r") as f:
        if "X" not in f or "Labels" not in f:
            raise KeyError("HDF5 must contain datasets 'X' and 'Labels'")
        X = f["X"]
        Labels = f["Labels"]

        if X.ndim != 2 or X.shape[1] != 375:
            raise ValueError(f"Expected X shape (N, 375), got {X.shape}")
        if Labels.ndim != 1 or Labels.shape[0] != X.shape[0]:
            raise ValueError(f"Labels shape {Labels.shape} incompatible with X shape {X.shape}")

        if "StyleNames" in f:
            style_names = decode_style_names(f["StyleNames"][:])
        else:
            max_label = int(np.max(Labels[:])) if Labels.shape[0] else -1
            style_names = [f"label_{i}" for i in range(max_label + 1)]

        target_label = parse_style_selector(args.style, style_names)

        labels_np = Labels[:]
        start, end, sampled_label, src_run = choose_segment(
            labels_np,
            num_frames=args.num_frames,
            seed=args.seed,
            target_label=target_label,
        )
        x_seg = X[start:end]

    if args.bvh_ref is not None:
        bvh_ref = args.bvh_ref.expanduser().resolve()
    else:
        bvh_ref = first_bvh_under(data_root)
    if not bvh_ref.exists():
        raise FileNotFoundError(f"Reference BVH not found: {bvh_ref}")

    bvh_data = bvh.load_zeroeggs(str(bvh_ref))
    parents_24 = build_reduced_parents(bvh_data["parents"], SELECTED_24)

    style_name = style_names[sampled_label] if 0 <= sampled_label < len(style_names) else f"label_{sampled_label}"
    interval_ms = int(round(1000.0 / args.fps))
    use_deg = False if args.root_wy_rad else True

    title = (
        f"100STYLE quick check | style={style_name} (label={sampled_label}) | "
        f"seg=[{start},{end})"
    )

    print(f"[Info] DB            : {db_path}")
    print(f"[Info] X segment     : {x_seg.shape}")
    print(f"[Info] Selected style: {style_name} ({sampled_label})")
    print(f"[Info] Source run    : start={src_run[0]}, end={src_run[1]}, label={src_run[2]}")
    print(f"[Info] Segment       : start={start}, end={end}")
    print(f"[Info] Ref BVH       : {bvh_ref}")
    outputs = []
    if need_gif:
        outputs.append(str(gif_path))
    if need_png:
        outputs.append(str(png_path))
    if need_json:
        outputs.append(str(json_path))
    if need_show:
        outputs.append("matplotlib window")
    print(f"[Info] Outputs       : {', '.join(outputs)}")

    if need_gif:
        animation_plot_from_375(
            x_seg,
            parents=parents_24,
            interval=interval_ms,
            save_path=str(gif_path),
            title=title,
            dt=args.dt,
            root_wy_in_degrees=use_deg,
            use_global=not args.local_only,
            single_frame=False,
            trail_len=min(120, args.num_frames),
        )

    if need_png:
        animation_plot_from_375(
            x_seg,
            parents=parents_24,
            save_path=str(png_path),
            title=title,
            dt=args.dt,
            root_wy_in_degrees=use_deg,
            use_global=not args.local_only,
            single_frame=True,
            frame_idx=args.num_frames // 2,
        )

    if need_show:
        animation_plot_from_375(
            x_seg,
            parents=parents_24,
            interval=interval_ms,
            save_path=None,
            title=title,
            dt=args.dt,
            root_wy_in_degrees=use_deg,
            use_global=not args.local_only,
            single_frame=False,
            trail_len=min(120, args.num_frames),
        )

    meta = {
        "db": str(db_path),
        "style_label": int(sampled_label),
        "style_name": style_name,
        "source_run": {"start": int(src_run[0]), "end": int(src_run[1]), "label": int(src_run[2])},
        "segment": {"start": int(start), "end": int(end), "num_frames": int(args.num_frames)},
        "fps": int(args.fps),
        "dt": float(args.dt),
        "root_wy_in_degrees": bool(use_deg),
        "use_global": bool(not args.local_only),
        "bvh_ref": str(bvh_ref),
        "gif": str(gif_path),
        "png": str(png_path),
    }
    if need_json:
        with json_path.open("w", encoding="utf-8") as fw:
            json.dump(meta, fw, ensure_ascii=False, indent=2)

    print("[Done] Quick visualization export complete.")


if __name__ == "__main__":
    main()
