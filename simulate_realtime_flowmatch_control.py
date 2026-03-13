"""
Simulate real-time character control with FlowMatchVQVAEController.

Workflow:
  1) Load a source trajectory segment (84-d traj per frame) from DB.
  2) Use the trained flow-matching controller to autoregressively generate motion.
  3) Export generated/reference 375-d sequences into HDF5 files.
  4) Show side-by-side real-time visualization, defaulting to GenoView.

Example:
  python NewCode/simulate_realtime_flowmatch_control.py \
    --db NewCode/database_100sty_375_memmap \
    --flow-ckpt NewCode/checkpoints/flowmatch_vqvae_controller/best.pt \
    --vqvae-ckpt NewCode/checkpoints/vqvae_375to291_fast_ema/best.pt \
    --num-frames 300 \
    --target-style Zombie \
    --out-dir NewCode/rollout_vis
"""

from __future__ import annotations

import argparse
import json
import secrets
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

from Utils import bvh
from arch import FlowMatchVQVAEController
from genoview_realtime_vis import visualize_dual_genoview_from_375
from model_loaders import build_vqvae_from_checkpoint
from quick_visualize_100sty_375 import SELECTED_24, build_reduced_parents, first_bvh_under
from realtime_utils import (
    choose_segment,
    load_db_arrays,
    load_norm_stats,
    parse_style_selector,
)
from runtime_utils import (
    load_checkpoint,
    resolve_device,
    sanitize_state_dict_keys,
    set_seed,
)
from simulate_realtime_baselines import visualize_side_by_side_from_375


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def _sync_device_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize_rollout_benchmark(
    *,
    frame_latencies_sec: np.ndarray,
    warmup_frames: int,
    target_fps: float,
) -> dict[str, float | int]:
    if frame_latencies_sec.ndim != 1:
        raise ValueError(f"Expected frame_latencies_sec rank 1, got {frame_latencies_sec.shape}")
    if frame_latencies_sec.size == 0:
        raise ValueError("No frame latencies collected for benchmark")
    if warmup_frames < 0:
        raise ValueError(f"warmup_frames must be >= 0, got {warmup_frames}")
    if target_fps <= 0.0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")

    warmup = min(int(warmup_frames), int(frame_latencies_sec.size))
    measured = frame_latencies_sec[warmup:]
    if measured.size == 0:
        raise ValueError(
            f"warmup_frames={warmup_frames} leaves no measured frames; collected={frame_latencies_sec.size}"
        )

    lat_ms = measured * 1000.0
    fps_inst = 1.0 / np.maximum(measured, 1e-12)
    total_sec = float(measured.sum())
    target_frame_ms = 1000.0 / float(target_fps)
    over_budget = lat_ms > target_frame_ms

    return {
        "warmup_frames": int(warmup),
        "measured_frames": int(measured.size),
        "target_fps": float(target_fps),
        "target_frame_ms": float(target_frame_ms),
        "latency_ms_mean": float(lat_ms.mean()),
        "latency_ms_std": float(lat_ms.std()),
        "latency_ms_p50": float(np.percentile(lat_ms, 50)),
        "latency_ms_p95": float(np.percentile(lat_ms, 95)),
        "latency_ms_p99": float(np.percentile(lat_ms, 99)),
        "latency_ms_min": float(lat_ms.min()),
        "latency_ms_max": float(lat_ms.max()),
        "fps_mean_effective": float(measured.size / max(total_sec, 1e-12)),
        "fps_mean_instant": float(fps_inst.mean()),
        "fps_std_instant": float(fps_inst.std()),
        "fps_p05": float(np.percentile(fps_inst, 5)),
        "fps_p50": float(np.percentile(fps_inst, 50)),
        "fps_p95": float(np.percentile(fps_inst, 95)),
        "fps_cv_instant": float(fps_inst.std() / max(fps_inst.mean(), 1e-12)),
        "over_budget_ratio": float(over_budget.mean()),
    }


def _print_benchmark_summary(summary: dict[str, float | int]) -> None:
    print(
        "[Benchmark] "
        f"frames={summary['measured_frames']} warmup={summary['warmup_frames']} "
        f"lat_mean={summary['latency_ms_mean']:.3f}ms "
        f"lat_p95={summary['latency_ms_p95']:.3f}ms "
        f"lat_p99={summary['latency_ms_p99']:.3f}ms "
        f"fps_eff={summary['fps_mean_effective']:.2f} "
        f"fps_p05={summary['fps_p05']:.2f} "
        f"fps_p95={summary['fps_p95']:.2f} "
        f"fps_cv={summary['fps_cv_instant']:.4f} "
        f"over_budget={100.0 * float(summary['over_budget_ratio']):.2f}%"
    )


def _resolve_path_candidate(path_like: str | Path) -> Path:
    return Path(path_like).expanduser()


def _resolve_existing_path(
    path_like: str | Path,
    *,
    label: str,
    search_roots: tuple[Path, ...] = (),
) -> Path:
    raw = _resolve_path_candidate(path_like)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((Path.cwd() / raw))
        for root in search_roots:
            candidates.append(root / raw)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        f"{label} not found: {path_like} (searched: {', '.join(str(c.resolve()) for c in candidates)})"
    )


def _resolve_vqvae_ckpt_path(
    explicit_path: Path | None,
    *,
    flow_args: dict,
    flow_ckpt_path: Path,
) -> Path:
    search_roots = (flow_ckpt_path.parent, SCRIPT_DIR, REPO_ROOT)
    if explicit_path is not None:
        return _resolve_existing_path(explicit_path, label="--vqvae-ckpt", search_roots=search_roots)

    vqvae_ckpt_raw = flow_args.get("vqvae_ckpt", None)
    if vqvae_ckpt_raw is None:
        raise ValueError("--vqvae-ckpt is required because flow checkpoint args has no vqvae_ckpt")
    return _resolve_existing_path(vqvae_ckpt_raw, label="vqvae_ckpt from flow checkpoint args", search_roots=search_roots)


def _resolve_norm_stats_path(
    explicit_path: Path | None,
    *,
    flow_args: dict,
    flow_ckpt_path: Path,
    vqvae_ckpt_path: Path,
) -> Path:
    search_roots = (flow_ckpt_path.parent, vqvae_ckpt_path.parent, SCRIPT_DIR, REPO_ROOT)
    if explicit_path is not None:
        return _resolve_existing_path(explicit_path, label="--norm-stats", search_roots=search_roots)

    flow_arg_norm = flow_args.get("norm_stats", None)
    if flow_arg_norm is not None:
        try:
            return _resolve_existing_path(
                flow_arg_norm,
                label="norm_stats from flow checkpoint args",
                search_roots=search_roots,
            )
        except FileNotFoundError:
            pass

    flow_sibling = flow_ckpt_path.parent / "norm_stats.npz"
    if flow_sibling.exists():
        return flow_sibling.resolve()

    vqvae_sibling = vqvae_ckpt_path.parent / "norm_stats.npz"
    if vqvae_sibling.exists():
        return vqvae_sibling.resolve()

    raise FileNotFoundError(
        "Could not resolve norm stats. Tried flow checkpoint args, "
        f"{flow_sibling}, and {vqvae_sibling}."
    )


def _resolve_solver(explicit_solver: Optional[str], *, flow_args: dict) -> tuple[str, str]:
    if explicit_solver is not None:
        return explicit_solver, "cli"
    return str(flow_args.get("solver", "euler")), "checkpoint"


def _resolve_ode_steps(explicit_steps: Optional[int], *, flow_args: dict) -> tuple[int, str]:
    if explicit_steps is not None:
        return int(explicit_steps), "cli"
    ckpt_solver_steps = flow_args.get("solver_steps", flow_args.get("recon_solver_steps", None))
    if ckpt_solver_steps is not None:
        return int(ckpt_solver_steps), "checkpoint"
    return 16, "fallback"


def _resolve_seed(explicit_seed: Optional[int]) -> tuple[int, str]:
    if explicit_seed is not None:
        return int(explicit_seed), "cli"
    return int(secrets.randbelow(2**31)), "random"


def rollout_sequence(
    *,
    model: FlowMatchVQVAEController,
    x_seed_norm: np.ndarray,
    traj_plan_norm: np.ndarray,
    style_label: int,
    ode_steps: int,
    noise_std: float,
    solver: str,
    device: torch.device,
    benchmark: bool = False,
    warmup_frames: int = 0,
    target_fps: float = 60.0,
) -> tuple[np.ndarray, dict[str, float | int] | None]:
    if x_seed_norm.shape != (375,):
        raise ValueError(f"x_seed_norm must be shape (375,), got {x_seed_norm.shape}")
    if traj_plan_norm.ndim != 2 or traj_plan_norm.shape[1] != 84:
        raise ValueError(f"traj_plan_norm must be shape (T,84), got {traj_plan_norm.shape}")
    if ode_steps < 1:
        raise ValueError(f"ode_steps must be >= 1, got {ode_steps}")
    if noise_std <= 0.0:
        raise ValueError(f"noise_std must be > 0, got {noise_std}")
    if solver not in {"euler", "midpoint", "heun"}:
        raise ValueError(f"Unsupported solver: {solver}")
    if warmup_frames < 0:
        raise ValueError(f"warmup_frames must be >= 0, got {warmup_frames}")
    if target_fps <= 0.0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")

    t_total = traj_plan_norm.shape[0]
    seq = torch.empty((t_total, 375), device=device, dtype=torch.float32)
    x_curr = torch.from_numpy(x_seed_norm.astype(np.float32, copy=False)).to(device=device).unsqueeze(0)
    seq[0] = x_curr[0]
    traj_plan_t = torch.from_numpy(traj_plan_norm.astype(np.float32, copy=False)).to(device=device)
    style_t = torch.tensor([style_label], dtype=torch.long, device=device)
    frame_latencies_sec: list[float] = []
    model.eval()
    with torch.no_grad():
        for i in range(1, t_total):
            traj_next = traj_plan_t[i : i + 1]
            if benchmark:
                _sync_device_if_needed(device)
                t0 = time.perf_counter()
            motion_hat, _ = model.predict_next_motion(
                x_curr=x_curr,
                traj_next=traj_next,
                style_label=style_t,
                num_steps=ode_steps,
                noise_std=noise_std,
                solver=solver,
            )
            if benchmark:
                _sync_device_if_needed(device)
                frame_latencies_sec.append(time.perf_counter() - t0)
            x_next = torch.cat((traj_next, motion_hat), dim=-1)
            seq[i] = x_next[0]
            x_curr = x_next

    benchmark_summary = None
    if benchmark:
        benchmark_summary = _summarize_rollout_benchmark(
            frame_latencies_sec=np.asarray(frame_latencies_sec, dtype=np.float64),
            warmup_frames=warmup_frames,
            target_fps=target_fps,
        )
    return seq.detach().cpu().numpy().astype(np.float32, copy=False), benchmark_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate real-time control with FlowMatchVQVAEController")
    parser.add_argument(
        "--db",
        type=Path,
        default=SCRIPT_DIR / "database_100sty_375_memmap",
        help="Input DB path (.h5/.hdf5 or memmap directory)",
    )
    parser.add_argument(
        "--flow-ckpt",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "flowmatch_vqvae_controller" / "best.pt",
        help="Trained flow matching checkpoint",
    )
    parser.add_argument(
        "--vqvae-ckpt",
        type=Path,
        default=None,
        help="VQ-VAE checkpoint (optional; default tries to read from flow checkpoint args)",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=None,
        help="Normalization stats npz with mean/std. Default: sibling of flow ckpt (norm_stats.npz)",
    )
    parser.add_argument("--num-frames", type=int, default=300, help="Generated rollout length")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for segment sampling and latent noise. Default: generate a fresh random seed each run.",
    )
    parser.add_argument(
        "--source-style",
        type=str,
        default="",
        help="Style selector for source trajectory segment (name or label id)",
    )
    parser.add_argument(
        "--target-style",
        type=str,
        default="",
        help="Target style for generation (name or label id). Empty means same as source style.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Optional fixed start index in DB (must keep same label for num_frames)",
    )
    parser.add_argument(
        "--ode-steps",
        type=int,
        default=None,
        help="ODE integration steps for latent rollout. Default: inherit solver_steps from flow checkpoint args.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["euler", "midpoint", "heun"],
        default=None,
        help="ODE solver used for rollout. Default: inherit solver from flow checkpoint args.",
    )
    parser.add_argument("--noise-std", type=float, default=1.0, help="Initial latent noise std")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Measure per-frame rollout latency and report tail latency / FPS stability",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=5,
        help="Number of initial predicted frames excluded from latency summary when --benchmark is enabled",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "rollout_vis",
        help="Directory for generated h5 and visualization outputs",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="flowmatch_realtime_300",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Show side-by-side reference/prediction visualization after rollout (default true)",
    )
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Skip quick visualization call",
    )
    parser.add_argument("--fps", type=int, default=60, help="Visualization fps")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0, help="Visualization dt")
    parser.add_argument("--local-only", action="store_true", help="Visualize local positions only")
    parser.add_argument(
        "--vis-backend",
        type=str,
        choices=["genoview", "matplotlib"],
        default="genoview",
        help="Visualization backend. Default uses GenoView; Matplotlib is kept as a fallback.",
    )
    parser.add_argument(
        "--geno-gap",
        type=float,
        default=2.4,
        help="Horizontal spacing between the two GenoView characters.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "100sty",
        help="100STYLE BVH root used for side-by-side visualization",
    )
    parser.add_argument(
        "--bvh-ref",
        type=Path,
        default=None,
        help="Optional BVH ref used for side-by-side visualization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.db = args.db.expanduser().resolve()
    args.flow_ckpt = _resolve_existing_path(args.flow_ckpt, label="--flow-ckpt", search_roots=(SCRIPT_DIR, REPO_ROOT))

    flow_ckpt = load_checkpoint(args.flow_ckpt, map_location="cpu")
    flow_args = flow_ckpt.get("args", {}) or {}

    args.vqvae_ckpt = _resolve_vqvae_ckpt_path(args.vqvae_ckpt, flow_args=flow_args, flow_ckpt_path=args.flow_ckpt)
    args.norm_stats = _resolve_norm_stats_path(
        args.norm_stats,
        flow_args=flow_args,
        flow_ckpt_path=args.flow_ckpt,
        vqvae_ckpt_path=args.vqvae_ckpt,
    )

    args.seed, seed_source = _resolve_seed(args.seed)
    args.solver, solver_source = _resolve_solver(args.solver, flow_args=flow_args)
    args.ode_steps, ode_source = _resolve_ode_steps(args.ode_steps, flow_args=flow_args)

    if args.num_frames < 2:
        raise ValueError("--num-frames must be >= 2")
    if args.solver not in {"euler", "midpoint", "heun"}:
        raise ValueError(f"--solver must be one of euler/midpoint/heun, got {args.solver!r}")
    if args.ode_steps < 1:
        raise ValueError("--ode-steps must be >= 1")
    if args.noise_std <= 0.0:
        raise ValueError("--noise-std must be > 0")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be >= 0")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.dt <= 0.0:
        raise ValueError("--dt must be > 0")

    set_seed(args.seed)

    device = resolve_device(args.device)
    print(f"[Info] device          : {device}")
    print(f"[Info] flow ckpt       : {args.flow_ckpt}")
    print(f"[Info] vqvae ckpt      : {args.vqvae_ckpt}")
    print(f"[Info] norm stats      : {args.norm_stats}")
    print(f"[Info] seed            : {args.seed} ({seed_source})")
    print(f"[Info] solver          : {args.solver} ({solver_source})")
    print(f"[Info] ode steps       : {args.ode_steps} ({ode_source})")
    print(f"[Info] benchmark       : {args.benchmark}")
    if args.benchmark:
        print(f"[Info] warmup frames   : {args.warmup_frames}")
    print(f"[Info] db              : {args.db}")

    x_source, labels, style_names, backend = load_db_arrays(args.db)
    if labels.size == 0:
        raise RuntimeError("DB is empty")
    print(f"[Info] db backend      : {backend}")
    print(f"[Info] db frames       : {labels.shape[0]}")
    print(f"[Info] num styles      : {len(style_names)}")

    src_style_label = parse_style_selector(args.source_style, style_names)
    start, end, src_label, src_run = choose_segment(
        labels=labels,
        num_frames=args.num_frames,
        seed=args.seed,
        target_label=src_style_label,
        start_index=args.start_index,
    )
    target_style_label = parse_style_selector(args.target_style, style_names)
    if target_style_label is None:
        target_style_label = src_label

    if target_style_label < 0 or target_style_label >= len(style_names):
        raise ValueError(
            f"target style label out of range: {target_style_label} vs [0, {len(style_names)-1}]"
        )

    x_ref_raw = np.asarray(x_source[start:end], dtype=np.float32)
    if x_ref_raw.shape != (args.num_frames, 375):
        raise RuntimeError(
            f"Unexpected sampled segment shape {x_ref_raw.shape}, expected {(args.num_frames, 375)}"
        )

    norm_mean, norm_std = load_norm_stats(args.norm_stats)
    x_ref_norm = (x_ref_raw - norm_mean[None, :]) / norm_std[None, :]

    vqvae = build_vqvae_from_checkpoint(args.vqvae_ckpt).to(device)

    num_styles = int(flow_args.get("num_styles", 0))
    if num_styles <= 0:
        num_styles = len(style_names)
    if target_style_label >= num_styles:
        raise ValueError(
            f"target style {target_style_label} exceeds model num_styles={num_styles}. "
            "Use matching style set or retrain with larger num_styles."
        )

    model = FlowMatchVQVAEController(
        vqvae=vqvae,
        num_styles=num_styles,
        content_dim=int(flow_args.get("content_dim", 256)),
        style_dim=int(flow_args.get("style_dim", 128)),
        time_embed_dim=int(flow_args.get("time_embed_dim", 64)),
        encoder_hidden_dim=int(flow_args.get("encoder_hidden_dim", 512)),
        flow_hidden_dim=int(flow_args.get("flow_hidden_dim", 512)),
        flow_layers=int(flow_args.get("flow_layers", 4)),
        dropout=float(flow_args.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(sanitize_state_dict_keys(flow_ckpt["model"]), strict=True)
    model.eval()

    x_gen_norm, benchmark_summary = rollout_sequence(
        model=model,
        x_seed_norm=x_ref_norm[0],
        traj_plan_norm=x_ref_norm[:, :84],
        style_label=target_style_label,
        ode_steps=args.ode_steps,
        noise_std=args.noise_std,
        solver=args.solver,
        device=device,
        benchmark=bool(args.benchmark),
        warmup_frames=args.warmup_frames,
        target_fps=float(args.fps),
    )
    x_gen_raw = x_gen_norm * norm_std[None, :] + norm_mean[None, :]
    x_gen_raw = np.asarray(x_gen_raw, dtype=np.float32)

    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gen_h5_path = args.out_dir / f"{args.prefix}.generated.hdf5"
    ref_h5_path = args.out_dir / f"{args.prefix}.reference.hdf5"
    with h5py.File(gen_h5_path, "w") as f:
        f.create_dataset("X", data=x_gen_raw, dtype=np.float32)
        f.create_dataset("Labels", data=np.full((args.num_frames,), target_style_label, dtype=np.int32))
        style_names_np = np.asarray(style_names, dtype=h5py.string_dtype(encoding="utf-8"))
        f.create_dataset("StyleNames", data=style_names_np)
    with h5py.File(ref_h5_path, "w") as f:
        f.create_dataset("X", data=x_ref_raw, dtype=np.float32)
        f.create_dataset("Labels", data=np.full((args.num_frames,), src_label, dtype=np.int32))
        style_names_np = np.asarray(style_names, dtype=h5py.string_dtype(encoding="utf-8"))
        f.create_dataset("StyleNames", data=style_names_np)

    meta = {
        "flow_ckpt": str(args.flow_ckpt),
        "vqvae_ckpt": str(args.vqvae_ckpt),
        "norm_stats": str(args.norm_stats),
        "db": str(args.db),
        "db_backend": backend,
        "segment_start": int(start),
        "segment_end": int(end),
        "segment_source_label": int(src_label),
        "segment_source_style": style_names[src_label] if 0 <= src_label < len(style_names) else f"label_{src_label}",
        "source_run": {"start": int(src_run[0]), "end": int(src_run[1]), "label": int(src_run[2])},
        "target_style_label": int(target_style_label),
        "target_style_name": style_names[target_style_label],
        "num_frames": int(args.num_frames),
        "seed": int(args.seed),
        "seed_source": seed_source,
        "solver": str(args.solver),
        "ode_steps": int(args.ode_steps),
        "noise_std": float(args.noise_std),
        "reference_h5": str(ref_h5_path),
        "generated_h5": str(gen_h5_path),
        "benchmark": benchmark_summary,
    }
    meta_path = args.out_dir / f"{args.prefix}.rollout_meta.json"
    with meta_path.open("w", encoding="utf-8") as fw:
        json.dump(meta, fw, ensure_ascii=False, indent=2)

    print(f"[Info] source segment  : [{start}, {end}) label={src_label} ({meta['segment_source_style']})")
    print(f"[Info] target style    : {target_style_label} ({style_names[target_style_label]})")
    print(f"[Info] generated h5    : {gen_h5_path}")
    print(f"[Info] reference h5    : {ref_h5_path}")
    print(f"[Info] rollout meta    : {meta_path}")
    if benchmark_summary is not None:
        _print_benchmark_summary(benchmark_summary)

    if backend == "hdf5":
        x_source.file.close()

    if args.visualize:
        vis_title = (
            f"FlowMatch compare | src={meta['segment_source_style']} -> tgt={style_names[target_style_label]} | "
            f"seg=[{start},{end})"
        )
        print(f"[Info] visualize backend: {args.vis_backend}")
        print("[Info] visualize mode   : left=reference | right=prediction")
        if args.vis_backend == "genoview":
            visualize_dual_genoview_from_375(
                ref_features_375=x_ref_raw,
                pred_features_375=x_gen_raw,
                fps=args.fps,
                dt=args.dt,
                use_global=not args.local_only,
                root_wy_in_degrees=True,
                title=vis_title,
                model_gap=args.geno_gap,
            )
        else:
            data_root = args.data_root.expanduser().resolve()
            if args.bvh_ref is not None:
                bvh_ref = args.bvh_ref.expanduser().resolve()
            else:
                bvh_ref = first_bvh_under(data_root)
            if not bvh_ref.exists():
                raise FileNotFoundError(f"Reference BVH not found: {bvh_ref}")

            bvh_data = bvh.load_zeroeggs(str(bvh_ref))
            parents_24 = build_reduced_parents(bvh_data["parents"], SELECTED_24)
            print(f"[Info] visualize bvh  : {bvh_ref}")
            visualize_side_by_side_from_375(
                ref_features_375=x_ref_raw,
                pred_features_375=x_gen_raw,
                parents=parents_24,
                fps=args.fps,
                dt=args.dt,
                use_global=not args.local_only,
                root_wy_in_degrees=True,
                trail_len=min(120, args.num_frames),
                title=vis_title,
            )
        print("[Done] Visualization complete.")
    else:
        print("[Done] Rollout complete (visualization skipped).")


if __name__ == "__main__":
    main()
