"""
Unified visualization test script for VQ-VAE / LMM / MVAE baselines.

Supported models:
  - vqvae
  - lmm
  - mvae

Supported modes:
  - reconstruct
  - rollout

The script will:
  1) sample a contiguous segment from DB (default 300 frames)
  2) run selected model in selected mode
  3) write generated/reference HDF5 files
  4) show side-by-side Matplotlib animation:
     left = reference, right = prediction
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

from Utils import bvh
from arch import LMMCompressorDecompressorAE, MVAE375to291, VQVAE375to291
from model_loaders import (
    build_lmm_from_checkpoint,
    build_mvae_from_checkpoint,
    build_vqvae_from_checkpoint,
)
from quick_visualize_100sty_375 import SELECTED_24, build_reduced_parents, first_bvh_under
from realtime_utils import (
    choose_segment,
    load_db_arrays,
    load_norm_stats,
    parse_style_selector,
)
from runtime_utils import resolve_device, set_seed


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def _decode_375_state(features_375: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features_375, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 375:
        raise ValueError(f"Expected shape (T, 375) or (375,), got {x.shape}")

    state = x[:, 84:]
    root_vel = state[:, :3]

    n_joints = 24
    lp_start = 3
    lp_end = lp_start + n_joints * 3
    local_pos = state[:, lp_start:lp_end].reshape(-1, n_joints, 3)
    return root_vel, local_pos


def _reconstruct_global_positions_from_375(
    features_375: np.ndarray,
    dt: float,
    root_wy_in_degrees: bool,
) -> np.ndarray:
    from Utils import quat as quat_ops

    root_vel, local_pos = _decode_375_state(features_375)
    t_total = local_pos.shape[0]

    root_motion = root_vel * float(dt)
    trans_offset = root_motion.copy()
    trans_offset[:, 1] = 0.0

    yaw_delta = root_motion[:, 1]
    if root_wy_in_degrees:
        yaw_delta = np.radians(yaw_delta)
    yaw_offset = quat_ops.from_angle_axis(yaw_delta, np.array([0.0, 1.0, 0.0], dtype=np.float32))

    root_pos = np.zeros((3,), dtype=np.float32)
    root_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    root_pos_seq = np.zeros((t_total, 3), dtype=np.float32)
    root_rot_seq = np.zeros((t_total, 4), dtype=np.float32)

    for i in range(t_total):
        root_pos += trans_offset[i]
        root_pos_seq[i] = root_pos
        root_rot = quat_ops.mul(yaw_offset[i : i + 1], root_rot[None, :])[0]
        root_rot_seq[i] = root_rot

    return quat_ops.mul_vec(root_rot_seq[:, None, :], local_pos) + root_pos_seq[:, None, :]


def _prepare_motion_for_plot(motion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    motion = np.asarray(motion, dtype=np.float32).copy()
    if motion.ndim != 3 or motion.shape[-1] != 3:
        raise ValueError(f"Expected motion shape (T, J, 3), got {motion.shape}")

    root_traj = motion[:, 0, [0, 2]].copy()
    floor = float(motion[..., 1].min())
    motion[..., 1] -= floor
    motion[..., 0] -= motion[:, 0:1, 0]
    motion[..., 2] -= motion[:, 0:1, 2]
    return motion, root_traj


def _limits_from_motion(motion: np.ndarray, pad: float = 0.15) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    x = motion[..., 0]
    y = motion[..., 2]
    z = motion[..., 1]

    radius = max(float(np.abs(x).max()), float(np.abs(y).max()), 1.0)
    radius *= 1.0 + pad
    z_max = max(float(z.max()) * (1.0 + pad), 1.0)
    return (-radius, radius), (-radius, radius), (0.0, z_max)


def visualize_side_by_side_from_375(
    ref_features_375: np.ndarray,
    pred_features_375: np.ndarray,
    parents: list[int],
    *,
    fps: int,
    dt: float,
    use_global: bool,
    root_wy_in_degrees: bool = True,
    trail_len: int = 120,
    title: Optional[str] = None,
) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def _style_axes(ax, xlim, ylim, zlim):
        ax.set_xlim3d(*xlim)
        ax.set_ylim3d(*ylim)
        ax.set_zlim3d(*zlim)
        ax.view_init(elev=20, azim=45)
        ax.grid(False)
        ax.set_axis_off()
        try:
            ax.set_box_aspect((1.0, 1.0, 0.85))
        except Exception:
            pass

    def _add_ground_plane(ax, xlim, ylim):
        verts = [[
            [xlim[0], ylim[0], 0.0],
            [xlim[0], ylim[1], 0.0],
            [xlim[1], ylim[1], 0.0],
            [xlim[1], ylim[0], 0.0],
        ]]
        plane = Poly3DCollection(verts, alpha=0.25, facecolor=(0.75, 0.78, 0.84), edgecolor="none")
        ax.add_collection3d(plane)

    def _bone_indices(parent_ids: np.ndarray) -> list[int]:
        return [j for j, p in enumerate(parent_ids) if p != -1]

    def _bone_colors(n: int):
        cmap = plt.get_cmap("tab20")
        return [cmap(i % 20) for i in range(n)]

    parents_np = np.asarray(parents, dtype=np.int32)

    if use_global:
        ref_motion = _reconstruct_global_positions_from_375(
            ref_features_375,
            dt=dt,
            root_wy_in_degrees=root_wy_in_degrees,
        )
        pred_motion = _reconstruct_global_positions_from_375(
            pred_features_375,
            dt=dt,
            root_wy_in_degrees=root_wy_in_degrees,
        )
    else:
        _, ref_motion = _decode_375_state(ref_features_375)
        _, pred_motion = _decode_375_state(pred_features_375)

    if ref_motion.shape[1] != len(parents_np):
        raise ValueError(f"Reference joints {ref_motion.shape[1]} mismatch parents length {len(parents_np)}")
    if pred_motion.shape[1] != len(parents_np):
        raise ValueError(f"Prediction joints {pred_motion.shape[1]} mismatch parents length {len(parents_np)}")

    t_total = min(ref_motion.shape[0], pred_motion.shape[0])
    if t_total <= 1:
        raise ValueError(f"Need at least 2 frames for animation, got {t_total}")
    ref_motion = ref_motion[:t_total]
    pred_motion = pred_motion[:t_total]

    ref_motion, ref_root_traj = _prepare_motion_for_plot(ref_motion)
    pred_motion, pred_root_traj = _prepare_motion_for_plot(pred_motion)

    ref_xlim, ref_ylim, ref_zlim = _limits_from_motion(ref_motion)
    pred_xlim, pred_ylim, pred_zlim = _limits_from_motion(pred_motion)
    xlim = (min(ref_xlim[0], pred_xlim[0]), max(ref_xlim[1], pred_xlim[1]))
    ylim = (min(ref_ylim[0], pred_ylim[0]), max(ref_ylim[1], pred_ylim[1]))
    zlim = (min(ref_zlim[0], pred_zlim[0]), max(ref_zlim[1], pred_zlim[1]))

    fig = plt.figure(figsize=(13, 6.5), dpi=120)
    ax_left = fig.add_subplot(121, projection="3d")
    ax_right = fig.add_subplot(122, projection="3d")
    _style_axes(ax_left, xlim, ylim, zlim)
    _style_axes(ax_right, xlim, ylim, zlim)
    _add_ground_plane(ax_left, xlim, ylim)
    _add_ground_plane(ax_right, xlim, ylim)
    ax_left.set_title("Reference", fontsize=11, pad=6)
    ax_right.set_title("Prediction", fontsize=11, pad=6)
    if title is not None:
        fig.suptitle(title, fontsize=12)

    bones = _bone_indices(parents_np)
    colors = _bone_colors(len(bones))
    left_lines = []
    right_lines = []
    for k, j in enumerate(bones):
        p = int(parents_np[j])
        lw = 2.8 if p == 0 else 2.0
        (line_l,) = ax_left.plot([], [], [], color=colors[k], lw=lw, solid_capstyle="round")
        (line_r,) = ax_right.plot([], [], [], color=colors[k], lw=lw, solid_capstyle="round")
        left_lines.append((j, line_l))
        right_lines.append((j, line_r))

    (left_traj_line,) = ax_left.plot([], [], [], color="#2E6FBB", lw=1.6, alpha=0.9)
    (right_traj_line,) = ax_right.plot([], [], [], color="#2E6FBB", lw=1.6, alpha=0.9)
    artists = [line for _, line in left_lines] + [line for _, line in right_lines] + [left_traj_line, right_traj_line]

    def init():
        for line in artists:
            line.set_data_3d([], [], [])
        return artists

    def animate(i):
        for j, line in left_lines:
            p = int(parents_np[j])
            line.set_data_3d(
                [ref_motion[i, j, 0], ref_motion[i, p, 0]],
                [ref_motion[i, j, 2], ref_motion[i, p, 2]],
                [ref_motion[i, j, 1], ref_motion[i, p, 1]],
            )
        for j, line in right_lines:
            p = int(parents_np[j])
            line.set_data_3d(
                [pred_motion[i, j, 0], pred_motion[i, p, 0]],
                [pred_motion[i, j, 2], pred_motion[i, p, 2]],
                [pred_motion[i, j, 1], pred_motion[i, p, 1]],
            )

        if i > 1:
            start = max(0, i - trail_len)
            left_trail = ref_root_traj[start : i + 1] - ref_root_traj[i : i + 1]
            right_trail = pred_root_traj[start : i + 1] - pred_root_traj[i : i + 1]
            left_traj_line.set_data_3d(
                left_trail[:, 0],
                left_trail[:, 1],
                np.zeros(left_trail.shape[0], dtype=np.float32),
            )
            right_traj_line.set_data_3d(
                right_trail[:, 0],
                right_trail[:, 1],
                np.zeros(right_trail.shape[0], dtype=np.float32),
            )
        else:
            left_traj_line.set_data_3d([], [], [])
            right_traj_line.set_data_3d([], [], [])
        return artists

    interval = int(round(1000.0 / fps))
    ani = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=t_total,
        interval=interval,
        blit=False,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.show()
    del ani


def run_vqvae_reconstruct(model: VQVAE375to291, x_norm: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        x_ref_t = torch.from_numpy(x_norm).to(device=device, dtype=torch.float32)
        motion_hat = model.reconstruct(x_ref_t)  # [T,291]
        x_hat = torch.cat((x_ref_t[:, :84], motion_hat), dim=-1)  # [T,375]
    return x_hat.detach().cpu().numpy().astype(np.float32, copy=False)


def run_vqvae_rollout(model: VQVAE375to291, x_norm: np.ndarray, device: torch.device) -> np.ndarray:
    t_total = x_norm.shape[0]
    seq = np.zeros((t_total, 375), dtype=np.float32)

    with torch.no_grad():
        motion_seed = torch.from_numpy(x_norm[0:1, 84:]).to(device=device, dtype=torch.float32)
        for i in range(t_total):
            traj_curr = torch.from_numpy(x_norm[i : i + 1, :84]).to(device=device, dtype=torch.float32)
            x_curr = torch.cat((traj_curr, motion_seed), dim=-1)
            z_curr, _, _ = model.encode(x_curr)  # [1,latent_dim]
            motion_curr = model.decode_step(z_curr, traj_curr, from_indices=False)  # [1,291]
            x_curr = torch.cat((traj_curr, motion_curr), dim=-1)
            seq[i] = x_curr[0].detach().cpu().numpy().astype(np.float32, copy=False)
            motion_seed = motion_curr
    return seq


def run_lmm_reconstruct(model: LMMCompressorDecompressorAE, x_norm: np.ndarray, device: torch.device) -> np.ndarray:
    traj = torch.from_numpy(x_norm[:, :84]).to(device=device, dtype=torch.float32)
    motion = torch.from_numpy(x_norm[:, 84:]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        motion_hat = model(motion=motion, traj=traj)  # [T,291]
        x_hat = torch.cat((traj, motion_hat), dim=-1)
    return x_hat.detach().cpu().numpy().astype(np.float32, copy=False)


def run_lmm_rollout(model: LMMCompressorDecompressorAE, x_norm: np.ndarray, device: torch.device) -> np.ndarray:
    t_total = x_norm.shape[0]
    seq = np.zeros((t_total, 375), dtype=np.float32)
    seq[0] = x_norm[0]

    motion_prev = torch.from_numpy(seq[0, 84:][None, :]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        for i in range(1, t_total):
            traj_next = torch.from_numpy(x_norm[i, :84][None, :]).to(device=device, dtype=torch.float32)
            latent = model.encode(motion_prev)  # [1,32]
            motion_next = model.decode(traj=traj_next, latent=latent)  # [1,291]
            x_next = torch.cat((traj_next, motion_next), dim=-1)
            seq[i] = x_next[0].detach().cpu().numpy().astype(np.float32, copy=False)
            motion_prev = motion_next
    return seq


def _build_mvae_condition_window(seq: np.ndarray, end_idx: int, c: int) -> np.ndarray:
    """
    Return condition [C,375] ending right before `end_idx`.
    Uses generated frames in seq.
    """
    start = end_idx - c
    return seq[start:end_idx]


def run_mvae_reconstruct(model: MVAE375to291, x_norm: np.ndarray, device: torch.device) -> np.ndarray:
    c = model.num_condition_frames
    f = model.num_future_predictions
    if f != 1:
        raise ValueError(
            f"Current unified script supports MVAE only when num_future_predictions=1, got {f}"
        )
    t_total = x_norm.shape[0]
    if t_total <= c:
        raise ValueError(f"Need num_frames > num_condition_frames ({c}), got {t_total}")

    seq = np.zeros((t_total, 375), dtype=np.float32)
    seq[:c] = x_norm[:c]

    with torch.no_grad():
        for i in range(c, t_total):
            cond_np = _build_mvae_condition_window(seq, end_idx=i, c=c)  # [C,375]
            cond = torch.from_numpy(cond_np[None, ...]).to(device=device, dtype=torch.float32)  # [1,C,375]
            x_future = torch.from_numpy(x_norm[i : i + 1][None, ...]).to(device=device, dtype=torch.float32)  # [1,1,375]
            traj_next = torch.from_numpy(x_norm[i : i + 1, :84]).to(device=device, dtype=torch.float32)  # [1,84]

            mu, _ = model.encode(x_future=x_future, condition=cond)  # deterministic latent
            motion_next = model.sample(z=mu, future_traj=traj_next, condition=cond)  # [1,291]
            x_next = torch.cat((traj_next, motion_next), dim=-1)
            seq[i] = x_next[0].detach().cpu().numpy().astype(np.float32, copy=False)
    return seq


def run_mvae_rollout(
    model: MVAE375to291, x_norm: np.ndarray, device: torch.device, noise_std: float
) -> np.ndarray:
    c = model.num_condition_frames
    f = model.num_future_predictions
    if f != 1:
        raise ValueError(
            f"Current unified script supports MVAE only when num_future_predictions=1, got {f}"
        )
    if noise_std <= 0.0:
        raise ValueError(f"noise_std must be > 0, got {noise_std}")
    t_total = x_norm.shape[0]
    if t_total <= c:
        raise ValueError(f"Need num_frames > num_condition_frames ({c}), got {t_total}")

    seq = np.zeros((t_total, 375), dtype=np.float32)
    seq[:c] = x_norm[:c]

    with torch.no_grad():
        for i in range(c, t_total):
            cond_np = _build_mvae_condition_window(seq, end_idx=i, c=c)
            cond = torch.from_numpy(cond_np[None, ...]).to(device=device, dtype=torch.float32)  # [1,C,375]
            traj_next = torch.from_numpy(x_norm[i : i + 1, :84]).to(device=device, dtype=torch.float32)  # [1,84]
            z = torch.randn((1, model.latent_dim), device=device, dtype=torch.float32) * noise_std
            motion_next = model.sample(z=z, future_traj=traj_next, condition=cond)  # [1,291]
            x_next = torch.cat((traj_next, motion_next), dim=-1)
            seq[i] = x_next[0].detach().cpu().numpy().astype(np.float32, copy=False)
    return seq


def default_ckpt_for_model(model_name: str) -> Path:
    if model_name == "vqvae":
        return SCRIPT_DIR / "checkpoints" / "vqvae_375to291_fast_ema" / "best.pt"
    if model_name == "lmm":
        return SCRIPT_DIR / "checkpoints" / "lmm_ae_291to84plus32" / "best.pt"
    if model_name == "mvae":
        return SCRIPT_DIR / "checkpoints" / "mvae_375to291" / "best.pt"
    raise ValueError(f"Unsupported model: {model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified realtime baseline visualization test")
    parser.add_argument(
        "--model",
        type=str,
        choices=["vqvae", "lmm", "mvae"],
        required=True,
        help="Model type to test",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Checkpoint path for selected model. Default uses standard checkpoints/<model>/best.pt",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=SCRIPT_DIR / "database_100sty_375_memmap",
        help="Input DB path (.h5/.hdf5 or memmap directory)",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=None,
        help="Normalization stats npz. Default: sibling of ckpt (norm_stats.npz)",
    )
    parser.add_argument("--num-frames", type=int, default=300, help="Segment length")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for segment sampling")
    parser.add_argument(
        "--style",
        type=str,
        default="",
        help="Style selector for sampled segment (name or label id)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Optional fixed start index in DB (must keep same label for num_frames)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["reconstruct", "rollout"],
        default="rollout",
        help="Generation mode",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=1.0,
        help="Noise std for stochastic rollout (currently used by MVAE rollout)",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "rollout_vis",
        help="Directory for generated h5 and metadata",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Output filename prefix. Default: <model>_<mode>_<num_frames>",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Show side-by-side visualization after generation (default true)",
    )
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Skip visualization window",
    )
    parser.add_argument("--fps", type=int, default=60, help="Visualization fps")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0, help="Visualization dt")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Visualize local positions only (skip global root reconstruction)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "100sty",
        help="100STYLE BVH root used to auto-pick a reference BVH for topology",
    )
    parser.add_argument(
        "--bvh-ref",
        type=Path,
        default=None,
        help="Optional BVH ref for deriving 24-joint parent topology",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.db = args.db.expanduser().resolve()
    if args.ckpt is None:
        args.ckpt = default_ckpt_for_model(args.model)
    args.ckpt = args.ckpt.expanduser().resolve()
    if not args.ckpt.exists():
        raise FileNotFoundError(f"--ckpt not found: {args.ckpt}")
    if args.norm_stats is None:
        args.norm_stats = args.ckpt.parent / "norm_stats.npz"
    args.norm_stats = args.norm_stats.expanduser().resolve()
    if args.prefix is None:
        args.prefix = f"{args.model}_{args.mode}_{args.num_frames}"

    if args.num_frames < 2:
        raise ValueError("--num-frames must be >= 2")
    if args.noise_std <= 0.0:
        raise ValueError("--noise-std must be > 0")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.dt <= 0.0:
        raise ValueError("--dt must be > 0")

    set_seed(args.seed)

    device = resolve_device(args.device)
    print(f"[Info] model           : {args.model}")
    print(f"[Info] mode            : {args.mode}")
    print(f"[Info] device          : {device}")
    print(f"[Info] ckpt            : {args.ckpt}")
    print(f"[Info] norm stats      : {args.norm_stats}")
    print(f"[Info] db              : {args.db}")

    x_source, labels, style_names, backend = load_db_arrays(args.db)
    if labels.size == 0:
        raise RuntimeError("DB is empty")
    print(f"[Info] db backend      : {backend}")
    print(f"[Info] db frames       : {labels.shape[0]}")
    print(f"[Info] num styles      : {len(style_names)}")

    style_label = parse_style_selector(args.style, style_names)
    start, end, sampled_label, src_run = choose_segment(
        labels=labels,
        num_frames=args.num_frames,
        seed=args.seed,
        target_label=style_label,
        start_index=args.start_index,
    )
    x_ref_raw = np.asarray(x_source[start:end], dtype=np.float32)
    if x_ref_raw.shape != (args.num_frames, 375):
        raise RuntimeError(
            f"Unexpected sampled segment shape {x_ref_raw.shape}, expected {(args.num_frames, 375)}"
        )

    norm_mean, norm_std = load_norm_stats(args.norm_stats)
    x_ref_norm = (x_ref_raw - norm_mean[None, :]) / norm_std[None, :]

    if args.model == "vqvae":
        model = build_vqvae_from_checkpoint(args.ckpt).to(device)
        x_hat_norm = (
            run_vqvae_reconstruct(model=model, x_norm=x_ref_norm, device=device)
            if args.mode == "reconstruct"
            else run_vqvae_rollout(model=model, x_norm=x_ref_norm, device=device)
        )
    elif args.model == "lmm":
        model = build_lmm_from_checkpoint(args.ckpt).to(device)
        x_hat_norm = (
            run_lmm_reconstruct(model=model, x_norm=x_ref_norm, device=device)
            if args.mode == "reconstruct"
            else run_lmm_rollout(model=model, x_norm=x_ref_norm, device=device)
        )
    elif args.model == "mvae":
        model = build_mvae_from_checkpoint(args.ckpt).to(device)
        x_hat_norm = (
            run_mvae_reconstruct(model=model, x_norm=x_ref_norm, device=device)
            if args.mode == "reconstruct"
            else run_mvae_rollout(model=model, x_norm=x_ref_norm, device=device, noise_std=args.noise_std)
        )
    else:
        raise RuntimeError(f"Unhandled model: {args.model}")

    x_hat_raw = x_hat_norm * norm_std[None, :] + norm_mean[None, :]

    mse_norm = float(np.mean((x_hat_norm[:, 84:] - x_ref_norm[:, 84:]) ** 2))
    mae_norm = float(np.mean(np.abs(x_hat_norm[:, 84:] - x_ref_norm[:, 84:])))
    print(f"[Info] motion mse(norm): {mse_norm:.6f}")
    print(f"[Info] motion mae(norm): {mae_norm:.6f}")

    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gen_h5_path = args.out_dir / f"{args.prefix}.generated.hdf5"
    ref_h5_path = args.out_dir / f"{args.prefix}.reference.hdf5"
    with h5py.File(gen_h5_path, "w") as f:
        f.create_dataset("X", data=np.asarray(x_hat_raw, dtype=np.float32), dtype=np.float32)
        f.create_dataset("Labels", data=np.full((args.num_frames,), sampled_label, dtype=np.int32))
        style_names_np = np.asarray(style_names, dtype=h5py.string_dtype(encoding="utf-8"))
        f.create_dataset("StyleNames", data=style_names_np)
    with h5py.File(ref_h5_path, "w") as f:
        f.create_dataset("X", data=np.asarray(x_ref_raw, dtype=np.float32), dtype=np.float32)
        f.create_dataset("Labels", data=np.full((args.num_frames,), sampled_label, dtype=np.int32))
        style_names_np = np.asarray(style_names, dtype=h5py.string_dtype(encoding="utf-8"))
        f.create_dataset("StyleNames", data=style_names_np)

    meta = {
        "model": args.model,
        "mode": args.mode,
        "ckpt": str(args.ckpt),
        "norm_stats": str(args.norm_stats),
        "db": str(args.db),
        "db_backend": backend,
        "segment_start": int(start),
        "segment_end": int(end),
        "sampled_label": int(sampled_label),
        "sampled_style": style_names[sampled_label] if 0 <= sampled_label < len(style_names) else f"label_{sampled_label}",
        "source_run": {"start": int(src_run[0]), "end": int(src_run[1]), "label": int(src_run[2])},
        "num_frames": int(args.num_frames),
        "noise_std": float(args.noise_std),
        "generated_h5": str(gen_h5_path),
        "reference_h5": str(ref_h5_path),
        "motion_mse_norm": mse_norm,
        "motion_mae_norm": mae_norm,
    }
    meta_path = args.out_dir / f"{args.prefix}.meta.json"
    with meta_path.open("w", encoding="utf-8") as fw:
        json.dump(meta, fw, ensure_ascii=False, indent=2)

    print(f"[Info] source segment  : [{start}, {end}) label={sampled_label} ({meta['sampled_style']})")
    print(f"[Info] generated h5    : {gen_h5_path}")
    print(f"[Info] reference h5    : {ref_h5_path}")
    print(f"[Info] meta json       : {meta_path}")

    if backend == "hdf5":
        x_source.file.close()

    if args.visualize:
        data_root = args.data_root.expanduser().resolve()
        if args.bvh_ref is not None:
            bvh_ref = args.bvh_ref.expanduser().resolve()
        else:
            bvh_ref = first_bvh_under(data_root)
        if not bvh_ref.exists():
            raise FileNotFoundError(f"Reference BVH not found: {bvh_ref}")

        bvh_data = bvh.load_zeroeggs(str(bvh_ref))
        parents_24 = build_reduced_parents(bvh_data["parents"], SELECTED_24)
        vis_title = (
            f"Baseline compare | model={args.model}/{args.mode} | "
            f"style={meta['sampled_style']} (label={sampled_label}) | "
            f"seg=[{start},{end})"
        )
        print(f"[Info] visualize bvh  : {bvh_ref}")
        print("[Info] visualize mode : left=reference | right=prediction")
        visualize_side_by_side_from_375(
            ref_features_375=x_ref_raw,
            pred_features_375=x_hat_raw,
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
        print("[Done] Generation complete (visualization skipped).")


if __name__ == "__main__":
    main()
