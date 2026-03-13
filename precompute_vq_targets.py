"""
Precompute per-frame VQ target indices for FlowMatchVQVAEController training.

Output directory layout:
  <out_dir>/
    meta.json
    indices.u16 or indices.i32

The cache stores discrete VQ codebook indices for every frame after applying the
same normalization stats used by the frozen VQ-VAE encoder. During flow training,
these indices can be looked up and converted back to quantized latents cheaply,
avoiding repeated VQ-VAE encoder passes for every next-frame target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import json
import numpy as np
import torch

from model_loaders import build_vqvae_from_checkpoint
from runtime_utils import (
    MEMMAP_META_FILE,
    MEMMAP_X_FILE,
    configure_torch_runtime,
    load_norm_stats,
    resolve_device,
    set_seed,
    validate_memmap_db_dir,
)
from vq_target_cache import VQTargetCacheMeta, resolve_vq_indices_dtype, save_vq_target_cache_meta


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Precompute per-frame VQ target cache from memmap DB")
    parser.add_argument(
        "--db",
        type=Path,
        default=script_dir / "database_100sty_375_memmap",
        help="Input memmap directory",
    )
    parser.add_argument(
        "--vqvae-ckpt",
        type=Path,
        default=script_dir / "checkpoints" / "vqvae_375to291_fast_ema" / "best.pt",
        help="Frozen VQ-VAE checkpoint",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=None,
        help="Normalization stats npz with mean/std. Default: sibling of VQ-VAE checkpoint",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=script_dir / "cache" / "vq_targets_vqvae_375to291_fast_ema",
        help="Output cache directory",
    )
    parser.add_argument("--chunk-rows", type=int, default=4096, help="Frames encoded per chunk")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul/cudnn on CUDA")
    parser.add_argument(
        "--matmul-precision",
        type=str,
        choices=["highest", "high", "medium"],
        default="high",
    )
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.db = args.db.expanduser().resolve()
    args.vqvae_ckpt = args.vqvae_ckpt.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    validate_memmap_db_dir(args.db)
    if not args.vqvae_ckpt.exists():
        raise FileNotFoundError(f"--vqvae-ckpt not found: {args.vqvae_ckpt}")
    if args.norm_stats is None:
        args.norm_stats = args.vqvae_ckpt.parent / "norm_stats.npz"
    args.norm_stats = args.norm_stats.expanduser().resolve()
    if args.chunk_rows < 1:
        raise ValueError(f"--chunk-rows must be >= 1, got {args.chunk_rows}")

    set_seed(args.seed)
    device = resolve_device(args.device)
    configure_torch_runtime(
        device=device,
        tf32=args.tf32,
        cudnn_benchmark=args.cudnn_benchmark,
        matmul_precision=args.matmul_precision,
    )

    mean, std = load_norm_stats(args.norm_stats)

    with (args.db / MEMMAP_META_FILE).open("r", encoding="utf-8") as f:
        db_meta = json.load(f)
    num_frames = int(db_meta.get("num_frames", -1))
    feature_dim = int(db_meta.get("feature_dim", -1))
    if num_frames <= 0:
        raise ValueError(f"Invalid num_frames in {args.db / MEMMAP_META_FILE}: {num_frames}")
    if feature_dim != 375:
        raise ValueError(f"Invalid feature_dim in {args.db / MEMMAP_META_FILE}: {feature_dim}")

    vqvae = build_vqvae_from_checkpoint(args.vqvae_ckpt).to(device)
    vqvae.eval()
    indices_dtype, indices_dtype_name, indices_file = resolve_vq_indices_dtype(vqvae.quantizer.num_embeddings)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    indices_path = args.out_dir / indices_file
    meta_path = args.out_dir / "meta.json"
    if not args.overwrite:
        for path in (indices_path, meta_path):
            if path.exists():
                raise FileExistsError(f"{path} already exists. Use --overwrite to replace.")

    x_source = np.memmap(args.db / MEMMAP_X_FILE, dtype=np.float32, mode="r", shape=(num_frames, 375))
    indices_mm = np.memmap(
        indices_path,
        dtype=indices_dtype,
        mode="w+",
        shape=(num_frames, int(vqvae.num_latents)),
    )

    print(f"[Info] device          : {device}")
    print(f"[Info] db              : {args.db}")
    print(f"[Info] vqvae ckpt      : {args.vqvae_ckpt}")
    print(f"[Info] norm stats      : {args.norm_stats}")
    print(f"[Info] out dir         : {args.out_dir}")
    print(
        f"[Info] cache layout    : num_frames={num_frames} num_codebooks={vqvae.num_latents} "
        f"code_dim={vqvae.code_dim} dtype={indices_dtype_name}"
    )

    with torch.no_grad():
        for start in range(0, num_frames, args.chunk_rows):
            end = min(start + args.chunk_rows, num_frames)
            x_chunk = np.asarray(x_source[start:end], dtype=np.float32)
            x_chunk = (x_chunk - mean[None, :]) / std[None, :]
            x_t = torch.from_numpy(x_chunk).to(device=device, dtype=torch.float32)
            _, indices_t, _ = vqvae.encode(x_t)
            indices_np = indices_t.detach().cpu().numpy().astype(indices_dtype, copy=False)
            indices_mm[start:end] = indices_np
            print(f"[Encode] rows {start}:{end} / {num_frames}")

    indices_mm.flush()
    del indices_mm

    meta = VQTargetCacheMeta(
        db=args.db,
        vqvae_ckpt=args.vqvae_ckpt,
        norm_stats=args.norm_stats,
        num_frames=num_frames,
        feature_dim=feature_dim,
        num_codebooks=int(vqvae.num_latents),
        code_dim=int(vqvae.code_dim),
        latent_dim=int(vqvae.latent_dim),
        num_embeddings=int(vqvae.quantizer.num_embeddings),
        indices_dtype=indices_dtype_name,
        indices_file=indices_file,
    )
    save_vq_target_cache_meta(args.out_dir, meta)

    print(f"[Done] VQ target cache saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
