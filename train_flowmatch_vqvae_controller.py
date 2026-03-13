"""
Training script for style-conditioned flow matching controller on top of frozen VQ-VAE.

Pipeline:
  - E_c encodes current frame content (traj_i + motion_i)
  - E_s encodes style label
  - Flow matching generator predicts next-frame VQ latent
  - Frozen VQ-VAE decoder reconstructs next-frame motion from (traj_{i+1}, latent)

Usage example:
  python NewCode/train_flowmatch_vqvae_controller.py \
    --db NewCode/database_100sty_375_memmap \
    --vqvae-ckpt NewCode/checkpoints/vqvae_375to291_fast_ema/best.pt \
    --vq-target-cache NewCode/cache/vq_targets_vqvae_375to291_fast_ema \
    --out-dir NewCode/checkpoints/flowmatch_vqvae_controller \
    --seq-len 32 --batch-size 512 --epochs 50
"""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from arch import FlowMatchVQVAEController
from model_loaders import build_vqvae_from_checkpoint
from train_arg_utils import (
    CommonDataDefaults,
    CommonLoggingDefaults,
    CommonRuntimeDefaults,
    CommonTrainDefaults,
    add_common_data_args,
    add_common_io_args,
    add_common_logging_args,
    add_common_runtime_args,
    add_common_train_args,
    canonicalize_path_args,
)
from runtime_utils import (
    build_adamw,
    configure_torch_runtime,
    estimate_path_size_bytes,
    load_norm_stats,
    resolve_amp_dtype,
    resolve_device,
    sanitize_state_dict_keys,
    set_seed,
    to_float,
    validate_memmap_db_dir,
)
from train_common import (
    append_jsonl,
    load_best_checkpoint_for_eval,
    prepare_data_loaders_from_args,
    resume_training_state,
    save_training_checkpoint,
    validate_common_data_args,
    write_config_json,
)
from vq_target_cache import VQTargetCache


def _flatten_adjacent_pairs(
    x: torch.Tensor,
    style_label: torch.Tensor,
    start: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Convert [B,T,D] windows into adjacent frame pairs:
      x_curr:   [B*(T-1), D]
      x_next:   [B*(T-1), D]
      style_bt: [B*(T-1)]
    """
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if x.dim() != 3:
        raise ValueError(f"Expected x shape [B,T,D] or [B,D], got {tuple(x.shape)}")
    b, t, d = x.shape
    if t < 2:
        raise ValueError(
            "Need seq_len >= 2 for next-frame training, "
            f"got input with T={t}. Increase --seq-len."
        )
    if style_label.dim() == 0:
        style_label = style_label.unsqueeze(0)
    if style_label.dim() != 1:
        raise ValueError(f"Expected style_label shape [B], got {tuple(style_label.shape)}")
    if style_label.numel() != b:
        raise ValueError(
            f"Expected style_label count {b} (batch size), got {style_label.numel()}"
        )
    next_frame_ids = None
    if start is not None:
        if start.dim() == 0:
            start = start.unsqueeze(0)
        if start.dim() != 1:
            raise ValueError(f"Expected start shape [B], got {tuple(start.shape)}")
        if start.numel() != b:
            raise ValueError(f"Expected start count {b} (batch size), got {start.numel()}")
        offsets = torch.arange(1, t, device=start.device, dtype=start.dtype)
        next_frame_ids = start.view(b, 1) + offsets.view(1, t - 1)
        next_frame_ids = next_frame_ids.reshape(-1)

    x_curr = x[:, :-1, :].reshape(-1, d)
    x_next = x[:, 1:, :].reshape(-1, d)
    style_bt = style_label.view(b, 1).expand(b, t - 1).reshape(-1)
    return x_curr, x_next, style_bt, next_frame_ids


def epoch_pass(
    *,
    model: FlowMatchVQVAEController,
    loader,
    device: torch.device,
    flow_loss_type: str,
    recon_type: str,
    flow_weight: float,
    recon_weight: float,
    noise_std: float,
    solver_steps: int,
    solver: str,
    vq_target_cache: Optional[VQTargetCache] = None,
    cache_vqvae=None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip: float = 0.0,
    grad_accum_steps: int = 1,
    amp_enabled: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    log_interval: int = 100,
    epoch: int = 0,
    total_epochs: int = 0,
    split: str = "train",
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")

    num_steps = len(loader)
    metric_keys = [
        "loss",
        "flow_loss",
        "recon_loss",
        "latent_target_norm",
        "velocity_pred_norm",
        "velocity_target_norm",
    ]
    metric_sums = {k: torch.zeros((), device=device, dtype=torch.float32) for k in metric_keys}
    n_samples = 0
    n_updates = 0
    start_time = time.time()

    for step, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True).float()
        style_label = batch["label"].to(device, non_blocking=True).long()
        start_idx = batch.get("start", None)
        if start_idx is not None:
            start_idx = start_idx.long()
        x_curr, x_next, style_bt, next_frame_ids = _flatten_adjacent_pairs(x, style_label, start_idx)
        bs = x_curr.size(0)
        target_latent = None
        if vq_target_cache is not None:
            if cache_vqvae is None:
                raise RuntimeError("cache_vqvae must be provided when vq_target_cache is enabled")
            if next_frame_ids is None:
                raise RuntimeError("next_frame_ids are required when vq_target_cache is enabled")
            target_latent = vq_target_cache.lookup_latents(
                next_frame_ids,
                vqvae=cache_vqvae,
                device=device,
            )

        if is_train and ((step - 1) % grad_accum_steps == 0):
            optimizer.zero_grad(set_to_none=True)

        remainder = num_steps % grad_accum_steps
        is_last_incomplete_group = remainder != 0 and step > (num_steps - remainder)
        denom = float(remainder if is_last_incomplete_group else grad_accum_steps)
        is_update_step = (step % grad_accum_steps == 0) or (step == num_steps)

        autocast_ctx = (
            torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled and device.type == "cuda" and amp_dtype is not None,
            )
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            loss_dict = model.compute_loss(
                x_curr=x_curr,
                x_next=x_next,
                style_label=style_bt,
                target_latent=target_latent,
                flow_loss_type=flow_loss_type,
                recon_type=recon_type,
                flow_weight=flow_weight,
                recon_weight=recon_weight,
                noise_std=noise_std,
                solver_steps=solver_steps,
                solver=solver,
            )
            loss = loss_dict["loss"] / denom

        if is_train:
            if scaler is not None and amp_enabled and device.type == "cuda":
                scaler.scale(loss).backward()
                if is_update_step:
                    if grad_clip > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    n_updates += 1
            else:
                loss.backward()
                if is_update_step:
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    n_updates += 1

        for k in metric_keys:
            metric_sums[k] = metric_sums[k] + loss_dict[k].detach().to(torch.float32) * float(bs)
        n_samples += bs

        if is_train and (step % log_interval == 0 or step == len(loader)):
            elapsed = time.time() - start_time
            samples_per_sec = n_samples / max(elapsed, 1e-6)
            print(
                f"[{split}] epoch {epoch}/{total_epochs} step {step}/{len(loader)} "
                f"loss={to_float(loss_dict['loss']):.6f} "
                f"flow={to_float(loss_dict['flow_loss']):.6f} "
                f"recon={to_float(loss_dict['recon_loss']):.6f} "
                f"vpred={to_float(loss_dict['velocity_pred_norm']):.4f} "
                f"vtarget={to_float(loss_dict['velocity_target_norm']):.4f} "
                f"updates={n_updates} "
                f"{samples_per_sec:.1f} pairs/s "
                f"time={elapsed:.1f}s"
            )

    if n_samples == 0:
        return {
            "loss": float("nan"),
            "flow_loss": float("nan"),
            "recon_loss": float("nan"),
            "latent_target_norm": float("nan"),
            "velocity_pred_norm": float("nan"),
            "velocity_target_norm": float("nan"),
            "num_samples": 0.0,
            "num_updates": 0.0,
        }

    mean_metrics = {k: to_float(v / float(n_samples)) for k, v in metric_sums.items()}
    return {
        "loss": mean_metrics["loss"],
        "flow_loss": mean_metrics["flow_loss"],
        "recon_loss": mean_metrics["recon_loss"],
        "latent_target_norm": mean_metrics["latent_target_norm"],
        "velocity_pred_norm": mean_metrics["velocity_pred_norm"],
        "velocity_target_norm": mean_metrics["velocity_target_norm"],
        "num_samples": float(n_samples),
        "num_updates": float(n_updates),
    }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Train flow-matching style controller with frozen VQ-VAE decoder"
    )
    add_common_io_args(parser, script_dir=script_dir, out_dir_name="flowmatch_vqvae_controller")
    parser.add_argument(
        "--vqvae-ckpt",
        type=Path,
        default=script_dir / "checkpoints" / "vqvae_375to291_fast_ema" / "best.pt",
        help="Path to pretrained VQ-VAE checkpoint",
    )
    parser.add_argument(
        "--vq-target-cache",
        type=Path,
        default=None,
        help="Optional precomputed VQ target cache directory",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=None,
        help="Optional normalization stats npz. Default: estimate from train split, or use cache metadata when --vq-target-cache is set",
    )
    add_common_data_args(
        parser,
        defaults=CommonDataDefaults(
            seq_len=32,
            stride=24,
            batch_size=512,
            max_windows_for_stats=0,
        ),
    )
    add_common_train_args(
        parser,
        defaults=CommonTrainDefaults(min_lr=1e-5),
    )
    add_common_runtime_args(
        parser,
        defaults=CommonRuntimeDefaults(),
    )

    parser.add_argument("--num-styles", type=int, default=0, help="0 means infer from dataset")
    parser.add_argument("--content-dim", type=int, default=256)
    parser.add_argument("--style-dim", type=int, default=128)
    parser.add_argument("--time-embed-dim", type=int, default=64)
    parser.add_argument("--encoder-hidden-dim", type=int, default=512)
    parser.add_argument("--flow-hidden-dim", type=int, default=512)
    parser.add_argument("--flow-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--flow-loss-type", type=str, choices=["l1", "mse", "smooth_l1"], default="mse")
    parser.add_argument("--recon-type", type=str, choices=["l1", "mse", "smooth_l1"], default="mse")
    parser.add_argument("--flow-weight", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=0.25)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument(
        "--solver",
        type=str,
        choices=["euler", "midpoint", "heun"],
        default="euler",
        help="ODE solver used for train-time reconstruction and default realtime inference.",
    )
    parser.add_argument(
        "--solver-steps",
        "--recon-solver-steps",
        dest="solver_steps",
        type=int,
        default=4,
        help="Euler steps used for train-time latent reconstruction; inference defaults to this value when not explicitly overridden.",
    )

    add_common_logging_args(
        parser,
        defaults=CommonLoggingDefaults(val_every=5),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonicalize_path_args(args, extra_fields=("vqvae_ckpt", "vq_target_cache", "norm_stats"))
    validate_memmap_db_dir(args.db)
    if not args.vqvae_ckpt.exists():
        raise FileNotFoundError(f"--vqvae-ckpt not found: {args.vqvae_ckpt}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)
    configure_torch_runtime(
        device=device,
        tf32=args.tf32,
        cudnn_benchmark=args.cudnn_benchmark,
        matmul_precision=args.matmul_precision,
    )
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device)
    amp_enabled = bool(args.amp and device.type == "cuda" and amp_dtype is not None)
    validate_common_data_args(
        grad_accum_steps=args.grad_accum_steps,
        val_every=args.val_every,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    if args.seq_len < 2:
        raise ValueError("--seq-len must be >= 2 for next-frame flow matching.")
    if args.vq_target_cache is not None and not args.normalize:
        raise ValueError("--vq-target-cache requires --normalize because cached VQ targets depend on normalized inputs.")
    if args.num_styles < 0:
        raise ValueError(f"--num-styles must be >= 0, got {args.num_styles}")
    if args.content_dim < 1:
        raise ValueError(f"--content-dim must be >= 1, got {args.content_dim}")
    if args.style_dim < 1:
        raise ValueError(f"--style-dim must be >= 1, got {args.style_dim}")
    if args.time_embed_dim < 1:
        raise ValueError(f"--time-embed-dim must be >= 1, got {args.time_embed_dim}")
    if args.encoder_hidden_dim < 1:
        raise ValueError(f"--encoder-hidden-dim must be >= 1, got {args.encoder_hidden_dim}")
    if args.flow_hidden_dim < 1:
        raise ValueError(f"--flow-hidden-dim must be >= 1, got {args.flow_hidden_dim}")
    if args.flow_layers < 1:
        raise ValueError(f"--flow-layers must be >= 1, got {args.flow_layers}")
    if not (0.0 <= args.dropout < 1.0):
        raise ValueError(f"--dropout must be in [0,1), got {args.dropout}")
    if args.flow_weight < 0.0:
        raise ValueError(f"--flow-weight must be >= 0, got {args.flow_weight}")
    if args.recon_weight < 0.0:
        raise ValueError(f"--recon-weight must be >= 0, got {args.recon_weight}")
    if args.noise_std <= 0.0:
        raise ValueError(f"--noise-std must be > 0, got {args.noise_std}")
    if args.solver_steps < 1:
        raise ValueError(f"--solver-steps must be >= 1, got {args.solver_steps}")
    print(f"[Info] device          : {device}")
    print(f"[Info] amp             : {amp_enabled} ({args.amp_dtype})")
    print(
        f"[Info] backend         : tf32={args.tf32} cudnn_benchmark={args.cudnn_benchmark} "
        f"matmul_precision={args.matmul_precision}"
    )
    print(f"[Info] db              : {args.db}")
    print(f"[Info] vqvae ckpt      : {args.vqvae_ckpt}")
    print(f"[Info] out_dir         : {args.out_dir}")
    db_size_gb = estimate_path_size_bytes(args.db) / (1024.0 ** 3) if args.db.exists() else float("nan")
    print(f"[Info] db size         : {db_size_gb:.2f} GB")
    print(
        f"[Info] losses          : flow={args.flow_weight} ({args.flow_loss_type}) "
        f"recon={args.recon_weight} ({args.recon_type})"
    )
    print(
        f"[Info] flow setup      : noise_std={args.noise_std} "
        f"solver={args.solver} solver_steps={args.solver_steps}"
    )
    print(f"[Info] grad accum      : {args.grad_accum_steps}")
    print(f"[Info] split ratio     : val={args.val_ratio} test={args.test_ratio}")
    print(
        f"[Info] data fraction   : train={args.train_fraction} "
        f"val={args.val_fraction} test={args.test_fraction}"
    )
    print(f"[Info] val every       : {args.val_every}")
    print(
        f"[Info] dataloader      : workers={args.num_workers} "
        f"persistent={args.persistent_workers} prefetch={args.prefetch_factor}"
    )
    print(f"[Info] stats chunk     : {args.stats_windows_chunk_size}")
    if not args.load_into_memory and np.isfinite(db_size_gb):
        print(
            "[Tip] if RAM is sufficient, add --load-into-memory for faster training throughput "
            f"(needs about {db_size_gb + 2.0:.1f} GB memory for X + overhead)."
        )

    vq_target_cache = None
    fixed_norm_mean = None
    fixed_norm_std = None
    if args.vq_target_cache is not None:
        vq_target_cache = VQTargetCache(args.vq_target_cache)
        if args.norm_stats is None:
            args.norm_stats = vq_target_cache.meta.norm_stats
        print(f"[Info] vq target cache : {args.vq_target_cache}")
    if args.norm_stats is not None:
        if not args.normalize:
            raise ValueError("--norm-stats requires normalization to be enabled.")
        fixed_norm_mean, fixed_norm_std = load_norm_stats(args.norm_stats)
        print(f"[Info] fixed norm stats: {args.norm_stats}")

    write_config_json(out_dir=args.out_dir, args=args)

    prepared = prepare_data_loaders_from_args(
        args=args,
        test_loader_note="prepared (will evaluate after training)",
        norm_mean=fixed_norm_mean,
        norm_std=fixed_norm_std,
    )
    bundle = prepared.bundle
    train_loader = prepared.train_loader
    val_loader = prepared.val_loader
    test_loader = prepared.test_loader

    if args.num_styles > 0:
        num_styles = int(args.num_styles)
    else:
        style_names = getattr(bundle.train_dataset, "style_names", None)
        if style_names is not None and len(style_names) > 0:
            num_styles = len(style_names)
        else:
            labels = getattr(bundle.train_dataset, "labels", None)
            if labels is None or len(labels) == 0:
                raise RuntimeError("Cannot infer num_styles from dataset; set --num-styles explicitly.")
            num_styles = int(np.max(labels)) + 1
    print(f"[Info] num styles      : {num_styles}")

    vqvae = build_vqvae_from_checkpoint(args.vqvae_ckpt).to(device)
    print(
        f"[Info] frozen vqvae    : latent_dim={vqvae.latent_dim} "
        f"num_latents={vqvae.num_latents} code_dim={vqvae.code_dim}"
    )
    if vq_target_cache is not None:
        vq_target_cache.validate(
            db=args.db,
            vqvae_ckpt=args.vqvae_ckpt,
            norm_stats=args.norm_stats,
            vqvae=vqvae,
        )

    model = FlowMatchVQVAEController(
        vqvae=vqvae,
        num_styles=num_styles,
        content_dim=args.content_dim,
        style_dim=args.style_dim,
        time_embed_dim=args.time_embed_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        flow_hidden_dim=args.flow_hidden_dim,
        flow_layers=args.flow_layers,
        dropout=args.dropout,
    ).to(device)

    if args.compile:
        if not hasattr(torch, "compile"):
            print("[Warn] torch.compile is not available in this torch build; skipping --compile.")
        else:
            model = torch.compile(
                model,
                mode=args.compile_mode,
                fullgraph=bool(args.compile_fullgraph),
            )
            print(
                f"[Info] compile         : enabled (mode={args.compile_mode}, fullgraph={args.compile_fullgraph})"
            )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found in flow matching model.")
    optimizer, use_fused_adamw = build_adamw(
        trainable_params,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        use_fused=bool(args.fused_adamw and device.type == "cuda"),
    )
    print(f"[Info] optimizer       : AdamW(fused={use_fused_adamw})")

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler_enabled = bool(amp_enabled and amp_dtype == torch.float16 and device.type == "cuda")
    scaler = None
    if device.type == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            # PyTorch >= 2.4 preferred API.
            scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        else:
            # Backward compatibility for older torch builds.
            scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    best_metric = float("inf")
    history_path = args.out_dir / "history.jsonl"
    start_epoch, best_metric = resume_training_state(
        resume_path=args.resume,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        reset_optimizer=args.reset_optimizer,
        best_metric=best_metric,
        state_dict_transform=sanitize_state_dict_keys,
        strict=True,
    )

    if start_epoch > args.epochs:
        print(
            f"[Warn] start_epoch ({start_epoch}) > epochs ({args.epochs}). "
            "Nothing to train. Increase --epochs or use another checkpoint."
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\n[Epoch {epoch}/{args.epochs}] lr={lr_now:.8f}")

        train_metrics = epoch_pass(
            model=model,
            loader=train_loader,
            device=device,
            flow_loss_type=args.flow_loss_type,
            recon_type=args.recon_type,
            flow_weight=args.flow_weight,
            recon_weight=args.recon_weight,
            noise_std=args.noise_std,
            solver_steps=args.solver_steps,
            solver=args.solver,
            vq_target_cache=vq_target_cache,
            cache_vqvae=vqvae,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            log_interval=args.log_interval,
            epoch=epoch,
            total_epochs=args.epochs,
            split="train",
        )

        if val_loader is None:
            val_metrics = None
            metric_for_best = train_metrics["loss"]
        else:
            if (epoch % args.val_every) == 0:
                with torch.no_grad():
                    val_metrics = epoch_pass(
                        model=model,
                        loader=val_loader,
                        device=device,
                        flow_loss_type=args.flow_loss_type,
                        recon_type=args.recon_type,
                        flow_weight=args.flow_weight,
                        recon_weight=args.recon_weight,
                        noise_std=args.noise_std,
                        solver_steps=args.solver_steps,
                        solver=args.solver,
                        vq_target_cache=vq_target_cache,
                        cache_vqvae=vqvae,
                        optimizer=None,
                        scaler=None,
                        grad_clip=0.0,
                        grad_accum_steps=1,
                        amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype,
                        log_interval=args.log_interval,
                        epoch=epoch,
                        total_epochs=args.epochs,
                        split="val",
                    )
                metric_for_best = val_metrics["loss"]
            else:
                val_metrics = None
                metric_for_best = None

        scheduler.step()

        epoch_sec = time.time() - epoch_t0
        msg = (
            f"[Epoch {epoch}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_flow={train_metrics['flow_loss']:.6f} "
            f"train_recon={train_metrics['recon_loss']:.6f} "
            f"train_vpred={train_metrics['velocity_pred_norm']:.4f} "
            f"train_vtarget={train_metrics['velocity_target_norm']:.4f}"
        )
        if val_metrics is not None:
            msg += (
                f" | val_loss={val_metrics['loss']:.6f} "
                f"val_flow={val_metrics['flow_loss']:.6f} "
                f"val_recon={val_metrics['recon_loss']:.6f} "
                f"val_vpred={val_metrics['velocity_pred_norm']:.4f} "
                f"val_vtarget={val_metrics['velocity_target_norm']:.4f}"
            )
        msg += f" | time={epoch_sec:.1f}s"
        print(msg)

        record = {
            "epoch": epoch,
            "lr": lr_now,
            "train": train_metrics,
            "val": val_metrics,
            "time_sec": epoch_sec,
        }
        append_jsonl(path=history_path, record=record)

        save_training_checkpoint(
            ckpt_path=args.out_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=best_metric,
            args=args,
        )

        if metric_for_best is not None and metric_for_best < best_metric:
            best_metric = metric_for_best
            save_training_checkpoint(
                ckpt_path=args.out_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                args=args,
            )
            print(f"[Info] saved best checkpoint (metric={best_metric:.6f})")

        if args.save_every > 0 and (epoch % args.save_every == 0):
            save_training_checkpoint(
                ckpt_path=args.out_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                args=args,
            )

    if test_loader is not None:
        best_ckpt_path = args.out_dir / "best.pt"
        load_best_checkpoint_for_eval(
            model=model,
            ckpt_path=best_ckpt_path,
            state_dict_transform=sanitize_state_dict_keys,
            strict=True,
        )

        with torch.no_grad():
            test_metrics = epoch_pass(
                model=model,
                loader=test_loader,
                device=device,
                flow_loss_type=args.flow_loss_type,
                recon_type=args.recon_type,
                flow_weight=args.flow_weight,
                recon_weight=args.recon_weight,
                noise_std=args.noise_std,
                solver_steps=args.solver_steps,
                solver=args.solver,
                vq_target_cache=vq_target_cache,
                cache_vqvae=vqvae,
                optimizer=None,
                scaler=None,
                grad_clip=0.0,
                grad_accum_steps=1,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                log_interval=args.log_interval,
                epoch=args.epochs,
                total_epochs=args.epochs,
                split="test",
            )
        print(
            "[Test] "
            f"loss={test_metrics['loss']:.6f} "
            f"flow={test_metrics['flow_loss']:.6f} "
            f"recon={test_metrics['recon_loss']:.6f} "
            f"vpred={test_metrics['velocity_pred_norm']:.4f} "
            f"vtarget={test_metrics['velocity_target_norm']:.4f}"
        )
        append_jsonl(
            path=history_path,
            record={"epoch": "test", "split": "test", "metrics": test_metrics},
        )

    print("\n[Done] Training complete.")
    print(f"[Done] Best metric: {best_metric:.6f}")
    print(f"[Done] Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
