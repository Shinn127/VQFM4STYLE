"""
Training script for VQVAE375to291.

Usage example:
  python NewCode/train_vqvae_375to291.py \
    --db NewCode/database_100sty_375_memmap \
    --out-dir NewCode/checkpoints/vqvae_375to291 \
    --seq-len 32 --batch-size 32 --epochs 120
"""

from __future__ import annotations

import argparse
import contextlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from arch import VQVAE375to291
from train_arg_utils import (
    CommonDataDefaults,
    CommonLoggingDefaults,
    CommonRuntimeDefaults,
    CommonTrainDefaults,
    add_bool_arg,
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
    resolve_amp_dtype,
    resolve_device,
    set_seed,
    to_float,
    validate_memmap_db_dir,
)
from train_common import (
    append_jsonl,
    prepare_data_loaders_from_args,
    resume_training_state,
    save_training_checkpoint,
    validate_common_data_args,
    write_config_json,
)

METRIC_KEYS = (
    "loss",
    "recon_loss",
    "vq_loss",
    "vel_loss",
    "acc_loss",
    "lat_acc_loss",
    "codebook_loss",
    "commitment_loss",
    "perplexity",
)


@dataclass(frozen=True)
class LossConfig:
    recon_type: str
    temporal_loss_type: Optional[str]
    recon_weight: float
    vq_weight: float
    vel_weight: float
    acc_weight: float
    lat_acc_weight: float


@dataclass(frozen=True)
class AmpConfig:
    enabled: bool
    dtype: Optional[torch.dtype]


def epoch_pass(
    *,
    model: VQVAE375to291,
    loader,
    device: torch.device,
    loss_cfg: LossConfig,
    amp_cfg: AmpConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip: float = 0.0,
    grad_accum_steps: int = 1,
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

    metric_sums = {k: torch.zeros((), device=device, dtype=torch.float32) for k in METRIC_KEYS}
    n_samples = 0
    n_updates = 0
    start_time = time.time()

    for step, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True).float()
        target_motion = batch["motion"].to(device, non_blocking=True).float()
        bs = x.shape[0]

        if is_train and ((step - 1) % grad_accum_steps == 0):
            optimizer.zero_grad(set_to_none=True)

        remainder = num_steps % grad_accum_steps
        is_last_incomplete_group = remainder != 0 and step > (num_steps - remainder)
        denom = float(remainder if is_last_incomplete_group else grad_accum_steps)
        is_update_step = (step % grad_accum_steps == 0) or (step == num_steps)

        autocast_ctx = (
            torch.autocast(
                device_type=device.type,
                dtype=amp_cfg.dtype,
                enabled=amp_cfg.enabled and device.type == "cuda" and amp_cfg.dtype is not None,
            )
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            loss_dict = model.compute_loss(
                x=x,
                target_motion=target_motion,
                recon_type=loss_cfg.recon_type,
                temporal_loss_type=loss_cfg.temporal_loss_type,
                recon_weight=loss_cfg.recon_weight,
                vq_weight=loss_cfg.vq_weight,
                vel_weight=loss_cfg.vel_weight,
                acc_weight=loss_cfg.acc_weight,
                lat_acc_weight=loss_cfg.lat_acc_weight,
            )
            loss = loss_dict["loss"] / denom

        if is_train:
            if scaler is not None and amp_cfg.enabled and device.type == "cuda":
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

        for k in METRIC_KEYS:
            metric_sums[k] = metric_sums[k] + loss_dict[k].detach().to(torch.float32) * float(bs)
        n_samples += bs

        if is_train and (step % log_interval == 0 or step == len(loader)):
            elapsed = time.time() - start_time
            samples_per_sec = n_samples / max(elapsed, 1e-6)
            print(
                f"[{split}] epoch {epoch}/{total_epochs} step {step}/{len(loader)} "
                f"loss={to_float(loss_dict['loss']):.6f} "
                f"recon={to_float(loss_dict['recon_loss']):.6f} "
                f"vq={to_float(loss_dict['vq_loss']):.6f} "
                f"vel={to_float(loss_dict['vel_loss']):.6f} "
                f"acc={to_float(loss_dict['acc_loss']):.6f} "
                f"lat_acc={to_float(loss_dict['lat_acc_loss']):.6f} "
                f"ppl={to_float(loss_dict['perplexity']):.3f} "
                f"updates={n_updates} "
                f"{samples_per_sec:.1f} samples/s "
                f"time={elapsed:.1f}s"
            )

    if n_samples == 0:
        return {
            "loss": float("nan"),
            "recon_loss": float("nan"),
            "vq_loss": float("nan"),
            "vel_loss": float("nan"),
            "acc_loss": float("nan"),
            "lat_acc_loss": float("nan"),
            "codebook_loss": float("nan"),
            "commitment_loss": float("nan"),
            "perplexity": float("nan"),
            "num_samples": 0.0,
            "num_updates": 0.0,
        }

    mean_metrics = {k: to_float(v / float(n_samples)) for k, v in metric_sums.items()}
    return {
        "loss": mean_metrics["loss"],
        "recon_loss": mean_metrics["recon_loss"],
        "vq_loss": mean_metrics["vq_loss"],
        "vel_loss": mean_metrics["vel_loss"],
        "acc_loss": mean_metrics["acc_loss"],
        "lat_acc_loss": mean_metrics["lat_acc_loss"],
        "codebook_loss": mean_metrics["codebook_loss"],
        "commitment_loss": mean_metrics["commitment_loss"],
        "perplexity": mean_metrics["perplexity"],
        "num_samples": float(n_samples),
        "num_updates": float(n_updates),
    }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Train VQVAE375to291 on 100STYLE memmap DB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("I/O")
    data_group = parser.add_argument_group("Data")
    train_group = parser.add_argument_group("Training")
    runtime_group = parser.add_argument_group("Runtime")
    loss_group = parser.add_argument_group("Loss")
    model_group = parser.add_argument_group("Model")
    log_group = parser.add_argument_group("Logging / Checkpoints")

    add_common_io_args(io_group, script_dir=script_dir, out_dir_name="vqvae_375to291")
    add_common_data_args(
        data_group,
        defaults=CommonDataDefaults(
            seq_len=16,
            stride=8,
            batch_size=64,
            max_windows_for_stats=50000,
        ),
    )
    add_common_train_args(
        train_group,
        defaults=CommonTrainDefaults(min_lr=2e-5),
    )
    add_common_runtime_args(
        runtime_group,
        defaults=CommonRuntimeDefaults(),
    )

    loss_group.add_argument("--recon-type", type=str, choices=["l1", "mse", "smooth_l1"], default="mse")
    loss_group.add_argument(
        "--temporal-loss-type",
        type=str,
        choices=["auto", "l1", "mse", "smooth_l1"],
        default="auto",
    )
    loss_group.add_argument("--recon-weight", type=float, default=1.0)
    loss_group.add_argument("--vq-weight", type=float, default=1.0)
    loss_group.add_argument("--vel-weight", type=float, default=0.25)
    loss_group.add_argument("--acc-weight", type=float, default=0.25)
    loss_group.add_argument("--lat-acc-weight", type=float, default=0.0)

    model_group.add_argument("--hidden-dim", type=int, default=512)
    model_group.add_argument("--encoder-layers", type=int, default=3)
    model_group.add_argument("--encoder-heads", type=int, default=8)
    model_group.add_argument("--encoder-ffn-dim", type=int, default=1024)
    model_group.add_argument("--decoder-hidden-dim", type=int, default=512)
    model_group.add_argument("--decoder-layers", type=int, default=3)
    model_group.add_argument("--num-latents", type=int, default=32)
    model_group.add_argument("--code-dim", type=int, default=4)
    model_group.add_argument("--num-embeddings", type=int, default=2048)
    model_group.add_argument("--commitment-cost", type=float, default=0.25)
    model_group.add_argument("--codebook-cost", type=float, default=1.0)
    add_bool_arg(model_group, name="use-ema-codebook", default=True)
    model_group.add_argument("--ema-decay", type=float, default=0.99)
    model_group.add_argument("--ema-eps", type=float, default=1e-5)
    model_group.add_argument("--dropout", type=float, default=0.1)
    model_group.add_argument("--max-seq-len", type=int, default=4096)
    model_group.add_argument("--rope-base", type=float, default=10000.0)

    add_common_logging_args(
        log_group,
        defaults=CommonLoggingDefaults(val_every=1),
    )
    return parser.parse_args()


def canonicalize_paths(args: argparse.Namespace) -> None:
    canonicalize_path_args(args)


def validate_args(args: argparse.Namespace) -> None:
    validate_common_data_args(
        grad_accum_steps=args.grad_accum_steps,
        val_every=args.val_every,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )


def build_loss_config(args: argparse.Namespace) -> LossConfig:
    temporal_loss_type = None if args.temporal_loss_type == "auto" else args.temporal_loss_type
    return LossConfig(
        recon_type=args.recon_type,
        temporal_loss_type=temporal_loss_type,
        recon_weight=float(args.recon_weight),
        vq_weight=float(args.vq_weight),
        vel_weight=float(args.vel_weight),
        acc_weight=float(args.acc_weight),
        lat_acc_weight=float(args.lat_acc_weight),
    )


def build_model(args: argparse.Namespace, device: torch.device) -> VQVAE375to291:
    model = VQVAE375to291(
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        encoder_heads=args.encoder_heads,
        encoder_ffn_dim=args.encoder_ffn_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_layers=args.decoder_layers,
        num_latents=args.num_latents,
        code_dim=args.code_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost,
        codebook_cost=args.codebook_cost,
        use_ema_codebook=args.use_ema_codebook,
        ema_decay=args.ema_decay,
        ema_eps=args.ema_eps,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        rope_base=args.rope_base,
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
    return model


def build_optimizer(model: VQVAE375to291, args: argparse.Namespace, device: torch.device):
    return build_adamw(
        model.parameters(),
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        use_fused=bool(args.fused_adamw and device.type == "cuda"),
    )


def main() -> None:
    args = parse_args()
    canonicalize_paths(args)
    validate_memmap_db_dir(args.db)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    configure_torch_runtime(
        device=device,
        tf32=args.tf32,
        cudnn_benchmark=args.cudnn_benchmark,
        matmul_precision=args.matmul_precision,
    )
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device=device)
    amp_cfg = AmpConfig(
        enabled=bool(args.amp and device.type == "cuda" and amp_dtype is not None),
        dtype=amp_dtype,
    )
    loss_cfg = build_loss_config(args)

    print(f"[Info] device          : {device}")
    print(f"[Info] amp             : {amp_cfg.enabled} ({args.amp_dtype})")
    print(
        f"[Info] backend         : tf32={args.tf32} cudnn_benchmark={args.cudnn_benchmark} "
        f"matmul_precision={args.matmul_precision}"
    )
    print(f"[Info] db              : {args.db}")
    print(f"[Info] out_dir         : {args.out_dir}")
    db_size_gb = estimate_path_size_bytes(args.db) / (1024.0 ** 3) if args.db.exists() else float("nan")
    print(f"[Info] db size         : {db_size_gb:.2f} GB")
    print(
        f"[Info] temporal loss   : type={args.temporal_loss_type} "
        f"vel={loss_cfg.vel_weight} acc={loss_cfg.acc_weight} lat_acc={loss_cfg.lat_acc_weight}"
    )
    print(f"[Info] grad accum      : {args.grad_accum_steps}")
    print(f"[Info] split ratio     : val={args.val_ratio} test={args.test_ratio}")
    print(
        f"[Info] data fraction   : train={args.train_fraction} "
        f"val={args.val_fraction} test={args.test_fraction}"
    )
    print(f"[Info] val every       : {args.val_every}")
    print(
        f"[Info] ema codebook    : enabled={args.use_ema_codebook} "
        f"decay={args.ema_decay} eps={args.ema_eps}"
    )
    print(f"[Info] decoder         : mlp hidden={args.decoder_hidden_dim} layers={args.decoder_layers}")
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

    write_config_json(out_dir=args.out_dir, args=args)

    prepared = prepare_data_loaders_from_args(
        args=args,
        test_loader_note="prepared (reserved for future evaluation)",
    )
    train_loader = prepared.train_loader
    val_loader = prepared.val_loader

    model = build_model(args, device=device)
    optimizer, use_fused_adamw = build_optimizer(model=model, args=args, device=device)
    print(f"[Info] optimizer       : AdamW(fused={use_fused_adamw})")

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler_enabled = bool(amp_cfg.enabled and amp_cfg.dtype == torch.float16 and device.type == "cuda")
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
            loss_cfg=loss_cfg,
            amp_cfg=amp_cfg,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
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
                        loss_cfg=loss_cfg,
                        amp_cfg=amp_cfg,
                        optimizer=None,
                        scaler=None,
                        grad_clip=0.0,
                        grad_accum_steps=1,
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
            f"train_recon={train_metrics['recon_loss']:.6f} "
            f"train_vq={train_metrics['vq_loss']:.6f} "
            f"train_vel={train_metrics['vel_loss']:.6f} "
            f"train_acc={train_metrics['acc_loss']:.6f} "
            f"train_lat_acc={train_metrics['lat_acc_loss']:.6f} "
            f"train_ppl={train_metrics['perplexity']:.3f}"
        )
        if val_metrics is not None:
            msg += (
                f" | val_loss={val_metrics['loss']:.6f} "
                f"val_recon={val_metrics['recon_loss']:.6f} "
                f"val_vq={val_metrics['vq_loss']:.6f} "
                f"val_vel={val_metrics['vel_loss']:.6f} "
                f"val_acc={val_metrics['acc_loss']:.6f} "
                f"val_lat_acc={val_metrics['lat_acc_loss']:.6f} "
                f"val_ppl={val_metrics['perplexity']:.3f}"
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

    print("\n[Done] Training complete.")
    print(f"[Done] Best metric: {best_metric:.6f}")
    print(f"[Done] Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
