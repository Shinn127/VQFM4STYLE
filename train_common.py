from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from dataset import DataBundle, build_train_val_test_dataloaders
from runtime_utils import (
    build_dataloader,
    load_checkpoint,
    save_norm_stats_if_present,
    subset_dataset_by_fraction,
)


@dataclass(frozen=True)
class PreparedDataLoaders:
    bundle: DataBundle
    train_dataset: Dataset
    val_dataset: Optional[Dataset]
    test_dataset: Optional[Dataset]
    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    test_loader: Optional[DataLoader]
    stats_path: Optional[Path]


def validate_common_data_args(
    *,
    grad_accum_steps: int,
    val_every: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    if grad_accum_steps < 1:
        raise ValueError(f"--grad-accum-steps must be >= 1, got {grad_accum_steps}")
    if val_every < 1:
        raise ValueError(f"--val-every must be >= 1, got {val_every}")
    if not (0.0 < train_fraction <= 1.0):
        raise ValueError(f"--train-fraction must be in (0,1], got {train_fraction}")
    if not (0.0 < val_fraction <= 1.0):
        raise ValueError(f"--val-fraction must be in (0,1], got {val_fraction}")
    if not (0.0 < test_fraction <= 1.0):
        raise ValueError(f"--test-fraction must be in (0,1], got {test_fraction}")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"--val-ratio must be in [0,1), got {val_ratio}")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError(f"--test-ratio must be in [0,1), got {test_ratio}")
    if (val_ratio + test_ratio) >= 1.0:
        raise ValueError(f"--val-ratio + --test-ratio must be < 1, got {val_ratio + test_ratio}")


def write_config_json(*, out_dir: Path, args: Any) -> Path:
    config_path = out_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)
    return config_path


def append_jsonl(*, path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_data_loaders_from_args(
    *,
    args: Any,
    test_loader_note: str,
    norm_mean: Any = None,
    norm_std: Any = None,
) -> PreparedDataLoaders:
    bundle: DataBundle = build_train_val_test_dataloaders(
        db_path=args.db,
        seq_len=args.seq_len,
        stride=args.stride,
        enforce_same_label=args.enforce_same_label,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        load_into_memory=args.load_into_memory,
        normalize=args.normalize,
        norm_mean=norm_mean,
        norm_std=norm_std,
        norm_eps=args.norm_eps,
        max_windows_for_stats=args.max_windows_for_stats,
        stats_windows_chunk_size=args.stats_windows_chunk_size,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    train_dataset = subset_dataset_by_fraction(
        bundle.train_dataset,
        args.train_fraction,
        args.subset_seed,
    )
    val_dataset = subset_dataset_by_fraction(
        bundle.val_dataset,
        args.val_fraction,
        args.subset_seed + 1,
    )
    test_dataset = subset_dataset_by_fraction(
        bundle.test_dataset,
        args.test_fraction,
        args.subset_seed + 2,
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    print(f"[Info] train samples   : {len(bundle.train_dataset)} -> {len(train_dataset)}")
    if bundle.val_dataset is None:
        print("[Info] val samples     : 0 (disabled)")
    else:
        print(f"[Info] val samples     : {len(bundle.val_dataset)} -> {len(val_dataset)}")
    if bundle.test_dataset is None:
        print("[Info] test samples    : 0 (disabled)")
    else:
        print(f"[Info] test samples    : {len(bundle.test_dataset)} -> {len(test_dataset)}")
    if test_loader is not None and test_loader_note:
        print(f"[Info] test loader     : {test_loader_note}")
    print(f"[Info] normalize       : {args.normalize}")

    stats_path = save_norm_stats_if_present(
        out_dir=args.out_dir,
        norm_mean=bundle.norm_mean,
        norm_std=bundle.norm_std,
    )
    if stats_path is not None:
        print(f"[Info] norm stats      : {stats_path}")

    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training dataset is empty. Check --db / --seq-len / --stride / --enforce-same-label settings."
        )

    return PreparedDataLoaders(
        bundle=bundle,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        stats_path=stats_path,
    )


def save_training_checkpoint(
    *,
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    best_metric: float,
    args: Any,
) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "args": vars(args),
    }
    torch.save(payload, ckpt_path)


def resume_training_state(
    *,
    resume_path: Optional[Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler],
    reset_optimizer: bool,
    best_metric: float,
    state_dict_transform: Optional[Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]] = None,
    strict: bool = True,
) -> tuple[int, float]:
    start_epoch = 1
    if resume_path is None:
        return start_epoch, best_metric
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    ckpt = load_checkpoint(resume_path, map_location="cpu")
    model_state = ckpt["model"]
    if state_dict_transform is not None:
        model_state = state_dict_transform(model_state)
    model.load_state_dict(model_state, strict=strict)

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_metric = float(ckpt.get("best_metric", best_metric))

    if not reset_optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])

    print(f"[Info] resumed from    : {resume_path} (start_epoch={start_epoch})")
    return start_epoch, best_metric


def load_best_checkpoint_for_eval(
    *,
    model: torch.nn.Module,
    ckpt_path: Path,
    state_dict_transform: Optional[Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]] = None,
    strict: bool = True,
    allow_incompatible: bool = False,
) -> bool:
    if not ckpt_path.exists():
        print("[Warn] best.pt not found, evaluating test with current model weights.")
        return False

    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    model_state = ckpt["model"]
    if state_dict_transform is not None:
        model_state = state_dict_transform(model_state)

    try:
        model.load_state_dict(model_state, strict=strict)
    except RuntimeError as exc:
        if not allow_incompatible:
            raise
        print(
            "[Warn] best checkpoint state_dict is incompatible with current model "
            f"(likely from older architecture): {exc}"
        )
        print("[Warn] evaluating test with current in-memory model weights.")
        return False

    print(f"[Info] loaded best checkpoint for test: {ckpt_path}")
    return True
