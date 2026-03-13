from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class CommonDataDefaults:
    seq_len: int
    stride: int
    batch_size: int
    max_windows_for_stats: int
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    num_workers: int = 8
    prefetch_factor: int = 4
    norm_eps: float = 1e-8
    train_fraction: float = 1.0
    val_fraction: float = 1.0
    test_fraction: float = 1.0
    subset_seed: int = 123


@dataclass(frozen=True)
class CommonTrainDefaults:
    epochs: int = 50
    lr: float = 2e-4
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.99
    grad_accum_steps: int = 2
    grad_clip: float = 1.0


@dataclass(frozen=True)
class CommonRuntimeDefaults:
    amp: bool = True
    tf32: bool = True
    cudnn_benchmark: bool = True
    fused_adamw: bool = True
    matmul_precision: str = "high"
    compile_mode: str = "reduce-overhead"
    seed: int = 42
    device: str = "auto"


@dataclass(frozen=True)
class CommonLoggingDefaults:
    save_every: int = 10
    val_every: int = 5
    log_interval: int = 100


def add_bool_arg(
    parser: Any,
    *,
    name: str,
    default: bool,
    help_enable: str | None = None,
    help_disable: str | None = None,
) -> None:
    dest = name.replace("-", "_")
    parser.set_defaults(**{dest: bool(default)})
    parser.add_argument(f"--{name}", dest=dest, action="store_true", help=help_enable)
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false", help=help_disable)


def add_common_io_args(
    parser: Any,
    *,
    script_dir: Path,
    out_dir_name: str,
    include_db: bool = True,
) -> None:
    if include_db:
        parser.add_argument(
            "--db",
            type=Path,
            default=script_dir / "database_100sty_375_memmap",
            help="Path to memmap directory",
        )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=script_dir / "checkpoints" / out_dir_name,
    )
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--reset-optimizer", action="store_true", help="Ignore optimizer/scheduler states on resume")


def add_common_data_args(
    parser: Any,
    *,
    defaults: CommonDataDefaults,
) -> None:
    parser.add_argument("--seq-len", type=int, default=defaults.seq_len)
    parser.add_argument("--stride", type=int, default=defaults.stride)
    add_bool_arg(parser, name="enforce-same-label", default=True)
    parser.add_argument("--val-ratio", type=float, default=defaults.val_ratio)
    parser.add_argument("--test-ratio", type=float, default=defaults.test_ratio)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    add_bool_arg(parser, name="persistent-workers", default=True)
    parser.add_argument("--prefetch-factor", type=int, default=defaults.prefetch_factor)
    add_bool_arg(parser, name="pin-memory", default=True)
    parser.add_argument("--load-into-memory", action="store_true")
    add_bool_arg(parser, name="normalize", default=True)
    parser.add_argument("--norm-eps", type=float, default=defaults.norm_eps)
    parser.add_argument(
        "--max-windows-for-stats",
        type=int,
        default=defaults.max_windows_for_stats,
        help="0 means use all train windows to estimate normalization stats",
    )
    parser.add_argument("--stats-windows-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=defaults.train_fraction,
        help="Use only this fraction of train data for quick tests",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=defaults.val_fraction,
        help="Use only this fraction of val data",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=defaults.test_fraction,
        help="Use only this fraction of test data",
    )
    parser.add_argument("--subset-seed", type=int, default=defaults.subset_seed)


def add_common_train_args(
    parser: Any,
    *,
    defaults: CommonTrainDefaults,
) -> None:
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--min-lr", type=float, default=defaults.min_lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--beta1", type=float, default=defaults.beta1)
    parser.add_argument("--beta2", type=float, default=defaults.beta2)
    parser.add_argument("--grad-accum-steps", type=int, default=defaults.grad_accum_steps)
    parser.add_argument("--grad-clip", type=float, default=defaults.grad_clip)


def add_common_runtime_args(
    parser: Any,
    *,
    defaults: CommonRuntimeDefaults,
) -> None:
    add_bool_arg(parser, name="amp", default=defaults.amp)
    parser.add_argument(
        "--amp-dtype",
        type=str,
        choices=["auto", "fp16", "bf16"],
        default="auto",
        help="AMP compute dtype on CUDA",
    )
    add_bool_arg(parser, name="tf32", default=defaults.tf32)
    add_bool_arg(parser, name="cudnn-benchmark", default=defaults.cudnn_benchmark)
    parser.add_argument(
        "--matmul-precision",
        type=str,
        choices=["highest", "high", "medium"],
        default=defaults.matmul_precision,
    )
    add_bool_arg(parser, name="fused-adamw", default=defaults.fused_adamw)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for the model")
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["default", "reduce-overhead", "max-autotune"],
        default=defaults.compile_mode,
    )
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", type=str, default=defaults.device, help="auto/cuda/cpu")


def add_common_logging_args(
    parser: Any,
    *,
    defaults: CommonLoggingDefaults,
) -> None:
    parser.add_argument("--save-every", type=int, default=defaults.save_every)
    parser.add_argument("--val-every", type=int, default=defaults.val_every, help="Run validation every N epochs")
    parser.add_argument("--log-interval", type=int, default=defaults.log_interval)


def canonicalize_path_args(
    args: argparse.Namespace,
    *,
    extra_fields: Iterable[str] = (),
) -> None:
    for field in ("db", "out_dir", "resume", *tuple(extra_fields)):
        if not hasattr(args, field):
            continue
        value = getattr(args, field)
        if value is None or not isinstance(value, Path):
            continue
        setattr(args, field, value.expanduser().resolve())
