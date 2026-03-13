from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset


MEMMAP_META_FILE = "meta.json"
MEMMAP_X_FILE = "X.f32"
MEMMAP_LABELS_FILE = "Labels.i32"
MEMMAP_STYLES_FILE = "StyleNames.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_amp_dtype(
    amp_dtype_arg: str,
    device: torch.device,
) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    key = amp_dtype_arg.lower().strip()
    if key == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype: {amp_dtype_arg}")


def configure_torch_runtime(
    *,
    device: torch.device,
    tf32: bool,
    cudnn_benchmark: bool,
    matmul_precision: str,
) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    torch.set_float32_matmul_precision(matmul_precision)


def estimate_path_size_bytes(path: Path) -> float:
    if path.is_file():
        return float(path.stat().st_size)
    if path.is_dir():
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return float(total)
    return float("nan")


def to_float(v: torch.Tensor) -> float:
    return float(v.detach().cpu().item())


def validate_memmap_db_dir(db_path: Path) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"--db not found: {db_path}")
    if not db_path.is_dir():
        raise ValueError(f"--db must be a memmap directory, got file path: {db_path}")
    required = [MEMMAP_META_FILE, MEMMAP_X_FILE, MEMMAP_LABELS_FILE]
    missing = [name for name in required if not (db_path / name).exists()]
    if missing:
        raise ValueError(f"Invalid memmap directory: {db_path}. Missing files: {missing}")


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_state_dict_prefix_if_present(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def sanitize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state_dict = strip_state_dict_prefix_if_present(state_dict, "_orig_mod.")
    state_dict = strip_state_dict_prefix_if_present(state_dict, "module.")
    return state_dict


def subset_dataset_by_fraction(
    dataset: Optional[Dataset],
    fraction: float,
    seed: int,
) -> Optional[Dataset]:
    if dataset is None or fraction >= 1.0:
        return dataset
    n = len(dataset)
    if n == 0:
        return dataset
    keep = max(1, int(round(n * fraction)))
    if keep >= n:
        return dataset
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=keep, replace=False)
    return Subset(dataset, idx.tolist())


def build_dataloader(
    dataset: Optional[Dataset],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> Optional[DataLoader]:
    if dataset is None:
        return None
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **kwargs)


def build_adamw(
    parameters: Iterable[torch.nn.Parameter],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    use_fused: bool,
) -> tuple[AdamW, bool]:
    params = list(parameters)
    try:
        optimizer = AdamW(
            params,
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            fused=bool(use_fused),
        )
        return optimizer, bool(use_fused)
    except TypeError:
        optimizer = AdamW(
            params,
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )
        return optimizer, False


def save_norm_stats_if_present(
    *,
    out_dir: Path,
    norm_mean: Optional[np.ndarray],
    norm_std: Optional[np.ndarray],
) -> Optional[Path]:
    if norm_mean is None or norm_std is None:
        return None
    stats_path = out_dir / "norm_stats.npz"
    np.savez_compressed(
        stats_path,
        mean=np.asarray(norm_mean, dtype=np.float32),
        std=np.asarray(norm_std, dtype=np.float32),
    )
    return stats_path


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
