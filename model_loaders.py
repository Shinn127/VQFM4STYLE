from __future__ import annotations

from pathlib import Path

from arch import LMMCompressorDecompressorAE, MVAE375to291, VQVAE375to291
from runtime_utils import load_checkpoint, sanitize_state_dict_keys


def build_vqvae_from_checkpoint(
    ckpt_path: Path,
    *,
    map_location: str = "cpu",
    freeze: bool = True,
) -> VQVAE375to291:
    ckpt = load_checkpoint(ckpt_path, map_location=map_location)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' key: {ckpt_path}")
    ckpt_args = ckpt.get("args", {}) or {}

    decoder_arch = str(ckpt_args.get("decoder_arch", "mlp"))
    if decoder_arch != "mlp":
        raise ValueError(
            f"Checkpoint uses decoder_arch={decoder_arch!r}, but restored VQVAE only supports "
            "decoder_arch='mlp'. Please provide/retrain an MLP-decoder checkpoint."
        )

    model = VQVAE375to291(
        input_dim=int(ckpt_args.get("input_dim", 375)),
        traj_dim=int(ckpt_args.get("traj_dim", 84)),
        motion_dim=int(ckpt_args.get("motion_dim", 291)),
        hidden_dim=int(ckpt_args.get("hidden_dim", 512)),
        encoder_layers=int(ckpt_args.get("encoder_layers", 6)),
        encoder_heads=int(ckpt_args.get("encoder_heads", 8)),
        encoder_ffn_dim=int(ckpt_args.get("encoder_ffn_dim", 1024)),
        decoder_hidden_dim=int(ckpt_args.get("decoder_hidden_dim", 512)),
        decoder_layers=int(ckpt_args.get("decoder_layers", 4)),
        num_latents=int(ckpt_args.get("num_latents", 4)),
        code_dim=int(ckpt_args.get("code_dim", 64)),
        num_embeddings=int(ckpt_args.get("num_embeddings", 1024)),
        commitment_cost=float(ckpt_args.get("commitment_cost", 0.25)),
        codebook_cost=float(ckpt_args.get("codebook_cost", 1.0)),
        use_ema_codebook=bool(ckpt_args.get("use_ema_codebook", True)),
        ema_decay=float(ckpt_args.get("ema_decay", 0.99)),
        ema_eps=float(ckpt_args.get("ema_eps", 1e-5)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
        max_seq_len=int(ckpt_args.get("max_seq_len", 4096)),
        rope_base=float(ckpt_args.get("rope_base", 10000.0)),
    )
    model.load_state_dict(sanitize_state_dict_keys(ckpt["model"]), strict=True)
    model.eval()
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def build_lmm_from_checkpoint(
    ckpt_path: Path,
    *,
    map_location: str = "cpu",
) -> LMMCompressorDecompressorAE:
    ckpt = load_checkpoint(ckpt_path, map_location=map_location)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' key: {ckpt_path}")
    ckpt_args = ckpt.get("args", {}) or {}

    model = LMMCompressorDecompressorAE(
        motion_dim=int(ckpt_args.get("motion_dim", 291)),
        traj_dim=int(ckpt_args.get("traj_dim", 84)),
        latent_dim=int(ckpt_args.get("latent_dim", 32)),
        hidden_dim=int(ckpt_args.get("hidden_dim", 512)),
    )
    model.load_state_dict(sanitize_state_dict_keys(ckpt["model"]), strict=True)
    model.eval()
    return model


def build_mvae_from_checkpoint(
    ckpt_path: Path,
    *,
    map_location: str = "cpu",
) -> MVAE375to291:
    ckpt = load_checkpoint(ckpt_path, map_location=map_location)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' key: {ckpt_path}")
    ckpt_args = ckpt.get("args", {}) or {}

    model = MVAE375to291(
        input_dim=int(ckpt_args.get("input_dim", 375)),
        traj_dim=int(ckpt_args.get("traj_dim", 84)),
        motion_dim=int(ckpt_args.get("motion_dim", 291)),
        latent_dim=int(ckpt_args.get("latent_dim", 32)),
        hidden_dim=int(ckpt_args.get("hidden_dim", 256)),
        gate_hidden_dim=int(ckpt_args.get("gate_hidden_dim", 64)),
        num_condition_frames=int(ckpt_args.get("num_condition_frames", 1)),
        num_future_predictions=int(ckpt_args.get("num_future_predictions", 1)),
        num_experts=int(ckpt_args.get("num_experts", 6)),
    )
    model.load_state_dict(sanitize_state_dict_keys(ckpt["model"]), strict=True)
    model.eval()
    return model
