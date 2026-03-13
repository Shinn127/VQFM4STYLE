from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantizeResult:
    quantized: torch.Tensor
    indices: torch.Tensor
    loss: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    perplexity: torch.Tensor


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, H, T, Dh]
        seq_len = x.size(-2)
        if position_ids is None:
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        else:
            if position_ids.dim() != 1 or position_ids.size(0) != seq_len:
                raise ValueError(
                    f"Expected position_ids shape [T]={seq_len}, got {tuple(position_ids.shape)}"
                )
            t = position_ids.to(device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)


class RoPEMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim({embed_dim}) must be divisible by num_heads({num_heads})"
            )
        head_dim = embed_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires even head_dim, got embed_dim/num_heads={head_dim}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dropout = dropout

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(head_dim=head_dim, base=rope_base)

    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected x shape [B,T,C], got {tuple(x.shape)}")
        b, t, c = x.shape
        if c != self.embed_dim:
            raise ValueError(f"Expected embedding dim {self.embed_dim}, got {c}")
        qkv = self.qkv_proj(x).view(b, t, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
            )
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        probs = F.dropout(probs, p=self.attn_dropout, training=self.training)
        return torch.matmul(probs, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._project_qkv(x)
        q = self.rope(q)
        k = self.rope(k)
        attn_out = self._attend(q, k, v)
        b, _, t, _ = attn_out.shape
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, self.embed_dim)
        return self.out_dropout(self.out_proj(attn_out))


class TransformerEncoderLayerRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = RoPEMultiHeadSelfAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            rope_base=rope_base,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class VectorQuantizer(nn.Module):
    """
    Multi-vector quantizer for per-frame latent tokens.

    Input latent is expected as [N, latent_dim], where latent_dim = num_codebooks * code_dim.
    Each frame token is split into `num_codebooks` subvectors and quantized independently.
    """

    def __init__(
        self,
        num_embeddings: int,
        code_dim: int,
        num_codebooks: int = 4,
        commitment_cost: float = 0.25,
        codebook_cost: float = 1.0,
        eps: float = 1e-10,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.code_dim = code_dim
        self.num_codebooks = num_codebooks
        self.latent_dim = num_codebooks * code_dim
        self.commitment_cost = commitment_cost
        self.codebook_cost = codebook_cost
        self.eps = eps
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.ema_eps = float(ema_eps)

        if not (0.0 <= self.ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}")
        if self.ema_eps <= 0.0:
            raise ValueError(f"ema_eps must be > 0, got {self.ema_eps}")

        self.embedding = nn.Embedding(num_embeddings, code_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

        if self.use_ema:
            self.embedding.weight.requires_grad_(False)
            self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
            self.register_buffer("ema_embed", self.embedding.weight.detach().clone())

    @torch.no_grad()
    def _ema_update(self, flat_z: torch.Tensor, flat_indices: torch.Tensor) -> None:
        one_hot = F.one_hot(flat_indices, self.num_embeddings).type_as(flat_z)
        cluster_size = one_hot.sum(dim=0)
        embed_sum = one_hot.transpose(0, 1) @ flat_z

        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1.0 - self.ema_decay)
        self.ema_embed.mul_(self.ema_decay).add_(embed_sum, alpha=1.0 - self.ema_decay)

        n = self.ema_cluster_size.sum()
        cluster_size = (self.ema_cluster_size + self.ema_eps) / (
            n + self.num_embeddings * self.ema_eps
        ) * n
        embed_normalized = self.ema_embed / cluster_size.unsqueeze(1)
        self.embedding.weight.data.copy_(embed_normalized)

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.size(-1) != self.num_codebooks:
            raise ValueError(
                f"indices last dim mismatch: got {indices.size(-1)}, expected {self.num_codebooks}"
            )
        codes = F.embedding(indices.long(), self.embedding.weight)
        return codes.reshape(*indices.shape[:-1], self.latent_dim)

    def forward(self, z_bt: torch.Tensor) -> QuantizeResult:
        if z_bt.dim() != 2:
            raise ValueError(f"Expected z_bt shape [N, latent_dim], got {tuple(z_bt.shape)}")
        if z_bt.size(-1) != self.latent_dim:
            raise ValueError(
                f"z_bt latent_dim mismatch: got {z_bt.size(-1)}, expected {self.latent_dim}"
            )

        z_codes = z_bt.reshape(-1, self.num_codebooks, self.code_dim)
        flat_z = z_codes.reshape(-1, self.code_dim)
        embed = self.embedding.weight

        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat_z @ embed.t()
            + embed.pow(2).sum(dim=1).unsqueeze(0)
        )

        flat_indices = torch.argmin(distances, dim=1)
        quantized_flat = F.embedding(flat_indices, embed)
        quantized_codes = quantized_flat.reshape(-1, self.num_codebooks, self.code_dim)
        quantized_bt = quantized_codes.reshape(-1, self.latent_dim)

        if self.use_ema and self.training:
            self._ema_update(flat_z.detach(), flat_indices.detach())

        if self.use_ema:
            codebook_loss = z_bt.new_zeros(())
        else:
            codebook_loss = F.mse_loss(quantized_bt, z_bt.detach())
        commitment_loss = F.mse_loss(quantized_bt.detach(), z_bt)
        loss = self.codebook_cost * codebook_loss + self.commitment_cost * commitment_loss
        quantized_st = z_bt + (quantized_bt - z_bt).detach()

        one_hot = F.one_hot(flat_indices, self.num_embeddings).to(z_bt.dtype)
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + self.eps).log()).sum())

        return QuantizeResult(
            quantized=quantized_st,
            indices=flat_indices.reshape(-1, self.num_codebooks),
            loss=loss,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            perplexity=perplexity,
        )


class VQVAE375to291(nn.Module):
    """
    VQ-VAE for reconstructing 291-d motion from 375-d traj+motion features.

    Feature layout:
      - [0:84]   trajectory condition
      - [84:375] motion target (291 dims)
    """

    def __init__(
        self,
        input_dim: int = 375,
        traj_dim: int = 84,
        motion_dim: int = 291,
        hidden_dim: int = 512,
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        encoder_ffn_dim: int = 1024,
        decoder_hidden_dim: int = 512,
        decoder_layers: int = 4,
        num_latents: int = 4,
        code_dim: int = 64,
        num_embeddings: int = 1024,
        commitment_cost: float = 0.25,
        codebook_cost: float = 1.0,
        use_ema_codebook: bool = True,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        rope_base: float = 10000.0,
    ):
        super().__init__()

        if traj_dim + motion_dim != input_dim:
            raise ValueError(
                f"Expected traj_dim + motion_dim == input_dim, got "
                f"{traj_dim} + {motion_dim} != {input_dim}"
            )
        if encoder_layers < 1:
            raise ValueError("encoder_layers must be >= 1")
        if decoder_layers < 1:
            raise ValueError("decoder_layers must be >= 1")
        if num_latents < 1:
            raise ValueError("num_latents must be >= 1")
        if code_dim < 1:
            raise ValueError("code_dim must be >= 1")
        if hidden_dim % encoder_heads != 0:
            raise ValueError(
                f"hidden_dim({hidden_dim}) must be divisible by encoder_heads({encoder_heads})"
            )

        self.input_dim = input_dim
        self.traj_dim = traj_dim
        self.motion_dim = motion_dim
        self.num_latents = num_latents
        self.code_dim = code_dim
        self.latent_dim = num_latents * code_dim

        self.encoder_input_norm = nn.LayerNorm(input_dim)
        self.encoder_input_proj = nn.Linear(input_dim, hidden_dim)
        self.max_seq_len = max_seq_len
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayerRoPE(
                    d_model=hidden_dim,
                    nhead=encoder_heads,
                    dim_feedforward=encoder_ffn_dim,
                    dropout=dropout,
                    rope_base=rope_base,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.encoder_out_norm = nn.LayerNorm(hidden_dim)
        self.encoder_out = nn.Linear(hidden_dim, self.latent_dim)

        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            code_dim=code_dim,
            num_codebooks=num_latents,
            commitment_cost=commitment_cost,
            codebook_cost=codebook_cost,
            use_ema=use_ema_codebook,
            ema_decay=ema_decay,
            ema_eps=ema_eps,
        )

        decoder_input_dim = self.latent_dim + traj_dim
        self.decoder_in = nn.Linear(decoder_input_dim, decoder_hidden_dim)
        self.decoder_blocks = nn.Sequential(
            *[
                ResidualMLPBlock(decoder_hidden_dim, decoder_hidden_dim * 2, dropout=dropout)
                for _ in range(decoder_layers)
            ]
        )
        self.decoder_out_norm = nn.LayerNorm(decoder_hidden_dim)
        self.decoder_out = nn.Linear(decoder_hidden_dim, motion_dim)

    def split_traj_motion(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected last dim == {self.input_dim}, got {x.size(-1)}"
            )
        return x[..., : self.traj_dim], x[..., self.traj_dim :]

    def _to_btd(self, x: torch.Tensor, name: str, expected_last_dim: int) -> tuple[torch.Tensor, bool]:
        if x.size(-1) != expected_last_dim:
            raise ValueError(
                f"Expected {name} last dim == {expected_last_dim}, got {x.size(-1)}"
            )
        if x.dim() == 2:
            return x.unsqueeze(1), True
        if x.dim() == 3:
            return x, False
        raise ValueError(
            f"Expected {name} with shape [B,{expected_last_dim}] or [B,T,{expected_last_dim}], got {tuple(x.shape)}"
        )

    @staticmethod
    def _restore_from_btd(x: torch.Tensor, squeezed_time: bool) -> torch.Tensor:
        return x[:, 0] if squeezed_time else x

    @staticmethod
    def _ensure_3d(x: torch.Tensor, *, name: str) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(1)
        if x.dim() == 3:
            return x
        raise ValueError(f"Expected {name} dims to be 2 or 3, got {x.dim()}")

    @staticmethod
    def _to_btd_latents(
        latents: torch.Tensor,
        expected_last_dim: int,
        b: int,
        t: int,
        name: str,
    ) -> torch.Tensor:
        if latents.dim() == 2:
            if latents.size(-1) != expected_last_dim:
                raise ValueError(
                    f"Expected {name} last dim == {expected_last_dim}, got {latents.size(-1)}"
                )
            if latents.size(0) == b * t:
                return latents.reshape(b, t, expected_last_dim)
            if t == 1 and latents.size(0) == b:
                return latents.unsqueeze(1)
            raise ValueError(
                f"{name} batch mismatch: got first dim {latents.size(0)}, expected {b*t} "
                f"(or {b} when T=1)"
            )

        if latents.dim() == 3:
            if latents.shape != (b, t, expected_last_dim):
                raise ValueError(
                    f"Expected {name} shape {(b, t, expected_last_dim)}, got {tuple(latents.shape)}"
                )
            return latents

        raise ValueError(
            f"Expected {name} shape [B*T,{expected_last_dim}] or [B,T,{expected_last_dim}], got {tuple(latents.shape)}"
        )

    def _resolve_latents_btd(
        self,
        *,
        latents: torch.Tensor,
        from_indices: bool,
        b: int,
        t: int,
    ) -> torch.Tensor:
        if from_indices:
            indices_btd = self._to_btd_latents(
                latents,
                expected_last_dim=self.num_latents,
                b=b,
                t=t,
                name="indices",
            )
            return self.quantizer.lookup(indices_btd.long())
        return self._to_btd_latents(
            latents,
            expected_last_dim=self.latent_dim,
            b=b,
            t=t,
            name="latents",
        )

    def _encode_to_bt(self, x_btd: torch.Tensor) -> torch.Tensor:
        if x_btd.size(1) > self.max_seq_len:
            raise ValueError(
                f"Sequence length {x_btd.size(1)} exceeds configured max_seq_len {self.max_seq_len}"
            )
        h = self.encoder_input_proj(self.encoder_input_norm(x_btd))
        for layer in self.encoder_layers:
            h = layer(h)
        h = self.encoder_out_norm(h)
        z_btd = self.encoder_out(h)
        return z_btd.reshape(-1, self.latent_dim)

    def _decode_btd(self, latent_btd: torch.Tensor, traj_btd: torch.Tensor) -> torch.Tensor:
        if latent_btd.dim() != 3:
            raise ValueError(
                f"Expected latent_btd shape [B,T,{self.latent_dim}], got {tuple(latent_btd.shape)}"
            )
        if latent_btd.size(-1) != self.latent_dim:
            raise ValueError(
                f"Expected latent_btd last dim {self.latent_dim}, got {latent_btd.size(-1)}"
            )
        if traj_btd.dim() != 3:
            raise ValueError(
                f"Expected traj_btd shape [B,T,{self.traj_dim}], got {tuple(traj_btd.shape)}"
            )
        if traj_btd.size(-1) != self.traj_dim:
            raise ValueError(
                f"Expected traj_btd last dim {self.traj_dim}, got {traj_btd.size(-1)}"
            )
        if latent_btd.shape[:2] != traj_btd.shape[:2]:
            raise ValueError(
                f"latent/traj sequence mismatch: {tuple(latent_btd.shape[:2])} vs {tuple(traj_btd.shape[:2])}"
            )

        decoder_in = torch.cat([latent_btd, traj_btd], dim=-1)
        h = F.gelu(self.decoder_in(decoder_in))
        h = self.decoder_blocks(h)

        h = self.decoder_out(self.decoder_out_norm(h))
        return h

    def _decode_single_frame(
        self,
        latent_bt: torch.Tensor,
        traj_bt: torch.Tensor,
    ) -> torch.Tensor:
        if latent_bt.dim() != 2:
            raise ValueError(f"Expected latent_bt shape [B,{self.latent_dim}], got {tuple(latent_bt.shape)}")
        if latent_bt.size(-1) != self.latent_dim:
            raise ValueError(f"Expected latent_bt last dim {self.latent_dim}, got {latent_bt.size(-1)}")
        if traj_bt.dim() != 2:
            raise ValueError(f"Expected traj_bt shape [B,{self.traj_dim}], got {tuple(traj_bt.shape)}")
        if traj_bt.size(-1) != self.traj_dim:
            raise ValueError(f"Expected traj_bt last dim {self.traj_dim}, got {traj_bt.size(-1)}")
        if latent_bt.size(0) != traj_bt.size(0):
            raise ValueError(
                f"latent/traj batch mismatch: {latent_bt.size(0)} vs {traj_bt.size(0)}"
            )
        return self._decode_btd(latent_bt.unsqueeze(1), traj_bt.unsqueeze(1))[:, 0]

    def forward(self, x: torch.Tensor, return_details: bool = False):
        x_btd, squeezed_time = self._to_btd(x, name="x", expected_last_dim=self.input_dim)
        b, t, _ = x_btd.shape
        traj_btd, _ = self.split_traj_motion(x_btd)

        # Encode sequence with bidirectional Transformer, then flatten to [B*T, D] before VQ.
        z_bt = self._encode_to_bt(x_btd)
        q = self.quantizer(z_bt)
        quantized_btd = q.quantized.reshape(b, t, self.latent_dim)

        motion_hat = self._decode_btd(quantized_btd, traj_btd)
        motion_hat = self._restore_from_btd(motion_hat, squeezed_time)

        quantized_latent = quantized_btd
        quantized_latent = self._restore_from_btd(quantized_latent, squeezed_time)

        indices = q.indices.reshape(b, t, self.num_latents)
        indices = self._restore_from_btd(indices, squeezed_time)

        if return_details:
            return {
                "motion_hat": motion_hat,
                "quantized_latent": quantized_latent,
                "indices": indices,
                "vq_loss": q.loss,
                "codebook_loss": q.codebook_loss,
                "commitment_loss": q.commitment_loss,
                "perplexity": q.perplexity,
            }

        return motion_hat, q.loss, q.perplexity

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_btd, squeezed_time = self._to_btd(x, name="x", expected_last_dim=self.input_dim)
        b, t, _ = x_btd.shape

        z_bt = self._encode_to_bt(x_btd)
        q = self.quantizer(z_bt)

        quantized = q.quantized.reshape(b, t, self.latent_dim)
        quantized = self._restore_from_btd(quantized, squeezed_time)

        indices = q.indices.reshape(b, t, self.num_latents)
        indices = self._restore_from_btd(indices, squeezed_time)

        return quantized, indices, q.perplexity

    @torch.no_grad()
    def decode_step(
        self,
        latents: torch.Tensor,
        traj: torch.Tensor,
        *,
        from_indices: bool = False,
    ) -> torch.Tensor:
        traj_bt, _ = self._to_btd(traj, name="traj", expected_last_dim=self.traj_dim)
        if traj_bt.size(1) != 1:
            raise ValueError(
                f"decode_step expects single-frame traj with T=1, got shape {tuple(traj_bt.shape)}"
            )
        b = traj_bt.size(0)
        traj_2d = traj_bt[:, 0]
        latent_bt = self._resolve_latents_btd(
            latents=latents,
            from_indices=from_indices,
            b=b,
            t=1,
        )[:, 0]

        return self._decode_single_frame(latent_bt, traj_2d)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, traj: torch.Tensor, from_indices: bool = False) -> torch.Tensor:
        traj_btd, squeezed_time = self._to_btd(traj, name="traj", expected_last_dim=self.traj_dim)
        b, t, _ = traj_btd.shape
        latent_btd = self._resolve_latents_btd(
            latents=latents,
            from_indices=from_indices,
            b=b,
            t=t,
        )

        motion_hat = self._decode_btd(latent_btd, traj_btd)
        return self._restore_from_btd(motion_hat, squeezed_time)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        x_btd, squeezed_time = self._to_btd(x, name="x", expected_last_dim=self.input_dim)
        b, t, _ = x_btd.shape
        motion_frames: list[torch.Tensor] = []
        for frame_idx in range(t):
            x_frame = x_btd[:, frame_idx]
            traj_frame, _ = self.split_traj_motion(x_frame)
            latent_frame, _, _ = self.encode(x_frame)
            motion_frame = self.decode_step(latent_frame, traj_frame, from_indices=False)
            motion_frames.append(motion_frame)
        motion_hat = torch.stack(motion_frames, dim=1)
        return self._restore_from_btd(motion_hat, squeezed_time)

    @staticmethod
    def _loss_by_type(
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_type: Literal["l1", "mse", "smooth_l1"],
    ) -> torch.Tensor:
        if loss_type == "l1":
            return F.l1_loss(pred, target)
        if loss_type == "mse":
            return F.mse_loss(pred, target)
        if loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target)
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    def compute_loss(
        self,
        x: torch.Tensor,
        target_motion: torch.Tensor | None = None,
        recon_type: Literal["l1", "mse", "smooth_l1"] = "l1",
        recon_weight: float = 1.0,
        vq_weight: float = 1.0,
        vel_weight: float = 0.0,
        acc_weight: float = 0.0,
        lat_acc_weight: float = 0.0,
        temporal_loss_type: Literal["l1", "mse", "smooth_l1"] | None = None,
    ) -> dict[str, torch.Tensor]:
        details = self.forward(x, return_details=True)
        pred = details["motion_hat"]
        z_q = details["quantized_latent"]

        if target_motion is None:
            _, target_motion = self.split_traj_motion(x)

        if target_motion.shape != pred.shape:
            raise ValueError(
                f"target_motion shape mismatch: expected {tuple(pred.shape)}, got {tuple(target_motion.shape)}"
            )

        recon_loss = self._loss_by_type(pred, target_motion, recon_type)

        if temporal_loss_type is None:
            temporal_loss_type = recon_type

        pred_bt = self._ensure_3d(pred, name="pred")
        target_bt = self._ensure_3d(target_motion, name="target_motion")
        z_q_bt = self._ensure_3d(z_q, name="quantized_latent")

        vel_loss = pred.new_zeros(())
        acc_loss = pred.new_zeros(())
        lat_acc_loss = pred.new_zeros(())

        if vel_weight > 0.0 and pred_bt.size(1) >= 2:
            pred_vel = pred_bt[:, 1:] - pred_bt[:, :-1]
            target_vel = target_bt[:, 1:] - target_bt[:, :-1]
            vel_loss = self._loss_by_type(pred_vel, target_vel, temporal_loss_type)

        if acc_weight > 0.0 and pred_bt.size(1) >= 3:
            pred_acc = pred_bt[:, 2:] - 2.0 * pred_bt[:, 1:-1] + pred_bt[:, :-2]
            target_acc = target_bt[:, 2:] - 2.0 * target_bt[:, 1:-1] + target_bt[:, :-2]
            acc_loss = self._loss_by_type(pred_acc, target_acc, temporal_loss_type)

        if lat_acc_weight > 0.0 and z_q_bt.size(1) >= 3:
            # Penalize second-order temporal changes in quantized latent sequence.
            # This suppresses rapid code/latent switching that often causes frame jitter.
            lat_acc = z_q_bt[:, 2:] - 2.0 * z_q_bt[:, 1:-1] + z_q_bt[:, :-2]
            lat_acc_loss = self._loss_by_type(lat_acc, torch.zeros_like(lat_acc), temporal_loss_type)

        total = (
            recon_weight * recon_loss
            + vq_weight * details["vq_loss"]
            + vel_weight * vel_loss
            + acc_weight * acc_loss
            + lat_acc_weight * lat_acc_loss
        )
        return {
            "loss": total,
            "recon_loss": recon_loss,
            "vq_loss": details["vq_loss"],
            "vel_loss": vel_loss,
            "acc_loss": acc_loss,
            "lat_acc_loss": lat_acc_loss,
            "codebook_loss": details["codebook_loss"],
            "commitment_loss": details["commitment_loss"],
            "perplexity": details["perplexity"],
        }


__all__ = [
    "QuantizeResult",
    "VectorQuantizer",
    "VQVAE375to291",
]
