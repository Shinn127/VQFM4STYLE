from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vqvae_375to291 import VQVAE375to291


def _timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    if t.dim() == 2 and t.size(-1) == 1:
        t = t[:, 0]
    if t.dim() != 1:
        raise ValueError(f"Expected t shape [N] or [N,1], got {tuple(t.shape)}")
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")

    half = dim // 2
    if half == 0:
        return t.unsqueeze(-1)

    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / float(half)
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat((args.sin(), args.cos()), dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class ContentEncoder(nn.Module):
    """
    E_c: encode current frame content (traj_i + motion_i) to condition vector.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        content_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.content_dim = int(content_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.content_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected x shape [N,{self.input_dim}], got {tuple(x.shape)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected x last dim {self.input_dim}, got {x.size(-1)}")
        return self.net(x)


class StyleEncoder(nn.Module):
    """
    E_s: encode discrete style label to style embedding.
    """

    def __init__(self, num_styles: int, style_dim: int):
        super().__init__()
        if num_styles < 1:
            raise ValueError(f"num_styles must be >= 1, got {num_styles}")
        if style_dim < 1:
            raise ValueError(f"style_dim must be >= 1, got {style_dim}")
        self.num_styles = int(num_styles)
        self.style_dim = int(style_dim)
        self.embedding = nn.Embedding(self.num_styles, self.style_dim)

    def forward(self, style_label: torch.Tensor) -> torch.Tensor:
        if style_label.dim() != 1:
            raise ValueError(
                f"Expected style_label shape [N], got {tuple(style_label.shape)}"
            )
        if style_label.numel() == 0:
            raise ValueError("style_label is empty")
        if style_label.min() < 0:
            raise ValueError("style_label must be non-negative")
        if style_label.max() >= self.num_styles:
            raise ValueError(
                f"style_label out of range: max={int(style_label.max())}, num_styles={self.num_styles}"
            )
        return self.embedding(style_label.long())


class ConditionalVelocityField(nn.Module):
    """
    v_theta(z_t, t, e_c, e_s): conditional velocity model for flow matching.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        content_dim: int,
        style_dim: int,
        time_embed_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.latent_dim = int(latent_dim)
        self.content_dim = int(content_dim)
        self.style_dim = int(style_dim)
        self.time_embed_dim = int(time_embed_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        in_dim = self.latent_dim + self.content_dim + self.style_dim + self.time_embed_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[
                ResidualMLPBlock(
                    dim=hidden_dim,
                    hidden_dim=hidden_dim * 2,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.latent_dim)

    def forward(
        self,
        *,
        z_t: torch.Tensor,
        t: torch.Tensor,
        e_c: torch.Tensor,
        e_s: torch.Tensor,
    ) -> torch.Tensor:
        if z_t.dim() != 2:
            raise ValueError(f"Expected z_t shape [N,{self.latent_dim}], got {tuple(z_t.shape)}")
        if z_t.size(-1) != self.latent_dim:
            raise ValueError(f"Expected z_t last dim {self.latent_dim}, got {z_t.size(-1)}")
        if e_c.shape != (z_t.size(0), self.content_dim):
            raise ValueError(
                f"Expected e_c shape {(z_t.size(0), self.content_dim)}, got {tuple(e_c.shape)}"
            )
        if e_s.shape != (z_t.size(0), self.style_dim):
            raise ValueError(
                f"Expected e_s shape {(z_t.size(0), self.style_dim)}, got {tuple(e_s.shape)}"
            )

        t_emb = _timestep_embedding(t, dim=self.time_embed_dim).to(dtype=z_t.dtype)
        t_emb = self.time_mlp(t_emb)

        h = torch.cat((z_t, e_c, e_s, t_emb), dim=-1)
        h = F.silu(self.in_proj(h))
        h = self.blocks(h)
        h = self.out_proj(self.out_norm(h))
        return h


class FlowMatchVQVAEController(nn.Module):
    """
    Style-conditioned next-frame motion controller using flow matching latent generation.

    Pipeline:
      1) e_c = E_c(traj_i, motion_i)
      2) e_s = E_s(style_label)
      3) Flow matching generator predicts VQ latent/codebook vector for frame i+1
      4) Frozen VQ-VAE decoder reconstructs motion_{i+1} from (traj_{i+1}, latent)
    """

    def __init__(
        self,
        *,
        vqvae: VQVAE375to291,
        num_styles: int,
        content_dim: int = 256,
        style_dim: int = 128,
        time_embed_dim: int = 64,
        encoder_hidden_dim: int = 512,
        flow_hidden_dim: int = 512,
        flow_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vqvae = vqvae
        self.vqvae.eval()
        for p in self.vqvae.parameters():
            p.requires_grad_(False)

        self.input_dim = int(self.vqvae.input_dim)
        self.traj_dim = int(self.vqvae.traj_dim)
        self.motion_dim = int(self.vqvae.motion_dim)
        self.latent_dim = int(self.vqvae.latent_dim)

        self.num_styles = int(num_styles)
        self.content_dim = int(content_dim)
        self.style_dim = int(style_dim)
        self.time_embed_dim = int(time_embed_dim)

        self.content_encoder = ContentEncoder(
            input_dim=self.input_dim,
            hidden_dim=encoder_hidden_dim,
            content_dim=self.content_dim,
            dropout=dropout,
        )
        self.style_encoder = StyleEncoder(
            num_styles=self.num_styles,
            style_dim=self.style_dim,
        )
        self.velocity_field = ConditionalVelocityField(
            latent_dim=self.latent_dim,
            content_dim=self.content_dim,
            style_dim=self.style_dim,
            time_embed_dim=self.time_embed_dim,
            hidden_dim=flow_hidden_dim,
            num_layers=flow_layers,
            dropout=dropout,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep VQ-VAE frozen in eval mode to avoid EMA/codebook updates.
        self.vqvae.eval()
        return self

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

    def split_traj_motion(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected x last dim {self.input_dim}, got {x.size(-1)}")
        return x[..., : self.traj_dim], x[..., self.traj_dim :]

    def _to_2d_frame_input(self, x: torch.Tensor, name: str, expected_dim: int) -> torch.Tensor:
        if x.size(-1) != expected_dim:
            raise ValueError(f"Expected {name} last dim {expected_dim}, got {x.size(-1)}")
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            return x.reshape(-1, expected_dim)
        raise ValueError(
            f"Expected {name} shape [B,{expected_dim}] or [B,T,{expected_dim}], got {tuple(x.shape)}"
        )

    @staticmethod
    def _to_style_labels(style_label: torch.Tensor, n: int) -> torch.Tensor:
        if style_label.dim() == 0:
            style_label = style_label.unsqueeze(0)
        if style_label.dim() == 2 and style_label.size(-1) == 1:
            style_label = style_label[:, 0]
        if style_label.dim() != 1:
            raise ValueError(f"Expected style_label rank 1, got {tuple(style_label.shape)}")
        if style_label.numel() == n:
            return style_label.long()
        if style_label.numel() > 0 and (n % style_label.numel() == 0):
            repeat = n // style_label.numel()
            return style_label.long().repeat_interleave(repeat)
        raise ValueError(
            f"Cannot align style_label count {style_label.numel()} with target count {n}"
        )

    def encode_conditions(
        self,
        *,
        x_curr: torch.Tensor,
        style_label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_curr_bt = self._to_2d_frame_input(x_curr, name="x_curr", expected_dim=self.input_dim)
        style_bt = self._to_style_labels(style_label.to(device=x_curr_bt.device), n=x_curr_bt.size(0))
        e_c = self.content_encoder(x_curr_bt)
        e_s = self.style_encoder(style_bt)
        return e_c, e_s

    def velocity(
        self,
        *,
        z_t: torch.Tensor,
        t: torch.Tensor,
        e_c: torch.Tensor,
        e_s: torch.Tensor,
    ) -> torch.Tensor:
        return self.velocity_field(z_t=z_t, t=t, e_c=e_c, e_s=e_s)

    def integrate_euler(
        self,
        *,
        e_c: torch.Tensor,
        e_s: torch.Tensor,
        z0: torch.Tensor | None = None,
        num_steps: int = 16,
        noise_std: float = 1.0,
        solver: Literal["euler", "midpoint", "heun"] = "euler",
    ) -> torch.Tensor:
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        if noise_std <= 0.0:
            raise ValueError(f"noise_std must be > 0, got {noise_std}")
        if solver not in {"euler", "midpoint", "heun"}:
            raise ValueError(f"Unsupported solver: {solver}")
        if e_c.dim() != 2 or e_c.size(-1) != self.content_dim:
            raise ValueError(
                f"Expected e_c shape [N,{self.content_dim}], got {tuple(e_c.shape)}"
            )
        if e_s.dim() != 2 or e_s.size(-1) != self.style_dim:
            raise ValueError(
                f"Expected e_s shape [N,{self.style_dim}], got {tuple(e_s.shape)}"
            )
        if e_c.size(0) != e_s.size(0):
            raise ValueError(f"e_c/e_s batch mismatch: {e_c.size(0)} vs {e_s.size(0)}")

        n = e_c.size(0)
        if z0 is None:
            z = torch.randn((n, self.latent_dim), device=e_c.device, dtype=e_c.dtype) * noise_std
        else:
            if z0.shape != (n, self.latent_dim):
                raise ValueError(
                    f"Expected z0 shape {(n, self.latent_dim)}, got {tuple(z0.shape)}"
                )
            z = z0

        dt = 1.0 / float(num_steps)
        for k in range(num_steps):
            t = z.new_full((n, 1), float(k) / float(num_steps))
            v = self.velocity(z_t=z, t=t, e_c=e_c, e_s=e_s)
            if solver == "euler":
                z = z + dt * v
                continue

            if solver == "midpoint":
                t_mid = z.new_full((n, 1), (float(k) + 0.5) / float(num_steps))
                z_mid = z + 0.5 * dt * v
                v_mid = self.velocity(z_t=z_mid, t=t_mid, e_c=e_c, e_s=e_s)
                z = z + dt * v_mid
                continue

            t_next = z.new_full((n, 1), float(k + 1) / float(num_steps))
            z_pred = z + dt * v
            v_next = self.velocity(z_t=z_pred, t=t_next, e_c=e_c, e_s=e_s)
            z = z + 0.5 * dt * (v + v_next)
        return z

    @torch.no_grad()
    def target_latent(self, x_next: torch.Tensor) -> torch.Tensor:
        x_next_bt = self._to_2d_frame_input(x_next, name="x_next", expected_dim=self.input_dim)
        self.vqvae.eval()
        if x_next_bt.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                z_q, _, _ = self.vqvae.encode(x_next_bt.float())
        else:
            z_q, _, _ = self.vqvae.encode(x_next_bt.float())
        if z_q.shape != (x_next_bt.size(0), self.latent_dim):
            raise RuntimeError(
                f"Unexpected target latent shape {tuple(z_q.shape)}; expected {(x_next_bt.size(0), self.latent_dim)}"
            )
        return z_q

    def compute_loss(
        self,
        *,
        x_curr: torch.Tensor,
        x_next: torch.Tensor,
        style_label: torch.Tensor,
        target_latent: torch.Tensor | None = None,
        flow_loss_type: Literal["l1", "mse", "smooth_l1"] = "mse",
        recon_type: Literal["l1", "mse", "smooth_l1"] = "mse",
        flow_weight: float = 1.0,
        recon_weight: float = 0.0,
        noise_std: float = 1.0,
        solver_steps: int = 8,
        solver: Literal["euler", "midpoint", "heun"] = "euler",
        recon_solver_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        x_curr_bt = self._to_2d_frame_input(x_curr, name="x_curr", expected_dim=self.input_dim)
        x_next_bt = self._to_2d_frame_input(x_next, name="x_next", expected_dim=self.input_dim)
        if x_curr_bt.size(0) != x_next_bt.size(0):
            raise ValueError(
                f"x_curr/x_next token count mismatch: {x_curr_bt.size(0)} vs {x_next_bt.size(0)}"
            )
        if flow_weight < 0.0:
            raise ValueError(f"flow_weight must be >= 0, got {flow_weight}")
        if recon_weight < 0.0:
            raise ValueError(f"recon_weight must be >= 0, got {recon_weight}")
        if noise_std <= 0.0:
            raise ValueError(f"noise_std must be > 0, got {noise_std}")
        if recon_solver_steps is not None:
            solver_steps = int(recon_solver_steps)
        if solver_steps < 1:
            raise ValueError(f"solver_steps must be >= 1, got {solver_steps}")
        if solver not in {"euler", "midpoint", "heun"}:
            raise ValueError(f"Unsupported solver: {solver}")

        style_bt = self._to_style_labels(style_label.to(device=x_curr_bt.device), n=x_curr_bt.size(0))
        traj_next, motion_next = self.split_traj_motion(x_next_bt)

        e_c = self.content_encoder(x_curr_bt)
        e_s = self.style_encoder(style_bt)

        if target_latent is None:
            with torch.no_grad():
                z1 = self.target_latent(x_next_bt)
        else:
            z1 = target_latent
            if z1.shape != (x_next_bt.size(0), self.latent_dim):
                raise ValueError(
                    f"Expected target_latent shape {(x_next_bt.size(0), self.latent_dim)}, got {tuple(z1.shape)}"
                )
            z1 = z1.to(device=x_next_bt.device, dtype=torch.float32)

        z0 = torch.randn_like(z1) * noise_std
        t = torch.rand((z1.size(0), 1), device=z1.device, dtype=z1.dtype)
        z_t = (1.0 - t) * z0 + t * z1
        v_target = z1 - z0
        v_pred = self.velocity(z_t=z_t, t=t, e_c=e_c, e_s=e_s)
        flow_loss = self._loss_by_type(v_pred, v_target, flow_loss_type)

        recon_loss = z1.new_zeros(())
        if recon_weight > 0.0:
            z_hat = self.integrate_euler(
                e_c=e_c,
                e_s=e_s,
                z0=z0,
                num_steps=solver_steps,
                noise_std=noise_std,
                solver=solver,
            )
            motion_hat = self.vqvae.decode(z_hat, traj_next, from_indices=False)
            recon_loss = self._loss_by_type(motion_hat, motion_next, recon_type)

        total = flow_weight * flow_loss + recon_weight * recon_loss
        return {
            "loss": total,
            "flow_loss": flow_loss,
            "recon_loss": recon_loss,
            "latent_target_norm": z1.norm(dim=-1).mean(),
            "velocity_pred_norm": v_pred.norm(dim=-1).mean(),
            "velocity_target_norm": v_target.norm(dim=-1).mean(),
        }

    @torch.no_grad()
    def sample_latent(
        self,
        *,
        x_curr: torch.Tensor,
        style_label: torch.Tensor,
        num_steps: int = 16,
        noise_std: float = 1.0,
        z0: torch.Tensor | None = None,
        solver: Literal["euler", "midpoint", "heun"] = "euler",
    ) -> torch.Tensor:
        self.eval()
        x_curr_bt = self._to_2d_frame_input(x_curr, name="x_curr", expected_dim=self.input_dim)
        style_bt = self._to_style_labels(style_label.to(device=x_curr_bt.device), n=x_curr_bt.size(0))
        e_c = self.content_encoder(x_curr_bt)
        e_s = self.style_encoder(style_bt)
        return self.integrate_euler(
            e_c=e_c,
            e_s=e_s,
            z0=z0,
            num_steps=num_steps,
            noise_std=noise_std,
            solver=solver,
        )

    @torch.no_grad()
    def predict_next_motion(
        self,
        *,
        x_curr: torch.Tensor,
        traj_next: torch.Tensor,
        style_label: torch.Tensor,
        num_steps: int = 16,
        noise_std: float = 1.0,
        z0: torch.Tensor | None = None,
        solver: Literal["euler", "midpoint", "heun"] = "euler",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        x_curr_bt = self._to_2d_frame_input(x_curr, name="x_curr", expected_dim=self.input_dim)
        traj_next_bt = self._to_2d_frame_input(traj_next, name="traj_next", expected_dim=self.traj_dim)
        if x_curr_bt.size(0) != traj_next_bt.size(0):
            raise ValueError(
                f"x_curr/traj_next token count mismatch: {x_curr_bt.size(0)} vs {traj_next_bt.size(0)}"
            )
        z_hat = self.sample_latent(
            x_curr=x_curr_bt,
            style_label=style_label,
            num_steps=num_steps,
            noise_std=noise_std,
            z0=z0,
            solver=solver,
        )
        motion_hat = self.vqvae.decode(z_hat, traj_next_bt, from_indices=False)
        return motion_hat, z_hat


__all__ = [
    "ContentEncoder",
    "StyleEncoder",
    "ConditionalVelocityField",
    "FlowMatchVQVAEController",
]
