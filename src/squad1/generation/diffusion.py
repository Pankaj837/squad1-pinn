"""Discrete-time diffusion utilities (epsilon prediction, cosine schedule, deterministic DDIM)."""

from __future__ import annotations

import math

import torch

from squad1.errors import ContractError


class DiffusionScheduler:
    """Cosine noise schedule (Nichol & Dhariwal 2021) with DDIM sampling helpers.

    ``t`` indexes ``0 .. timesteps-1``; ``alpha_bar[t]`` is the fraction of signal variance kept at step ``t``.
    """

    def __init__(self, timesteps: int = 1000, s: float = 0.008, max_beta: float = 0.999):
        if timesteps < 2:
            raise ContractError("timesteps must be >= 2")
        self.timesteps = int(timesteps)
        u = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        f = torch.cos((u + s) / (1 + s) * math.pi / 2) ** 2
        ab = f / f[0]
        betas = (1 - ab[1:] / ab[:-1]).clamp(max=max_beta)
        self.betas = betas.float()
        self.alpha_bar = torch.cumprod(1 - betas, dim=0).float()

    def _gather(self, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return self.alpha_bar.to(like.device)[t].view(-1, *([1] * (like.ndim - 1))).to(like.dtype)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        noise = torch.randn_like(x0) if noise is None else noise
        ab = self._gather(t, x0)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        ab = self._gather(t, xt)
        return (xt - (1 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-4)

    def ddim_step(
        self, xt: torch.Tensor, t: int, t_prev: int, eps: torch.Tensor, x0: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Deterministic DDIM update ``x_t -> x_{t_prev}`` (``t_prev = -1`` means the clean sample)."""
        ab_t = self.alpha_bar[t].to(xt)
        ab_p = self.alpha_bar[t_prev].to(xt) if t_prev >= 0 else torch.ones((), dtype=xt.dtype, device=xt.device)
        if x0 is None:
            x0 = (xt - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp_min(1e-4)
        eps_c = (xt - ab_t.sqrt() * x0) / (1 - ab_t).sqrt().clamp_min(1e-8)  # eps consistent with a clipped x0
        return ab_p.sqrt() * x0 + (1 - ab_p).sqrt() * eps_c

    def timestep_sequence(self, steps: int) -> list[int]:
        if not 1 <= steps <= self.timesteps:
            raise ContractError(f"steps must be in [1, {self.timesteps}], got {steps}")
        seq = torch.linspace(self.timesteps - 1, 0, steps).round().long().tolist()
        out: list[int] = []
        for v in seq:
            if not out or v != out[-1]:
                out.append(int(v))
        return out
