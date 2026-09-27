"""PINN network: input non-dimensionalisation, optional Fourier features, tanh MLP with Xavier init."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import torch
import torch.nn as nn

from squad1.errors import ContractError


class MLP(nn.Module):
    """``x (N, d_in) -> u (N, d_out)``.

    * ``lo``/``hi`` (per input dimension) map inputs to ``[-1, 1]`` before the network (non-dimensionalisation);
    * ``fourier_features > 0`` prepends random Fourier features ``[sin(2 pi B x), cos(2 pi B x)]`` (Tancik et al. 2020;
      helps against spectral bias, see Wang, Wang & Perdikaris 2023) with a seeded, fixed ``B``;
    * ``tanh`` by default (smooth: PDE residuals need higher derivatives).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int = 64,
        layers: int = 4,
        activation: str = "tanh",
        lo: Sequence[float] | None = None,
        hi: Sequence[float] | None = None,
        fourier_features: int = 0,
        fourier_scale: float = 1.0,
        seed: int = 0,
    ):
        super().__init__()
        if in_dim < 1 or out_dim < 1 or hidden < 1 or layers < 1:
            raise ContractError("in_dim, out_dim, hidden and layers must be >= 1")
        acts = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU, "sin": None}
        if activation not in acts:
            raise ContractError(f"activation must be one of {sorted(acts)}, got {activation!r}")
        lo_t = torch.zeros(in_dim) if lo is None else torch.tensor(list(lo), dtype=torch.float32)
        hi_t = torch.ones(in_dim) if hi is None else torch.tensor(list(hi), dtype=torch.float32)
        if lo_t.shape != (in_dim,) or hi_t.shape != (in_dim,) or bool((hi_t <= lo_t).any()):
            raise ContractError("lo/hi must have one entry per input dimension with hi > lo")
        self.register_buffer("lo", lo_t)
        self.register_buffer("hi", hi_t)
        self.in_dim, self.out_dim = in_dim, out_dim
        if fourier_features > 0:
            g = torch.Generator().manual_seed(seed)
            self.register_buffer("B", torch.randn(in_dim, fourier_features, generator=g) * fourier_scale)
            width = 2 * fourier_features + in_dim
        else:
            self.B = None
            width = in_dim
        act = acts[activation]
        mods: list[nn.Module] = [nn.Linear(width, hidden)]
        for i in range(layers):
            mods.append(act() if act is not None else _Sin())
            mods.append(nn.Linear(hidden, hidden if i < layers - 1 else out_dim))
        self.net = nn.Sequential(*mods)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ContractError(f"expected (N, {self.in_dim}), got {tuple(x.shape)}")
        lo, hi = cast(torch.Tensor, self.lo), cast(torch.Tensor, self.hi)
        z = 2 * (x - lo) / (hi - lo) - 1
        if self.B is not None:
            p = 2 * math.pi * z @ self.B
            z = torch.cat([z, p.sin(), p.cos()], dim=1)
        return self.net(z)


class _Sin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)
