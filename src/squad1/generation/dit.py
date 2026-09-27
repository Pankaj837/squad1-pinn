"""Compact Diffusion Transformer (Peebles & Xie 2023) with adaLN-Zero blocks.

Reference implementation with the constructor signature used by the project's cooling-plate POC
(``DiT(img_size, patch_size, in_channels, hidden_size, depth, num_heads)``) plus optional conditioning:

* ``cond_dim``      — size of the conditioning vector ``cond_vec (B, Dv)`` (added to the timestep embedding);
* ``field_channels``— channels of the spatial conditioning field ``cond_field (B, Cf, H, W)`` (concatenated to input);
* a learned *null* conditioning vector enables classifier-free guidance (``drop_cond`` mask).

Predicts the noise ``eps`` with the same shape as the input design tensor.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from squad1.errors import ContractError


def _sincos_1d(dim: int, pos: torch.Tensor) -> torch.Tensor:
    omega = 1.0 / (10000 ** (torch.arange(dim // 2, dtype=torch.float64) / (dim / 2)))
    out = pos.double().reshape(-1)[:, None] * omega[None]
    return torch.cat([out.sin(), out.cos()], dim=1)


def sincos_pos_embed_2d(dim: int, grid: int) -> torch.Tensor:
    if dim % 4:
        raise ContractError("hidden_size must be divisible by 4 for the 2-D positional embedding")
    g = torch.arange(grid)
    yy, xx = torch.meshgrid(g, g, indexing="ij")
    return torch.cat([_sincos_1d(dim // 2, yy), _sincos_1d(dim // 2, xx)], dim=1).float()[None]


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([args.cos(), args.sin()], dim=-1)
    return F.pad(emb, (0, dim % 2))


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ContractError("hidden_size must be divisible by num_heads")
        self.h = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, N, D))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = _Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(approximate="tanh"), nn.Linear(hid, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).unsqueeze(1).chunk(6, dim=-1)
        x = x + g1 * self.attn(self.n1(x) * (1 + sc1) + s1)
        return x + g2 * self.mlp(self.n2(x) * (1 + sc2) + s2)


class DiT(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        hidden_size: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        cond_dim: int = 0,
        field_channels: int = 0,
    ):
        super().__init__()
        if img_size % patch_size:
            raise ContractError(f"img_size {img_size} must be divisible by patch_size {patch_size}")
        self.config = {
            "img_size": img_size,
            "patch_size": patch_size,
            "in_channels": in_channels,
            "hidden_size": hidden_size,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "cond_dim": cond_dim,
            "field_channels": field_channels,
        }
        self.img_size, self.patch, self.in_channels = img_size, patch_size, in_channels
        self.cond_dim, self.field_channels = cond_dim, field_channels
        self.grid = img_size // patch_size
        self.embed = nn.Conv2d(in_channels + field_channels, hidden_size, patch_size, patch_size)
        self.register_buffer("pos", sincos_pos_embed_2d(hidden_size, self.grid), persistent=False)
        self.t_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.hidden = hidden_size
        if cond_dim:
            self.c_mlp = nn.Sequential(nn.Linear(cond_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
            self.null_cond = nn.Parameter(torch.zeros(cond_dim))
        self.blocks = nn.ModuleList(DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth))
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.out = nn.Linear(hidden_size, patch_size * patch_size * in_channels)
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for blk in self.blocks:
            last = cast(nn.Linear, cast(nn.Sequential, cast(DiTBlock, blk).ada)[-1])
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        final = cast(nn.Linear, self.final_ada[-1])
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        w = self.embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_vec: torch.Tensor | None = None,
        cond_field: torch.Tensor | None = None,
        drop_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.in_channels != C or self.img_size != H or self.img_size != W:
            raise ContractError(
                f"expected (B, {self.in_channels}, {self.img_size}, {self.img_size}), got {tuple(x.shape)}"
            )
        if t.shape != (B,):
            raise ContractError(f"t must have shape ({B},), got {tuple(t.shape)}")
        if self.field_channels:
            if cond_field is None or cond_field.shape != (B, self.field_channels, H, W):
                raise ContractError(f"cond_field must be (B, {self.field_channels}, {H}, {W})")
            x_in = torch.cat([x, cond_field.to(x.dtype)], dim=1)
        else:
            if cond_field is not None:
                raise ContractError("model was built with field_channels=0 but cond_field was given")
            x_in = x
        c = self.t_mlp(timestep_embedding(t, self.hidden))
        if self.cond_dim:
            if cond_vec is None:
                cv = self.null_cond.expand(B, -1)
            else:
                if cond_vec.shape != (B, self.cond_dim):
                    raise ContractError(f"cond_vec must be (B, {self.cond_dim}), got {tuple(cond_vec.shape)}")
                cv = cond_vec
                if drop_cond is not None:
                    cv = torch.where(drop_cond.view(-1, 1), self.null_cond.expand(B, -1), cv)
            c = c + self.c_mlp(cv)
        elif cond_vec is not None:
            raise ContractError("model was built with cond_dim=0 but cond_vec was given")
        tok = self.embed(x_in).flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            tok = blk(tok, c)
        shift, scale = self.final_ada(c).unsqueeze(1).chunk(2, dim=-1)
        tok = self.out(self.final_norm(tok) * (1 + scale) + shift)  # (B, N, p*p*C)
        g, p = self.grid, self.patch
        tok = tok.view(B, g, g, p, p, C).permute(0, 5, 1, 3, 2, 4)
        return tok.reshape(B, C, g * p, g * p)
