"""Conditioning vector -> 768-D token embeddings with a non-degenerate cross-attention bridge.

Every scalar of ``cond_vec (B, Dv)`` becomes its own context token (``value * W_i + b_i + slot_i + group embedding``,
FT-Transformer style), so each output position can *select* the composition / inferred-parameter / target entries it
needs. (A single pooled context token makes the softmax identically 1: the "attention" would only add one vector to
every position.)

Outputs: token embeddings ``(B, L, embed_dim)``, greedy token ids ``(B, L)``, logits ``(B, L, V)``. ``L`` defaults to
``max_seq_len`` and can be shortened per call.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from squad1.contracts.cond import CondSpec
from squad1.errors import ContractError, NonFiniteError


class ConditioningTokenizer(nn.Module):
    """``(B, Dv)`` -> ``(B, Dv, E)`` context tokens."""

    def __init__(self, groups: Mapping[str, int], embed_dim: int = 768):
        super().__init__()
        self.sizes = [int(n) for n in groups.values()]
        if not self.sizes or min(self.sizes) < 0 or sum(self.sizes) == 0:
            raise ContractError(f"invalid groups {dict(groups)}")
        self.dv = sum(self.sizes)
        self.w = nn.Parameter(torch.randn(self.dv, embed_dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(self.dv, embed_dim))
        self.slot = nn.Parameter(torch.randn(self.dv, embed_dim) * 0.02)
        self.register_buffer(
            "gid", torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(self.sizes)])
        )
        self.group_emb = nn.Embedding(len(self.sizes), embed_dim)

    def forward(self, cond_vec: torch.Tensor) -> torch.Tensor:
        if cond_vec.ndim != 2 or cond_vec.shape[1] != self.dv:
            raise ContractError(f"cond_vec must be (B, {self.dv}), got {tuple(cond_vec.shape)}")
        if not torch.isfinite(cond_vec).all():
            raise NonFiniteError("cond_vec contains NaN/Inf")
        return cond_vec.unsqueeze(-1) * self.w + self.b + self.slot + self.group_emb(self.gid)


class ConditioningToTokens(nn.Module):
    def __init__(
        self,
        groups: Mapping[str, int],
        vocab_size: int,
        embed_dim: int = 768,
        max_seq_len: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        if embed_dim % num_heads:
            raise ContractError("embed_dim must be divisible by num_heads")
        self.max_seq_len = max_seq_len
        self.ctx = ConditioningTokenizer(groups, embed_dim)
        self.queries = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, ff_dim, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    @classmethod
    def from_spec(cls, spec: CondSpec, vocab_size: int, **kw) -> ConditioningToTokens:
        return cls(spec.groups, vocab_size, **kw)

    def forward(
        self, cond_vec: torch.Tensor, seq_len: int | None = None, return_attn: bool = False
    ) -> tuple[torch.Tensor, ...]:
        L = self.max_seq_len if seq_len is None else seq_len
        if not 1 <= L <= self.max_seq_len:
            raise ContractError(f"seq_len must be in [1, {self.max_seq_len}], got {L}")
        ctx = self.ctx(cond_vec)
        q = self.queries[:, :L].expand(cond_vec.size(0), -1, -1)
        a, w = self.attn(q, ctx, ctx, need_weights=True, average_attn_weights=True)
        emb = self.out_norm(self.transformer(self.norm(q + a)))
        logits = self.head(emb)
        out = (emb, logits.argmax(-1), logits)
        return (*out, w) if return_attn else out


def token_loss(logits: torch.Tensor, target_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Cross-entropy ignoring PAD positions. ``target_ids`` is ``(B, L)`` with the same ``L`` as ``logits``."""
    if logits.shape[:2] != target_ids.shape:
        raise ContractError(f"logits {tuple(logits.shape)} and targets {tuple(target_ids.shape)} disagree")
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1), ignore_index=pad_id
    )


def token_accuracy(logits: torch.Tensor, target_ids: torch.Tensor, pad_id: int) -> float:
    mask = target_ids != pad_id
    return float(((logits.argmax(-1) == target_ids) & mask).sum() / mask.sum().clamp_min(1))


def train_tokens(
    model: ConditioningToTokens,
    cond_vecs: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: int,
    steps: int = 300,
    lr: float = 2e-3,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Teacher-free supervised training on paired ``(cond_vec, token ids)``; returns loss/accuracy history."""
    from squad1.utils.seed import seed_everything

    if cond_vecs.shape[0] != target_ids.shape[0]:
        raise ContractError("cond_vecs and target_ids must have the same number of rows")
    seed_everything(seed)
    L = target_ids.shape[1]
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist: dict[str, list[float]] = {"loss": [], "acc": []}
    model.train()
    for _ in range(steps):
        _, _, logits = model(cond_vecs, seq_len=L)
        loss = token_loss(logits, target_ids, pad_id)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist["loss"].append(float(loss.detach()))
        hist["acc"].append(token_accuracy(logits.detach(), target_ids, pad_id))
    model.eval()
    return hist
