"""Diffusion training loop (epsilon prediction) with classifier-free conditioning dropout and EMA weights."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch

from squad1.errors import ContractError, NonFiniteError
from squad1.generation.diffusion import DiffusionScheduler
from squad1.generation.dit import DiT
from squad1.utils.seed import seed_everything


@dataclass
class TrainConfig:
    steps: int = 500
    batch_size: int = 32
    lr: float = 2e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    cond_dropout: float = 0.1
    ema_decay: float = 0.99
    seed: int = 0
    log_every: int = 50


def train_diffusion(
    model: DiT,
    scheduler: DiffusionScheduler,
    data: torch.Tensor,
    cfg: TrainConfig | None = None,
    cond_vec: torch.Tensor | None = None,
    cond_field: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> tuple[DiT, dict[str, list[float]]]:
    """Train ``model`` on model-space ``data (N, C, S, S)``; returns ``(ema_model, history)``."""
    cfg = cfg or TrainConfig()
    if data.ndim != 4 or data.shape[1] != model.in_channels or data.shape[2] != model.img_size:
        raise ContractError(
            f"data must be (N, {model.in_channels}, {model.img_size}, {model.img_size}); got {tuple(data.shape)}"
        )
    if not torch.isfinite(data).all():
        raise NonFiniteError("training data contains NaN/Inf")
    if cond_vec is not None and cond_vec.shape[0] != data.shape[0]:
        raise ContractError("cond_vec and data must have the same number of samples")
    if cond_field is not None and cond_field.shape[0] != data.shape[0]:
        raise ContractError("cond_field and data must have the same number of samples")
    seed_everything(cfg.seed)
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(dev).train()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            min(1.0, (s + 1) / 20) * (0.5 * (1 + math.cos(math.pi * min(s, cfg.steps) / cfg.steps)) * 0.95 + 0.05)
        ),
    )
    gen = torch.Generator().manual_seed(cfg.seed)
    N = data.shape[0]
    hist: dict[str, list[float]] = {"loss": [], "step": []}
    run = 0.0
    for step in range(cfg.steps):
        idx = torch.randint(0, N, (min(cfg.batch_size, N),), generator=gen)
        x0 = data[idx].to(dev)
        t = torch.randint(0, scheduler.timesteps, (x0.shape[0],), generator=gen).to(dev)
        noise = torch.randn(x0.shape, generator=gen).to(dev)
        xt = scheduler.q_sample(x0, t, noise)
        cv = None if cond_vec is None else cond_vec[idx].to(dev)
        cf = None if cond_field is None else cond_field[idx].to(dev)
        drop = None if cv is None else (torch.rand(x0.shape[0], generator=gen) < cfg.cond_dropout).to(dev)
        loss = (model(xt, t, cv, cf, drop) - noise).pow(2).mean()
        if not torch.isfinite(loss):
            raise NonFiniteError(f"loss became non-finite at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(cfg.ema_decay).add_(pm.detach(), alpha=1 - cfg.ema_decay)
        run += float(loss.detach())
        if (step + 1) % cfg.log_every == 0:
            hist["loss"].append(run / cfg.log_every)
            hist["step"].append(step + 1)
            run = 0.0
    return ema, hist
