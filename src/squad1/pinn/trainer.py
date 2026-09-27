"""PINN trainer: Adam -> L-BFGS, resampled collocation points, residual-adaptive refinement, loss balancing.

Validation is the **relative L2 error against the problem's exact solution** — never the training loss (a network can
drive the PDE residual to ~0 while being completely wrong when IC/BC are missing; see docs/DECISIONS.md).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from squad1.errors import ContractError, NonFiniteError
from squad1.pinn.models import MLP
from squad1.pinn.problems import Problem
from squad1.utils.seed import config_hash, seed_everything


@dataclass
class TrainerConfig:
    adam_steps: int = 2000
    lr: float = 2e-3
    lbfgs_steps: int = 0
    n_interior: int = 1000
    n_boundary: int = 200
    n_initial: int = 200
    resample: bool = True  # fresh uniform collocation points every step (else fixed set)
    rad_every: int = 0  # residual-adaptive refinement (Wu et al. 2023, RAD): 0 = off
    rad_pool: int = 10000
    rad_fraction: float = 0.5  # share of interior points drawn from the RAD distribution
    rad_k: float = 1.0
    rad_c: float = 1.0
    balance_every: int = 100  # gradient-norm loss balancing (Wang et al. 2021); 0 = fixed weights
    balance_alpha: float = 0.9
    bc_weight: float = 10.0
    ic_weight: float = 10.0
    grad_clip: float = 10.0
    seed: int = 0
    log_every: int = 100
    eval_points: int = 4000


@dataclass
class TrainResult:
    model: torch.nn.Module
    history: dict[str, list[float]]
    rel_l2: float | None
    weights: dict[str, float]
    seconds: float
    config_hash: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def relative_l2(model: torch.nn.Module, problem: Problem, n: int = 4000, seed: int = 123) -> float | None:
    gen = torch.Generator().manual_seed(seed)
    x = problem.sample_interior(n, gen)
    ex = problem.exact(x)
    if ex is None:
        return None
    with torch.no_grad():
        u = model(x)
    return float(torch.linalg.norm(u - ex) / torch.linalg.norm(ex).clamp_min(1e-12))


def rad_sample(
    problem: Problem,
    model: torch.nn.Module,
    n: int,
    pool: int,
    gen: torch.Generator,
    k: float = 1.0,
    c: float = 1.0,
) -> torch.Tensor:
    """Residual-based adaptive distribution: draw ``n`` pool points with probability ``~ |r|^k / mean + c``."""
    x = problem.sample_interior(pool, gen).requires_grad_(True)
    r = problem.residual(model, x).abs().pow(2).sum(1).sqrt().detach()
    p = r**k / r.pow(k).mean().clamp_min(1e-30) + c
    idx = torch.multinomial(p / p.sum(), n, replacement=False, generator=gen)
    return x.detach()[idx]


def _param_grad_norm(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> tuple[float, float]:
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat = torch.cat([g.flatten() for g in gs if g is not None])
    return float(flat.abs().max()), float(flat.abs().mean())


def train(problem: Problem, model: MLP | None = None, cfg: TrainerConfig | None = None, **model_kw: Any) -> TrainResult:
    cfg = cfg or TrainerConfig()
    if cfg.adam_steps < 0 or cfg.lbfgs_steps < 0 or cfg.n_interior < 1:
        raise ContractError("steps must be >= 0 and n_interior >= 1")
    seed_everything(cfg.seed)
    if model is None:
        model = MLP(problem.in_dim, problem.out_dim, lo=problem.lo, hi=problem.hi, seed=cfg.seed, **model_kw)
    gen = torch.Generator().manual_seed(cfg.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    w = {"bc": cfg.bc_weight, "ic": cfg.ic_weight}
    hist: dict[str, list[float]] = {k: [] for k in ("step", "loss", "pde", "bc", "ic")}
    t0 = time.perf_counter()

    fixed: dict[str, Any] = {
        "x": problem.sample_interior(cfg.n_interior, gen),
        "bc": problem.boundary_points(cfg.n_boundary, gen),
        "ic": problem.initial_points(cfg.n_initial, gen),
    }
    rad_pts: torch.Tensor | None = None

    def losses(step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        nonlocal rad_pts
        if cfg.resample:
            n_rad = int(cfg.n_interior * cfg.rad_fraction) if (cfg.rad_every and rad_pts is not None) else 0
            x = problem.sample_interior(cfg.n_interior - n_rad, gen)
            if n_rad:
                x = torch.cat([x, rad_pts[:n_rad]], 0)  # type: ignore[index]
            bc = problem.boundary_points(cfg.n_boundary, gen)
            ic = problem.initial_points(cfg.n_initial, gen)
        else:
            x, bc, ic = fixed["x"], fixed["bc"], fixed["ic"]
        x = x.clone().requires_grad_(True)
        l_pde = problem.residual(model, x).pow(2).mean()
        l_bc = (model(bc[0]) - bc[1]).pow(2).mean()
        l_ic = None if ic is None else (model(ic[0]) - ic[1]).pow(2).mean()
        return l_pde, l_bc, l_ic

    def total(l_pde: torch.Tensor, l_bc: torch.Tensor, l_ic: torch.Tensor | None) -> torch.Tensor:
        t = l_pde + w["bc"] * l_bc
        return t if l_ic is None else t + w["ic"] * l_ic

    opt = torch.optim.Adam(params, lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(cfg.adam_steps, 1), eta_min=cfg.lr * 0.02)
    for step in range(cfg.adam_steps):
        if cfg.rad_every and step % cfg.rad_every == 0:
            rad_pts = rad_sample(
                problem,
                model,
                int(cfg.n_interior * cfg.rad_fraction),
                cfg.rad_pool,
                gen,
                cfg.rad_k,
                cfg.rad_c,
            )
        l_pde, l_bc, l_ic = losses(step)
        if cfg.balance_every and step % cfg.balance_every == 0 and step > 0:
            gmax, _ = _param_grad_norm(l_pde, params)
            for key, lb in (("bc", l_bc), ("ic", l_ic)):
                if lb is None:
                    continue
                _, gmean = _param_grad_norm(lb, params)
                target = gmax / max(gmean, 1e-12)
                w[key] = float(min(1e4, max(1.0, cfg.balance_alpha * w[key] + (1 - cfg.balance_alpha) * target)))
        loss = total(l_pde, l_bc, l_ic)
        if not torch.isfinite(loss):
            raise NonFiniteError(f"loss became non-finite at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        sched.step()
        if step % cfg.log_every == 0 or step == cfg.adam_steps - 1:
            hist["step"].append(float(step))
            hist["loss"].append(float(loss.detach()))
            hist["pde"].append(float(l_pde.detach()))
            hist["bc"].append(float(l_bc.detach()))
            hist["ic"].append(float(l_ic.detach()) if l_ic is not None else 0.0)

    if cfg.lbfgs_steps:
        # full-batch stage on a fixed point set (L-BFGS needs a deterministic objective)
        x_fix = problem.sample_interior(cfg.n_interior, gen)
        bc_fix = problem.boundary_points(cfg.n_boundary, gen)
        ic_fix = problem.initial_points(cfg.n_initial, gen)
        lbfgs = torch.optim.LBFGS(params, lr=1.0, max_iter=20, history_size=50, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            x = x_fix.clone().requires_grad_(True)
            l_pde = problem.residual(model, x).pow(2).mean()
            l_bc = (model(bc_fix[0]) - bc_fix[1]).pow(2).mean()
            l_ic = None if ic_fix is None else (model(ic_fix[0]) - ic_fix[1]).pow(2).mean()
            loss = total(l_pde, l_bc, l_ic)
            loss.backward()
            return loss

        for _ in range(cfg.lbfgs_steps):
            loss = lbfgs.step(closure)
            if not torch.isfinite(loss):
                raise NonFiniteError("L-BFGS loss became non-finite")
        hist["loss"].append(float(torch.as_tensor(loss).detach()))
        hist["step"].append(float(cfg.adam_steps + cfg.lbfgs_steps))
    return TrainResult(
        model,
        hist,
        relative_l2(model, problem, cfg.eval_points),
        dict(w),
        time.perf_counter() - t0,
        config_hash(cfg),
    )
