"""DDIM sampling with optional classifier-free guidance and physics guidance."""

from __future__ import annotations

from collections.abc import Callable

import torch

from squad1.contracts.candidate import CandidateDesign
from squad1.contracts.channels import MODEL_HI, MODEL_LO, ChannelNormalizer, get_domain
from squad1.errors import ContractError
from squad1.generation.diffusion import DiffusionScheduler
from squad1.generation.dit import DiT
from squad1.physics.base import Physics


@torch.no_grad()
def sample(
    model: DiT,
    scheduler: DiffusionScheduler,
    n: int,
    *,
    cond_vec: torch.Tensor | None = None,
    cond_field: torch.Tensor | None = None,
    steps: int = 20,
    cfg_scale: float = 0.0,
    physics: Physics | None = None,
    normalizer: ChannelNormalizer | None = None,
    h: float | None = None,
    physics_scale: float = 0.0,
    seed: int | None = None,
    device: torch.device | str | None = None,
    on_step: Callable[[int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Return model-space samples ``(n, C, S, S)`` in ``[-1, 1]``.

    ``cfg_scale`` > 0 needs ``cond_vec`` and a model with ``cond_dim`` (classifier-free guidance).
    ``physics_scale`` > 0 needs ``physics``, ``normalizer`` and ``h``: each step is nudged down the gradient of the
    physics loss of the predicted clean sample (RMS step size = ``physics_scale``). With scale 0 nothing is touched
    (recommended default: guidance lowers the residual but never makes it zero — project afterwards).
    """
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    if cond_vec is not None and cond_vec.shape[0] != n:
        raise ContractError(f"cond_vec batch {cond_vec.shape[0]} != n {n}")
    if cfg_scale > 0 and (cond_vec is None or not model.cond_dim):
        raise ContractError("cfg_scale > 0 requires cond_vec and a model built with cond_dim > 0")
    if physics_scale > 0 and (physics is None or normalizer is None or h is None):
        raise ContractError("physics_scale > 0 requires physics, normalizer and h")
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    was_training = model.training
    model.eval()
    x = torch.randn(n, model.in_channels, model.img_size, model.img_size, generator=gen).to(dev)
    cv = None if cond_vec is None else cond_vec.to(dev)
    cf = None if cond_field is None else cond_field.to(dev)
    seq = scheduler.timestep_sequence(steps)
    for i, t in enumerate(seq):
        t_prev = seq[i + 1] if i + 1 < len(seq) else -1
        tt = torch.full((n,), t, dtype=torch.long, device=dev)
        eps = model(x, tt, cv, cf)
        if cfg_scale > 0:
            eps_u = model(x, tt, cv, cf, drop_cond=torch.ones(n, dtype=torch.bool, device=dev))
            eps = eps_u + cfg_scale * (eps - eps_u)
        x0 = scheduler.predict_x0(x, tt, eps).clamp(MODEL_LO, MODEL_HI)
        x_next = scheduler.ddim_step(x, t, t_prev, eps, x0=x0)
        if physics_scale > 0:
            with torch.enable_grad():
                xr = x.detach().clone().requires_grad_(True)
                e = model(xr, tt, cv, cf)
                x0g = scheduler.predict_x0(xr, tt, e).clamp(MODEL_LO, MODEL_HI)
                phys = normalizer.to_physical(x0g).double()  # type: ignore[union-attr]
                (g,) = torch.autograd.grad(physics.loss(phys, h).sum(), xr)  # type: ignore[union-attr, arg-type]
            g = torch.nan_to_num(g)
            rms = g.flatten(1).pow(2).mean(dim=1).sqrt().clamp_min(1e-12).view(-1, 1, 1, 1)
            x_next = x_next - physics_scale * g / rms
        x = x_next
        if on_step is not None:
            on_step(t, x)
    model.train(was_training)
    return x.clamp(MODEL_LO, MODEL_HI).float().cpu()


def generate_candidates(
    model: DiT,
    scheduler: DiffusionScheduler,
    domain: str,
    n: int,
    h: float,
    *,
    boundary: dict | None = None,
    physics_meta: dict | None = None,
    seed: int | None = None,
    **kwargs,
) -> CandidateDesign:
    """Sample and wrap into a contract :class:`CandidateDesign` (model space)."""
    spec = get_domain(domain)
    if spec.n_channels != model.in_channels:
        raise ContractError(f"domain {domain!r} has {spec.n_channels} channels but the model has {model.in_channels}")
    x = sample(model, scheduler, n, h=h, seed=seed, **kwargs)
    return CandidateDesign(
        domain,
        x,
        "model",
        h,
        boundary=dict(boundary or {}),
        physics=dict(physics_meta or {}),
        generation={
            "seed": seed,
            "steps": kwargs.get("steps", 20),
            "cfg_scale": kwargs.get("cfg_scale", 0.0),
            "physics_scale": kwargs.get("physics_scale", 0.0),
        },
    )
