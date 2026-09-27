"""Projection of candidate designs onto the physics constraint manifold ``c(x) = 0``.

All projectors work in **physical space** on ``(B, C, H, W)`` tensors and return ``(x_projected, info)``.

* ``gauss_newton``  — hard constraint: repeated minimum-norm (damped) Gauss-Newton steps with per-sample backtracking
  line search. Dense Jacobian for small problems, matrix-free CG (``torch.func`` jvp/vjp) for large/GPU problems.
* ``solve``         — exact partial projection through the physics' own solver (Darcy: re-solve ``p`` for fixed ``k``).
* ``gradient``      — baseline penalty descent  ``x <- x - eta * g/|g|``  with per-sample Armijo backtracking.
* ``residual_weighted`` — as ``gradient`` but minimises ``sum w_i c_i^2`` with residual-feedback weights
  ``w_i ~ (|c_i| + eps)^beta`` (re-computed periodically). Our own definition of a residual-feedback scheme; see
  docs/DECISIONS.md (the team's "PIRF" has no formal specification).

``sCM-PINN`` is not implemented: no formal definition exists in the project documents (consistency models are a
sampler, not a corrector).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from squad1.errors import ConvergenceError, PhysicsError
from squad1.physics.base import Physics

CorrectionFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def rms_correction(x_new: torch.Tensor, x_old: torch.Tensor) -> torch.Tensor:
    """Per-sample RMS-per-element correction ``(B,)`` (grid-size independent)."""
    return (x_new - x_old).flatten(1).pow(2).mean(dim=1).sqrt()


@dataclass
class ProjectorConfig:
    max_iters: int = 10
    tol: float = 1e-8  # converged when constraint RMS <= tol (scaled residual units, float64)
    max_correction_rms: float | None = None  # rejection threshold (measured by the correction function)
    fixed_channels: Sequence[int] = field(default_factory=tuple)


class Projector(ABC):
    name: str = ""

    def __init__(self, config: ProjectorConfig | None = None):
        self.config = config or ProjectorConfig()

    @abstractmethod
    def _run(self, x0: torch.Tensor, physics: Physics, h: float) -> tuple[torch.Tensor, dict[str, Any]]: ...

    def project(
        self,
        x0: torch.Tensor,
        physics: Physics,
        h: float,
        correction_fn: CorrectionFn = rms_correction,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        physics.check(x0)
        x0 = x0.detach()
        with torch.no_grad():
            c0 = physics.constraint_vector(x0, h)
        loss_before = c0.pow(2).mean(dim=1)
        x, info = self._run(x0, physics, h)
        with torch.no_grad():
            c1 = physics.constraint_vector(x, h)
        loss_after = c1.pow(2).mean(dim=1)
        corr = correction_fn(x, x0)
        reasons: list[str | None] = [None] * x0.shape[0]
        finite = torch.isfinite(x).flatten(1).all(dim=1) & torch.isfinite(loss_after)
        cap = self.config.max_correction_rms
        rejected = ~finite
        for b in range(x0.shape[0]):
            if not bool(finite[b]):
                reasons[b] = "non_finite_result"
            elif cap is not None and float(corr[b]) > cap:
                rejected[b] = True
                reasons[b] = "excessive_correction"
        if bool(rejected.any()):
            x = torch.where(rejected.view(-1, 1, 1, 1), x0, x)
            loss_after = torch.where(rejected, loss_before, loss_after)
            corr = torch.where(rejected, torch.zeros_like(corr), corr)
        converged = (loss_after.sqrt() <= self.config.tol) & ~rejected
        info.update(
            method=self.name,
            loss_before=loss_before,
            loss_after=loss_after,
            correction=corr,
            converged=converged,
            rejected=rejected,
            reject_reason=reasons,
        )
        return x, info


# ------------------------------------------------------------------------------------------- Gauss-Newton
def _damped_solve(A: torch.Tensor, b: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Solve ``(A + (lam + rel * mean(diag A)) I) y = b`` per sample; escalate the shift if a matrix is singular."""
    B, n = A.shape[0], A.shape[1]
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    scale = torch.diagonal(A, dim1=-2, dim2=-1).abs().mean(dim=-1).clamp_min(1e-300)
    rel = 1e-13
    for _ in range(8):
        shift = (lam + rel * scale).view(B, 1, 1)
        y, info = torch.linalg.solve_ex(A + shift * eye, b)
        if bool((info == 0).all()) and bool(torch.isfinite(y).all()):
            return y
        rel *= 100
    return torch.nan_to_num(y)


@dataclass
class GaussNewtonConfig(ProjectorConfig):
    max_iters: int = 25
    damping: float = 1e-12  # initial / minimum Levenberg-Marquardt damping (adapted per sample)
    max_damping: float = 1e8
    linear_solver: str = "auto"  # "dense" | "cg" | "auto"
    dense_max_elems: int = 6_000_000  # M*D per sample above which "auto" switches to CG
    cg_tol: float = 1e-8
    cg_max_iter: int = 3000
    line_search: Sequence[float] = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
    active_set_rounds: int = 3


class GaussNewtonProjector(Projector):
    name = "gauss_newton"

    def __init__(self, config: GaussNewtonConfig | None = None):
        super().__init__(config or GaussNewtonConfig())
        self.config: GaussNewtonConfig

    def _base_mask(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, D)`` mask of variables allowed to move (1) / held fixed (0)."""
        m = torch.ones_like(x)
        for c in self.config.fixed_channels:
            m[:, c] = 0.0
        return m.reshape(x.shape[0], -1)

    def _direction_dense(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        physics: Physics,
        h: float,
        mask: torch.Tensor,
        lam: torch.Tensor,
    ) -> torch.Tensor:
        B = x.shape[0]
        shape = x.shape[1:]

        def f1(xf: torch.Tensor) -> torch.Tensor:
            return physics.constraint_vector(xf.reshape(1, *shape), h)[0]

        J = torch.func.vmap(torch.func.jacrev(f1))(x.reshape(B, -1))  # (B, M, D)
        J = J * mask.unsqueeze(1)
        M, D = J.shape[1], J.shape[2]
        finite = torch.isfinite(J).flatten(1).all(dim=1)
        J = torch.nan_to_num(J)
        if M <= D:
            A = J @ J.transpose(1, 2)
            y = _damped_solve(A, c.unsqueeze(-1), lam)
            dx = -(J.transpose(1, 2) @ y).squeeze(-1)
        else:
            A = J.transpose(1, 2) @ J
            dx = -_damped_solve(A, J.transpose(1, 2) @ c.unsqueeze(-1), lam).squeeze(-1)
        return (dx * finite.unsqueeze(1)).reshape(x.shape)

    def _direction_cg(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        physics: Physics,
        h: float,
        mask: torch.Tensor,
        lam: torch.Tensor,
    ) -> torch.Tensor:
        """Matrix-free minimum-norm step: solve ``(J J^T + lam I) y = c`` by CG, ``dx = -J^T y``."""
        cfg = self.config
        vjp_fn = torch.func.vjp(lambda z: physics.constraint_vector(z, h), x)[1]
        m = mask.reshape(x.shape)

        def jt(v: torch.Tensor) -> torch.Tensor:
            return vjp_fn(v)[0] * m

        def jj(v: torch.Tensor) -> torch.Tensor:
            t = jt(v)
            return torch.func.jvp(lambda z: physics.constraint_vector(z, h), (x,), (t,))[1] + lam[:, None] * v

        y = torch.zeros_like(c)
        r = c - jj(y)
        d = r.clone()
        rs = (r * r).sum(dim=1)
        bn = c.norm(dim=1).clamp_min(1e-300)
        for _ in range(cfg.cg_max_iter):
            Ad = jj(d)
            alpha = rs / (d * Ad).sum(dim=1).clamp_min(1e-300)
            y = y + alpha[:, None] * d
            r = r - alpha[:, None] * Ad
            rs_new = (r * r).sum(dim=1)
            if bool((rs_new.sqrt() / bn < cfg.cg_tol).all()):
                break
            d = r + (rs_new / rs.clamp_min(1e-300))[:, None] * d
            rs = rs_new
        else:
            raise ConvergenceError(
                f"CG (Gauss-Newton direction) did not reach {cfg.cg_tol:g} in {cfg.cg_max_iter} iterations"
            )
        return -jt(y)

    def _run(self, x0: torch.Tensor, physics: Physics, h: float) -> tuple[torch.Tensor, dict[str, Any]]:
        cfg = self.config
        x = x0.clone()
        base_mask = self._base_mask(x)
        B = x.shape[0]
        M = physics.constraint_vector(x[:1], h).shape[1]
        D = x[0].numel()
        solver = cfg.linear_solver
        if solver == "auto":
            solver = "dense" if cfg.dense_max_elems >= M * D else "cg"
        if solver not in ("dense", "cg"):
            raise PhysicsError(f"linear_solver must be 'dense', 'cg' or 'auto', got {cfg.linear_solver!r}")
        iters = 0
        lam = torch.full((B,), cfg.damping, dtype=x.dtype, device=x.device)
        for it in range(cfg.max_iters):
            with torch.no_grad():
                c = physics.constraint_vector(x, h)
            loss = c.pow(2).mean(dim=1)
            active = loss.sqrt() > cfg.tol
            if not bool(active.any()):
                break
            direction = self._direction_dense if solver == "dense" else self._direction_cg
            lo, hi = physics.bounds(x)
            span = (hi - lo).reshape(B, -1)
            mask = base_mask
            for _ in range(
                cfg.active_set_rounds + 1
            ):  # projected GN: freeze variables the step would push through a bound
                dx = direction(x, c, physics, h, mask, lam)
                flat_dx, flat_x = dx.reshape(B, -1), x.reshape(B, -1)
                at_lo, at_hi = (
                    (flat_x - lo.reshape(B, -1)) <= 1e-9 * span,
                    (hi.reshape(B, -1) - flat_x) <= 1e-9 * span,
                )
                hit = ((at_lo & (flat_dx < 0)) | (at_hi & (flat_dx > 0))) & (mask > 0)
                if not bool(hit.any()):
                    break
                mask = mask * (~hit).to(mask.dtype)
            best_x, best_loss = x, loss
            for a in cfg.line_search:
                with torch.no_grad():
                    cand = physics.clamp_feasible(x + a * dx)
                    ln = physics.constraint_vector(cand, h).pow(2).mean(dim=1)
                better = (ln < best_loss) & torch.isfinite(ln)
                best_x = torch.where(better.view(B, 1, 1, 1), cand, best_x)
                best_loss = torch.where(better, ln, best_loss)
            improved = best_loss < loss
            # Levenberg-Marquardt: relax damping where the step helped, raise it where it did not
            lam = torch.where(improved, (lam * 0.1).clamp_min(cfg.damping), (lam * 100).clamp(1e-8, cfg.max_damping))
            x = best_x
            iters = it + 1
            if not bool(improved.any()) and bool((lam[active] >= cfg.max_damping).all()):
                break
        return x, {"iterations": iters, "linear_solver": solver}


# ------------------------------------------------------------------------------------------- exact solve
class SolveProjector(Projector):
    """Re-solve the dependent channels through the physics' own solver (Darcy: ``p`` from ``k``)."""

    name = "solve"

    def _run(self, x0: torch.Tensor, physics: Physics, h: float) -> tuple[torch.Tensor, dict[str, Any]]:
        out = physics.solve_dependent(x0, h)
        if out is None:
            raise PhysicsError(f"{type(physics).__name__} has no exact solver (solve_dependent); use 'gauss_newton'")
        return out.detach(), {"iterations": 1}


# ------------------------------------------------------------------------------------ first-order methods
@dataclass
class FirstOrderConfig(ProjectorConfig):
    max_iters: int = 300
    lr: float = 0.05  # initial normalised step length (in state units)
    reweight_every: int = 25
    beta: float = 1.0
    eps: float = 1e-8
    armijo: float = 1e-4
    max_backtracks: int = 12


class GradientProjector(Projector):
    """Baseline: normalised gradient descent with Armijo backtracking (monotone by construction)."""

    name = "gradient"

    def __init__(self, config: FirstOrderConfig | None = None):
        super().__init__(config or FirstOrderConfig())
        self.config: FirstOrderConfig

    def _weights(self, c: torch.Tensor) -> torch.Tensor | None:
        return None

    def _weighted_loss(self, physics: Physics, x: torch.Tensor, h: float, w: torch.Tensor | None) -> torch.Tensor:
        c = physics.constraint_vector(x, h)
        return (c.pow(2) * w).mean(dim=1) if w is not None else c.pow(2).mean(dim=1)

    def _run(self, x0: torch.Tensor, physics: Physics, h: float) -> tuple[torch.Tensor, dict[str, Any]]:
        cfg = self.config
        x = x0.clone()
        B = x.shape[0]
        keep = torch.ones_like(x[:1])
        for c in cfg.fixed_channels:
            keep[:, c] = 0.0
        step = torch.full((B,), cfg.lr, dtype=x.dtype, device=x.device)
        w = None
        iters = 0
        for it in range(cfg.max_iters):
            if it % max(cfg.reweight_every, 1) == 0:
                with torch.no_grad():
                    w = self._weights(physics.constraint_vector(x, h))
            xr = x.detach().clone().requires_grad_(True)
            loss = self._weighted_loss(physics, xr, h, w)
            (g,) = torch.autograd.grad(loss.sum(), xr)
            g = g * keep
            gn = g.flatten(1).norm(dim=1).clamp_min(1e-300)
            direction = -g / gn.view(-1, 1, 1, 1)
            with torch.no_grad():
                if float(loss.sqrt().max()) <= cfg.tol:
                    break
                accepted = torch.zeros(B, dtype=torch.bool, device=x.device)
                new_x = x.clone()
                new_loss = loss.detach().clone()
                t = step.clone()
                for _ in range(cfg.max_backtracks):
                    cand = physics.clamp_feasible(x + t.view(-1, 1, 1, 1) * direction)
                    lc = self._weighted_loss(physics, cand, h, w)
                    ok = (lc <= loss - cfg.armijo * t * gn) & ~accepted & torch.isfinite(lc)
                    new_x = torch.where(ok.view(-1, 1, 1, 1), cand, new_x)
                    new_loss = torch.where(ok, lc, new_loss)
                    accepted = accepted | ok
                    step = torch.where(ok, t * 1.5, step)  # grow after success
                    t = torch.where(accepted, t, t * 0.5)
                    if bool(accepted.all()):
                        break
                step = torch.where(accepted, step, step * 0.25)
                x = new_x
            iters = it + 1
            if not bool(accepted.any()) and float(step.max()) < 1e-14:
                break
        return x, {"iterations": iters}


class ResidualWeightedProjector(GradientProjector):
    """Residual-feedback weighting: cells with larger residual get larger weight (``w ~ (|c|+eps)^beta``, mean 1)."""

    name = "residual_weighted"

    def _weights(self, c: torch.Tensor) -> torch.Tensor | None:
        w = (c.abs() + self.config.eps).pow(self.config.beta)
        return w / w.mean(dim=1, keepdim=True)


PROJECTORS: dict[str, type[Projector]] = {
    "gauss_newton": GaussNewtonProjector,
    "solve": SolveProjector,
    "gradient": GradientProjector,
    "residual_weighted": ResidualWeightedProjector,
}


def get_projector(name: str, **config: Any) -> Projector:
    if name in ("scm_pinn", "scm-pinn", "pirf"):
        raise NotImplementedError(
            f"{name!r} has no formal definition in the project documents. Use one of {sorted(PROJECTORS)}; "
            "see docs/DECISIONS.md (residual_weighted is our explicit residual-feedback variant)."
        )
    if name not in PROJECTORS:
        raise PhysicsError(f"unknown projector {name!r}; available: {sorted(PROJECTORS)}")
    cls = PROJECTORS[name]
    if name == "gauss_newton":
        return cls(GaussNewtonConfig(**config))
    if name in ("gradient", "residual_weighted"):
        return cls(FirstOrderConfig(**config))
    return cls(ProjectorConfig(**config))
