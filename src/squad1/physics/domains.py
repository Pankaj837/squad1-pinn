"""Domain library: steady PDE families with *valid* data generators.

Every domain provides
  * a registered :class:`~squad1.contracts.DomainSpec` (channel names + physical ranges),
  * a :class:`~squad1.physics.base.Physics` whose constraint is the h^2-scaled central-difference PDE residual
    on interior points,
  * ``sample(n, size, seed)``: physical-space fields that lie on the constraint manifold — either analytic
    (residual = O(h^2) truncation error) or obtained by a Newton solve of the *discrete* equations (residual ~ 1e-12).

Domains: ``laplace_heat``, ``stress_equilibrium``, ``reaction_diffusion`` (steady Fisher-KPP),
``diffusion_decay``, ``thermal_advection``, ``navier_stokes`` (Kovasznay flow).
Biology (Turing) and Maxwell need a team-defined steady formulation and are intentionally not included.
"""

from __future__ import annotations

import math
from abc import abstractmethod

import numpy as np
import torch

from squad1.contracts.channels import ChannelSpec, DomainSpec, register_domain
from squad1.errors import ConvergenceError, PhysicsError
from squad1.physics.base import Physics

# ----------------------------------------------------------------------------------------------- stencils


def _c(f: torch.Tensor) -> torch.Tensor:
    return f[:, 1:-1, 1:-1]


def _sh(f: torch.Tensor, di: int, dj: int) -> torch.Tensor:
    H, W = f.shape[1:]
    return f[:, 1 + di : H - 1 + di, 1 + dj : W - 1 + dj]


def lap_h2(f: torch.Tensor) -> torch.Tensor:
    """``h^2 * laplacian`` (5-point stencil sum) on interior points, ``(B, H-2, W-2)``."""
    return _sh(f, 1, 0) + _sh(f, -1, 0) + _sh(f, 0, 1) + _sh(f, 0, -1) - 4.0 * _c(f)


def dx_h(f: torch.Tensor) -> torch.Tensor:
    """``h * d/dx`` (central) on interior points."""
    return (_sh(f, 1, 0) - _sh(f, -1, 0)) / 2.0


def dy_h(f: torch.Tensor) -> torch.Tensor:
    return (_sh(f, 0, 1) - _sh(f, 0, -1)) / 2.0


# ----------------------------------------------------------------------------------------------- base


class InteriorPDE(Physics):
    """Physics defined by interior residual components (each ``(B, H-2, W-2)``, h^2-scaled)."""

    @abstractmethod
    def residual_components(self, x: torch.Tensor, h: float) -> list[torch.Tensor]: ...

    @abstractmethod
    def sample(self, n: int, size: int, seed: int = 0) -> torch.Tensor:
        """Physical-space samples ``(n, C, size, size)`` float32 on ``[0,1]^2`` (h = 1/(size-1))."""

    def constraint_vector(self, x: torch.Tensor, h: float) -> torch.Tensor:
        return torch.cat([r.flatten(1) for r in self.residual_components(x, h)], dim=1)


def _grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.linspace(0.0, 1.0, size)
    return np.meshgrid(g, g, indexing="ij")


def _scale_to(x: np.ndarray, amp: float = 0.9) -> np.ndarray:
    m = np.abs(x).max()
    return x if m == 0 else x * (amp / m)


# ----------------------------------------------------------------------------------------- Laplace / heat
register_domain(
    DomainSpec(
        "laplace_heat",
        (ChannelSpec("T", -1.0, 1.0, unit="K/K0"),),
        "Steady heat conduction: laplacian(T) = 0",
    )
)


class LaplaceHeat(InteriorPDE):
    domain_name = "laplace_heat"

    def residual_components(self, x: torch.Tensor, h: float) -> list[torch.Tensor]:
        return [lap_h2(x[:, 0])]

    def sample(self, n: int, size: int, seed: int = 0) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        X, Y = _grid(size)
        out = []
        for _ in range(n):
            f = rng.normal() * 0.2 + rng.normal() * 0.3 * X + rng.normal() * 0.3 * Y
            for m in range(1, 4):  # exact harmonic building blocks
                a, b, c, d = rng.normal(size=4) / m
                f += a * np.sin(m * np.pi * X) * np.sinh(m * np.pi * Y) / np.sinh(m * np.pi)
                f += b * np.sin(m * np.pi * Y) * np.sinh(m * np.pi * X) / np.sinh(m * np.pi)
                f += c * np.cos(m * np.pi * X) * np.cosh(m * np.pi * Y) / np.cosh(m * np.pi)
                f += d * np.cos(m * np.pi * Y) * np.cosh(m * np.pi * X) / np.cosh(m * np.pi)
            out.append(_scale_to(f)[None])
        return torch.tensor(np.stack(out), dtype=torch.float32)


# ------------------------------------------------------------------------------------ stress equilibrium
register_domain(
    DomainSpec(
        "stress_equilibrium",
        (ChannelSpec("sxx", -1.0, 1.0), ChannelSpec("sxy", -1.0, 1.0), ChannelSpec("syy", -1.0, 1.0)),
        "2-D stress tensor in equilibrium without body force: div(sigma) = 0",
    )
)


class StressEquilibrium(InteriorPDE):
    domain_name = "stress_equilibrium"

    def residual_components(self, x: torch.Tensor, h: float) -> list[torch.Tensor]:
        sxx, sxy, syy = x[:, 0], x[:, 1], x[:, 2]
        return [h * (dx_h(sxx) + dy_h(sxy)), h * (dx_h(sxy) + dy_h(syy))]

    def sample(self, n: int, size: int, seed: int = 0) -> torch.Tensor:
        """Airy stress function ``phi``: sxx = phi_yy, syy = phi_xx, sxy = -phi_xy (identically div-free)."""
        rng = np.random.default_rng(seed)
        X, Y = _grid(size)
        out = []
        for _ in range(n):
            sxx = np.zeros_like(X)
            syy = np.zeros_like(X)
            sxy = np.zeros_like(X)
            for _ in range(3):
                m, k = rng.integers(1, 4, size=2)
                a = rng.normal() / (m * m + k * k)
                ph = rng.uniform(0, 2 * np.pi)
                mx, ky = m * np.pi, k * np.pi
                # phi = a sin(mx X + ph) sin(ky Y)
                sxx += a * (-(ky**2)) * np.sin(mx * X + ph) * np.sin(ky * Y)
                syy += a * (-(mx**2)) * np.sin(mx * X + ph) * np.sin(ky * Y)
                sxy += -a * mx * ky * np.cos(mx * X + ph) * np.cos(ky * Y)
            s = np.stack([sxx, sxy, syy])
            out.append(s * (0.9 / np.abs(s).max()))
        return torch.tensor(np.stack(out), dtype=torch.float32)


# --------------------------------------------------------------------------------- Newton (Dirichlet ring)


def _random_ring(size: int, rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    """Smooth random boundary values on the ring, returned as a full (size,size) array (interior zero)."""
    s = np.linspace(0, 1, 4 * (size - 1), endpoint=False)
    v = np.zeros_like(s)
    for m in range(1, 4):
        v += rng.normal() * np.cos(2 * np.pi * m * s) / m + rng.normal() * np.sin(2 * np.pi * m * s) / m
    v = (v - v.min()) / max(v.max() - v.min(), 1e-9)
    v = lo + (hi - lo) * (0.15 + 0.7 * v)
    ring = np.zeros((size, size))
    n = size - 1
    ring[0, :n] = v[0:n]  # top edge, j = 0..n-1
    ring[:n, n] = v[n : 2 * n]  # right edge, i = 0..n-1
    ring[n, n:0:-1] = v[2 * n : 3 * n]  # bottom edge, j = n..1
    ring[n:0:-1, 0] = v[3 * n : 4 * n]  # left edge, i = n..1
    return ring


def newton_dirichlet(
    physics: InteriorPDE, ring: np.ndarray, h: float, iters: int = 40, tol: float = 1e-13
) -> torch.Tensor:
    """Solve the discrete interior equations of a 1-channel PDE with the boundary ring fixed (``(1, S, S)`` float64)."""
    size = ring.shape[0]
    base = torch.tensor(ring, dtype=torch.float64)
    mask = torch.zeros(size, size, dtype=torch.float64)
    mask[1:-1, 1:-1] = 1.0
    n_int = (size - 2) ** 2

    def full(u: torch.Tensor) -> torch.Tensor:
        interior = torch.zeros(size, size, dtype=torch.float64)
        interior[1:-1, 1:-1] = u.reshape(size - 2, size - 2)
        return (base * (1 - mask) + interior)[None, None]  # (1,1,S,S)

    def res(u: torch.Tensor) -> torch.Tensor:
        return torch.cat([r.flatten() for r in physics.residual_components(full(u), h)])

    u = torch.full((n_int,), float(ring[ring != 0].mean()) if (ring != 0).any() else 0.0, dtype=torch.float64)
    r = res(u)
    for _ in range(iters):
        if r.abs().max() < tol:
            return full(u)[0]
        J = torch.func.jacrev(res)(u)
        step = torch.linalg.solve(J, -r)
        t = 1.0
        while t > 1e-6:
            rn = res(u + t * step)
            if torch.linalg.norm(rn) < torch.linalg.norm(r):
                break
            t *= 0.5
        u = u + t * step
        r = rn
    if r.abs().max() >= 1e-9:
        raise ConvergenceError(f"Newton did not converge (max |r| = {r.abs().max():.2e})")
    return full(u)[0]


class _NewtonDomain(InteriorPDE):
    lo: float = 0.0
    hi: float = 1.0

    def sample(self, n: int, size: int, seed: int = 0) -> torch.Tensor:
        if size > 40:
            raise PhysicsError(
                "Newton data generation is a dense reference (size <= 40); use analytic domains for larger grids"
            )
        rng = np.random.default_rng(seed)
        h = 1.0 / (size - 1)
        out = [newton_dirichlet(self, _random_ring(size, rng, self.lo, self.hi), h) for _ in range(n)]
        return torch.stack(out).to(torch.float32)


register_domain(
    DomainSpec("reaction_diffusion", (ChannelSpec("u", 0.0, 1.0),), "Steady Fisher-KPP: D lap(u) + r u (1-u) = 0")
)


class ReactionDiffusion(_NewtonDomain):
    domain_name = "reaction_diffusion"

    def __init__(self, D: float = 0.1, r: float = 1.0):
        if D <= 0 or r < 0:
            raise PhysicsError("need D > 0 and r >= 0")
        self.D, self.r = float(D), float(r)

    def residual_components(self, x: torch.Tensor, h: float) -> list[torch.Tensor]:
        u = x[:, 0]
        return [self.D * lap_h2(u) + h * h * self.r * _c(u) * (1 - _c(u))]


register_domain(
    DomainSpec(
        "diffusion_decay",
        (ChannelSpec("c", 0.0, 1.0),),
        "Diffusion with first-order decay: D lap(c) - k c = 0",
    )
)


class DiffusionDecay(_NewtonDomain):
    domain_name = "diffusion_decay"

    def __init__(self, D: float = 0.25, k: float = 0.5):
        if D <= 0 or k < 0:
            raise PhysicsError("need D > 0 and k >= 0")
        self.D, self.k = float(D), float(k)

    def residual_components(self, x: torch.Tensor, h: float) -> list[torch.Tensor]:
        c = x[:, 0]
        return [self.D * lap_h2(c) - h * h * self.k * _c(c)]


register_domain(
    DomainSpec(
        "thermal_advection",
        (ChannelSpec("T", 0.0, 1.0),),
        "Steady advection-diffusion: v.grad(T) - D lap(T) = 0",
    )
)


class ThermalAdvection(_NewtonDomain):
    domain_name = "thermal_advection"

    def __init__(self, D: float = 0.1, vx: float = 1.0, vy: float = 0.3):
        if D <= 0:
            raise PhysicsError("need D > 0")
        self.D, self.vx, self.vy = float(D), float(vx), float(vy)

    def residual_components(self, x: torch.Tensor, h: float) -> list[torch.Tensor]:
        T = x[:, 0]
        return [h * (self.vx * dx_h(T) + self.vy * dy_h(T)) - self.D * lap_h2(T)]


# ---------------------------------------------------------------------------------------- Navier-Stokes
register_domain(
    DomainSpec(
        "navier_stokes",
        (ChannelSpec("u", -2.0, 4.0), ChannelSpec("v", -1.0, 1.0), ChannelSpec("p", -4.0, 1.0)),
        "Steady incompressible Navier-Stokes (u, v, p), verified on Kovasznay flow",
    )
)


class NavierStokes(InteriorPDE):
    domain_name = "navier_stokes"

    def __init__(self, Re: float = 20.0):
        if Re <= 0:
            raise PhysicsError("Re must be positive")
        self.Re = float(Re)

    def residual_components(self, x: torch.Tensor, h: float) -> list[torch.Tensor]:
        u, v, p = x[:, 0], x[:, 1], x[:, 2]
        uc, vc = _c(u), _c(v)
        mom_x = h * (uc * dx_h(u) + vc * dy_h(u) + dx_h(p)) - lap_h2(u) / self.Re
        mom_y = h * (uc * dx_h(v) + vc * dy_h(v) + dy_h(p)) - lap_h2(v) / self.Re
        cont = dx_h(u) + dy_h(v)
        return [mom_x, mom_y, cont * h]

    def sample(self, n: int, size: int, seed: int = 0) -> torch.Tensor:
        """Kovasznay flow, window ``x in [x0, x0+1]`` (random x0), random y-phase and pressure offset."""
        rng = np.random.default_rng(seed)
        lam = self.Re / 2 - math.sqrt(self.Re**2 / 4 + 4 * math.pi**2)
        g = np.linspace(0.0, 1.0, size)
        out = []
        for _ in range(n):
            x0, y0, p0 = rng.uniform(-0.6, -0.4), rng.uniform(0, 1), rng.uniform(-0.5, 0.5)
            X, Y = np.meshgrid(x0 + g, y0 + g, indexing="ij")
            e = np.exp(lam * X)
            u = 1 - e * np.cos(2 * np.pi * Y)
            v = lam / (2 * np.pi) * e * np.sin(2 * np.pi * Y)
            p = 0.5 * (1 - np.exp(2 * lam * X)) + p0
            out.append(np.stack([u, v, p]))
        return torch.tensor(np.stack(out), dtype=torch.float32)


# --------------------------------------------------------------------------------------------- factory
PHYSICS_CLASSES: dict[str, type[Physics]] = {
    "laplace_heat": LaplaceHeat,
    "stress_equilibrium": StressEquilibrium,
    "reaction_diffusion": ReactionDiffusion,
    "diffusion_decay": DiffusionDecay,
    "thermal_advection": ThermalAdvection,
    "navier_stokes": NavierStokes,
}
