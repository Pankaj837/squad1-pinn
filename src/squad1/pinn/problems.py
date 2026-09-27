"""PDE problems for the PINN trainer: residuals, IC/BC point sets and exact solutions for validation.

A problem owns the *definition of correctness*: initial/boundary conditions are explicit point sets with targets
for **every** output (the earlier trainer added IC/BC loss only when the input had parametric columns and constrained
only output 0), and every built-in problem carries an exact solution so validation is a relative L2 error, not the
training loss.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch

from squad1.errors import PhysicsError


def grad(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    (g,) = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)
    return g


class Problem(ABC):
    name = ""
    in_dim = 2
    out_dim = 1
    lo: tuple[float, ...] = (0.0, 0.0)
    hi: tuple[float, ...] = (1.0, 1.0)

    @abstractmethod
    def residual(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """PDE residual ``(N, k)`` at points ``x (N, in_dim)`` (``x`` requires grad)."""

    @abstractmethod
    def boundary_points(self, n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """``(x, target)`` for boundary conditions; ``target`` is ``(N, out_dim)``."""

    def initial_points(self, n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor] | None:
        return None

    def exact(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    def sample_interior(self, n: int, gen: torch.Generator) -> torch.Tensor:
        lo, hi = torch.tensor(self.lo), torch.tensor(self.hi)
        return lo + (hi - lo) * torch.rand(n, self.in_dim, generator=gen)


class Heat1D(Problem):
    """``u_t = alpha u_xx`` on ``(t, x) in [0,1] x [-1,1]``; ``u(0,x) = sin(pi x)``, ``u(t,+-1) = 0``."""

    name = "heat1d"
    lo, hi = (0.0, -1.0), (1.0, 1.0)

    def __init__(self, alpha: float = 0.05):
        if alpha <= 0:
            raise PhysicsError("alpha must be positive")
        self.alpha = float(alpha)

    def residual(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        u = model(x)
        g = grad(u, x)
        u_xx = grad(g[:, 1:2], x)[:, 1:2]
        return g[:, 0:1] - self.alpha * u_xx

    def boundary_points(self, n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.rand(n, 1, generator=gen)
        side = torch.randint(0, 2, (n, 1), generator=gen).float() * 2 - 1
        return torch.cat([t, side], 1), torch.zeros(n, 1)

    def initial_points(self, n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(n, 1, generator=gen) * 2 - 1
        return torch.cat([torch.zeros_like(x), x], 1), torch.sin(math.pi * x)

    def exact(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.alpha * math.pi**2 * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2])


class Poisson2D(Problem):
    """``-lap(u) = 2 pi^2 sin(pi x) sin(pi y)`` on the unit square, ``u = 0`` on the boundary."""

    name = "poisson2d"

    def residual(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        u = model(x)
        g = grad(u, x)
        lap = grad(g[:, 0:1], x)[:, 0:1] + grad(g[:, 1:2], x)[:, 1:2]
        f = 2 * math.pi**2 * torch.sin(math.pi * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2])
        return -lap - f

    def boundary_points(self, n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        s = torch.rand(n, 1, generator=gen)
        e = torch.randint(0, 4, (n,), generator=gen)
        x = torch.where(
            e[:, None] < 2,
            torch.cat([s, (e[:, None] % 2).float()], 1),
            torch.cat([(e[:, None] % 2).float(), s], 1),
        )
        return x, torch.zeros(n, 1)

    def exact(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(math.pi * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2])


PROBLEMS: dict[str, type[Problem]] = {"heat1d": Heat1D, "poisson2d": Poisson2D}
