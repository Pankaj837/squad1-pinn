"""Physics interface shared by every domain (Darcy, Biot, the domain library, the cooling plate).

All methods take **physical-space** tensors ``(B, C, H, W)`` and the grid spacing ``h``.
Losses are per-sample ``(B,)`` so batches never mix.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from squad1.contracts.channels import DomainSpec, get_domain
from squad1.errors import ContractError, NonFiniteError, PhysicsError


class Physics(ABC):
    """A set of equality constraints ``c(x) = 0`` over physical-space design tensors."""

    domain_name: str = ""

    @property
    def spec(self) -> DomainSpec:
        return get_domain(self.domain_name)

    # -- required ---------------------------------------------------------------------------------
    @abstractmethod
    def constraint_vector(self, x: torch.Tensor, h: float) -> torch.Tensor:
        """Scaled constraint values ``(B, M)``; all-zero iff ``x`` satisfies the physics (incl. fixed BC rows)."""

    # -- provided -----------------------------------------------------------------------------------
    def check(self, x: torch.Tensor) -> None:
        """Raise on invalid input (layout, non-finite, domain-specific validity)."""
        if x.ndim != 4 or x.shape[1] != self.spec.n_channels:
            raise ContractError(
                f"{self.domain_name}: expected (B, {self.spec.n_channels}, H, W) channel-first tensor, "
                f"got {tuple(x.shape)}"
            )
        if not torch.is_floating_point(x):
            raise ContractError(f"{self.domain_name}: expected floating tensor, got {x.dtype}")
        if not torch.isfinite(x).all():
            raise NonFiniteError(f"{self.domain_name}: tensor contains NaN/Inf")
        self._check_values(x)

    def _check_values(self, x: torch.Tensor) -> None:  # pragma: no cover - default: nothing extra
        return None

    def loss(self, x: torch.Tensor, h: float) -> torch.Tensor:
        """Per-sample mean-squared constraint violation ``(B,)``."""
        return self.constraint_vector(x, h).pow(2).mean(dim=1)

    def residual_rms(self, x: torch.Tensor, h: float) -> torch.Tensor:
        return self.loss(x, h).sqrt()

    def gradient(self, x: torch.Tensor, h: float) -> torch.Tensor:
        """d(sum of per-sample losses)/dx via autograd — always the gradient of :meth:`loss` itself."""
        xr = x.detach().clone().requires_grad_(True)
        (g,) = torch.autograd.grad(self.loss(xr, h).sum(), xr)
        return g

    def bounds(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-channel registered physical range broadcast to ``x`` (the admissible box)."""
        shape = [1, -1] + [1] * (x.ndim - 2)
        lo = torch.tensor([c.lo for c in self.spec.channels], dtype=x.dtype, device=x.device).view(shape)
        hi = torch.tensor([c.hi for c in self.spec.channels], dtype=x.dtype, device=x.device).view(shape)
        return lo.expand_as(x), hi.expand_as(x)

    def clamp_feasible(self, x: torch.Tensor) -> torch.Tensor:
        """Map a state back into the admissible box: every channel inside its registered physical range.

        Keeping projected designs inside the registered ranges is part of the generator contract (model space is
        ``[-1, 1]``); it also guarantees e.g. strictly positive permeability.
        """
        lo, hi = self.bounds(x)
        return torch.maximum(torch.minimum(x, hi), lo)

    def solve_dependent(self, x: torch.Tensor, h: float) -> torch.Tensor | None:
        """Optional exact partial projection (e.g. Darcy: re-solve ``p`` from ``k``). ``None`` if unsupported."""
        return None


def require_positive(x: torch.Tensor, channel: int, name: str) -> None:
    if (x[:, channel] <= 0).any():
        raise PhysicsError(f"{name} must be strictly positive")
