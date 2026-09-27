"""Steady quasi-static Biot poroelasticity (channels ``[k, p, ux, uy]``).

Equations (2-D, isotropic, plane strain), ``x`` = dim ``i`` (rows), ``y`` = dim ``j`` (cols)::

    mu * lap(u) + (lambda + mu) * grad(div u) - alpha * grad(p) = f          (momentum balance)
    div(k grad p)                                             = 0            (Darcy flow, as in ``darcy.py``)

Boundary conditions: clamped solid (``u = 0`` on all four edges), pressure Dirichlet ``p_left`` / ``p_right``
on the ``i`` edges and no-flow on the ``j`` walls. Central second-order differences on interior points; every
constraint is scaled by ``h^2``. Optional body force ``f`` enables method-of-manufactured-solutions verification.
"""

from __future__ import annotations

import torch

from squad1.errors import PhysicsError
from squad1.physics.base import Physics, require_positive
from squad1.physics.darcy import DarcyBC, darcy_residual


class BiotPhysics(Physics):
    domain_name = "darcy_biot"

    def __init__(
        self,
        mu: float = 1.0,
        lam: float = 1.0,
        alpha: float = 0.8,
        bc: DarcyBC | None = None,
        body_force: tuple[torch.Tensor, torch.Tensor] | None = None,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        if mu <= 0 or lam + mu <= 0:
            raise PhysicsError("need mu > 0 and lambda + mu > 0")
        if not 0 <= alpha <= 1.5:
            raise PhysicsError("Biot-Willis coefficient alpha must be in [0, 1.5]")
        if any(w < 0 for w in weights):
            raise PhysicsError("component weights must be non-negative")
        self.mu, self.lam, self.alpha = float(mu), float(lam), float(alpha)
        self.bc = bc or DarcyBC()
        self.body_force = body_force
        self.weights = weights

    def _check_values(self, x: torch.Tensor) -> None:
        require_positive(x, 0, "permeability k")

    # ------------------------------------------------------------------ components
    def components(self, x: torch.Tensor, h: float) -> dict[str, torch.Tensor]:
        k, p, ux, uy = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        darcy = darcy_residual(k, p, h) * h * h  # (B, H-2, W)

        c = (slice(None), slice(1, -1), slice(1, -1))

        def sh(f: torch.Tensor, di: int, dj: int) -> torch.Tensor:
            H, W = f.shape[1:]
            return f[:, 1 + di : H - 1 + di, 1 + dj : W - 1 + dj]

        def dxx(f: torch.Tensor) -> torch.Tensor:
            return sh(f, 1, 0) - 2 * f[c] + sh(f, -1, 0)

        def dyy(f: torch.Tensor) -> torch.Tensor:
            return sh(f, 0, 1) - 2 * f[c] + sh(f, 0, -1)

        def dxy(f: torch.Tensor) -> torch.Tensor:
            return (sh(f, 1, 1) - sh(f, 1, -1) - sh(f, -1, 1) + sh(f, -1, -1)) / 4.0

        def dx1(f: torch.Tensor) -> torch.Tensor:
            return (sh(f, 1, 0) - sh(f, -1, 0)) / 2.0

        def dy1(f: torch.Tensor) -> torch.Tensor:
            return (sh(f, 0, 1) - sh(f, 0, -1)) / 2.0

        # all terms below are h^2 * (second derivative) = O(1) stencil sums
        mech_x = self.mu * (dxx(ux) + dyy(ux)) + (self.lam + self.mu) * (dxx(ux) + dxy(uy)) - self.alpha * h * dx1(p)
        mech_y = self.mu * (dxx(uy) + dyy(uy)) + (self.lam + self.mu) * (dxy(ux) + dyy(uy)) - self.alpha * h * dy1(p)
        if self.body_force is not None:
            fx, fy = self.body_force
            mech_x = mech_x - h * h * _interior(fx, x)
            mech_y = mech_y - h * h * _interior(fy, x)
        bc_rows = torch.cat(
            [
                p[:, 0, :] - self.bc.p_left,
                p[:, -1, :] - self.bc.p_right,
                *(u_edge.flatten(1) for u in (ux, uy) for u_edge in (u[:, 0, :], u[:, -1, :], u[:, :, 0], u[:, :, -1])),
            ],
            dim=1,
        )
        return {"darcy": darcy, "mech_x": mech_x, "mech_y": mech_y, "bc": bc_rows}

    def constraint_vector(self, x: torch.Tensor, h: float) -> torch.Tensor:
        comp = self.components(x, h)
        wd, wx, wy = self.weights
        return torch.cat(
            [
                wd**0.5 * comp["darcy"].flatten(1),
                wx**0.5 * comp["mech_x"].flatten(1),
                wy**0.5 * comp["mech_y"].flatten(1),
                comp["bc"],
            ],
            dim=1,
        )

    def component_losses(self, x: torch.Tensor, h: float) -> dict[str, torch.Tensor]:
        """Per-sample mean-square of each component (for diagnostics / loss balancing)."""
        return {name: v.pow(2).mean(dim=tuple(range(1, v.ndim))) for name, v in self.components(x, h).items()}


def _interior(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    f = torch.as_tensor(f, dtype=x.dtype, device=x.device)
    if f.ndim == 2:
        f = f.unsqueeze(0)
    return f[:, 1:-1, 1:-1]
