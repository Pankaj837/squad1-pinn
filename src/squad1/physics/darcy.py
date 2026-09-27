"""Darcy flow ``div(k grad p) = 0`` — conservative finite-volume discretisation.

Layout: fields ``(B, H, W)``; index ``i`` (rows, size H) is the flow direction ``x``; ``j`` (cols, size W) is ``y``.
Boundary conditions: Dirichlet ``p_left`` at ``i = 0`` and ``p_right`` at ``i = H-1``; no-flow (zero normal flux)
on the ``j`` walls. Face permeability is the harmonic mean of the neighbouring cells (correct for high-contrast
channel / solid layouts). The residual, its gradient (autograd) and the reference solver use the *same* stencil.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from squad1.errors import ConvergenceError, PhysicsError
from squad1.physics.base import Physics, require_positive


@dataclass(frozen=True)
class DarcyBC:
    p_left: float = 1.0
    p_right: float = 0.0


def harmonic_mean(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    return 2.0 * a * b / (a + b + eps)


def face_fluxes(k: torch.Tensor, p: torch.Tensor, h: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Face fluxes ``k_face * dp/dn``: ``qx (B, H-1, W)`` and ``qy (B, H, W-1)``."""
    qx = harmonic_mean(k[:, 1:, :], k[:, :-1, :]) * (p[:, 1:, :] - p[:, :-1, :]) / h
    qy = harmonic_mean(k[:, :, 1:], k[:, :, :-1]) * (p[:, :, 1:] - p[:, :, :-1]) / h
    return qx, qy


def darcy_residual(k: torch.Tensor, p: torch.Tensor, h: float) -> torch.Tensor:
    """Cell-wise ``div(k grad p)`` for interior rows ``i = 1..H-2``; shape ``(B, H-2, W)``."""
    qx, qy = face_fluxes(k, p, h)
    div_x = (qx[:, 1:, :] - qx[:, :-1, :]) / h
    zero = torch.zeros_like(qy[:, :, :1])
    qy_pad = torch.cat([zero, qy, zero], dim=2)  # no-flow walls
    div_y = (qy_pad[:, 1:-1, 1:] - qy_pad[:, 1:-1, :-1]) / h
    return div_x + div_y


def _conductances(k: torch.Tensor, h: float) -> tuple[torch.Tensor, torch.Tensor]:
    tx = harmonic_mean(k[:, 1:, :], k[:, :-1, :]) / h**2  # (B, H-1, W)
    ty = harmonic_mean(k[:, :, 1:], k[:, :, :-1]) / h**2  # (B, H, W-1)
    return tx, ty


def _assemble_dense(k: torch.Tensor, h: float, bc: DarcyBC) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense SPD system ``A u = b`` for the interior unknowns (rows 1..H-2), differentiable in ``k``."""
    B, H, W = k.shape
    tx, ty = _conductances(k, h)
    tw, te = tx[:, :-1, :], tx[:, 1:, :]  # (B, H-2, W): west/east face of interior row i
    ty_int = ty[:, 1:-1, :]  # (B, H-2, W-1)
    ts, tn = F.pad(ty_int, (1, 0)), F.pad(ty_int, (0, 1))
    n = (H - 2) * W
    diag = (tw + te + ts + tn).reshape(B, n)
    A = torch.diag_embed(diag)
    # x-neighbours: cell (i, j) <-> (i+1, j), i = 1..H-3   (H >= 4 guaranteed by the caller)
    m = (H - 3) * W
    ix = torch.arange(m, device=k.device)
    cx = te[:, :-1, :].reshape(B, m)
    A = _put(A, ix, ix + W, -cx)
    A = _put(A, ix + W, ix, -cx)
    # y-neighbours: (i, j) <-> (i, j+1)
    cy = ty_int.reshape(B, -1)
    rows = (torch.arange(H - 2, device=k.device)[:, None] * W + torch.arange(W - 1, device=k.device)[None, :]).reshape(
        -1
    )
    A = _put(A, rows, rows + 1, -cy)
    A = _put(A, rows + 1, rows, -cy)
    rhs = F.pad(tw[:, :1] * bc.p_left, (0, 0, 0, H - 3)) + F.pad(te[:, -1:] * bc.p_right, (0, 0, H - 3, 0))
    return A, rhs.reshape(B, n)


def _put(A: torch.Tensor, r: torch.Tensor, c: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Functional (autograd-safe) scatter of ``v (B, len(r))`` into ``A[:, r, c]``."""
    out = A.clone()
    out[:, r, c] = v
    return out


def solve_pressure(
    k: torch.Tensor,
    h: float,
    bc: DarcyBC | None = None,
    method: str = "dense",
    tol: float = 1e-10,
    max_iter: int = 5000,
) -> torch.Tensor:
    """Solve for the full pressure field ``(B, H, W)`` given ``k (B, H, W)``.

    ``method="dense"`` is differentiable and exact (use for H*W up to a few thousand);
    ``method="cg"`` is matrix-free Jacobi-preconditioned CG (device friendly, not differentiable).
    """
    bc = bc or DarcyBC()
    if k.ndim != 3 or k.shape[1] < 4 or k.shape[2] < 2:
        raise PhysicsError(f"k must be (B, H>=4, W>=2); got {tuple(k.shape)}")
    if (k <= 0).any() or not torch.isfinite(k).all():
        raise PhysicsError("k must be finite and strictly positive")
    B, H, W = k.shape
    if method == "dense":
        A, rhs = _assemble_dense(k, h, bc)
        u = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1).reshape(B, H - 2, W)
    elif method == "cg":
        u = _solve_cg(k, h, bc, tol, max_iter)
    else:
        raise PhysicsError(f"unknown method {method!r}; use 'dense' or 'cg'")
    left = torch.full((B, 1, W), bc.p_left, dtype=k.dtype, device=k.device)
    right = torch.full((B, 1, W), bc.p_right, dtype=k.dtype, device=k.device)
    return torch.cat([left, u, right], dim=1)


def _solve_cg(k: torch.Tensor, h: float, bc: DarcyBC, tol: float, max_iter: int) -> torch.Tensor:
    B, H, W = k.shape
    tx, ty = _conductances(k, h)
    ty_int = ty[:, 1:-1, :]
    diag = tx[:, :-1, :] + tx[:, 1:, :] + F.pad(ty_int, (1, 0)) + F.pad(ty_int, (0, 1))  # (B, H-2, W)

    def full(u: torch.Tensor) -> torch.Tensor:
        left = torch.full((B, 1, W), bc.p_left, dtype=k.dtype, device=k.device)
        right = torch.full((B, 1, W), bc.p_right, dtype=k.dtype, device=k.device)
        return torch.cat([left, u, right], dim=1)

    zero = torch.zeros(B, H - 2, W, dtype=k.dtype, device=k.device)
    r0 = darcy_residual(k, full(zero), h)  # residual at u = 0

    def op(u: torch.Tensor) -> torch.Tensor:  # SPD operator  -L u
        return -(darcy_residual(k, full(u), h) - r0)

    b = r0  # (-L) u = r0   <=>   L u + r0 = 0
    u = zero.clone()
    r = b - op(u)
    z = r / diag
    d = z.clone()
    rz = (r * z).sum(dim=(1, 2))
    bnorm = b.flatten(1).norm(dim=1).clamp_min(1e-300)
    for _ in range(max_iter):
        Ad = op(d)
        alpha = rz / (d * Ad).sum(dim=(1, 2)).clamp_min(1e-300)
        u = u + alpha.view(-1, 1, 1) * d
        r = r - alpha.view(-1, 1, 1) * Ad
        if bool((r.flatten(1).norm(dim=1) / bnorm < tol).all()):
            return u
        z = r / diag
        rz_new = (r * z).sum(dim=(1, 2))
        d = z + (rz_new / rz.clamp_min(1e-300)).view(-1, 1, 1) * d
        rz = rz_new
    raise ConvergenceError(f"CG did not reach tol={tol:g} in {max_iter} iterations")


class DarcyPhysics(Physics):
    """``(B, 2, H, W)`` design tensor with channels ``[k, p]`` (physical space)."""

    domain_name = "darcy"

    def __init__(self, bc: DarcyBC | None = None, solver: str = "dense"):
        self.bc = bc or DarcyBC()
        self.solver = solver

    def _check_values(self, x: torch.Tensor) -> None:
        require_positive(x, 0, "permeability k")

    def constraint_vector(self, x: torch.Tensor, h: float) -> torch.Tensor:
        k, p = x[:, 0], x[:, 1]
        r = darcy_residual(k, p, h) * h * h
        bc = torch.cat([p[:, 0, :] - self.bc.p_left, p[:, -1, :] - self.bc.p_right], dim=1)
        return torch.cat([r.flatten(1), bc], dim=1)

    def solve_dependent(self, x: torch.Tensor, h: float) -> torch.Tensor:
        """Exact projection of ``p`` for fixed ``k``: re-solve the flow. ``k`` is returned unchanged."""
        k = x[:, 0]
        p = solve_pressure(k.detach(), h, self.bc, method=self.solver)
        return torch.stack([k, p], dim=1)

    def solve(self, k: torch.Tensor, h: float) -> torch.Tensor:
        """Convenience: ``k (B, H, W)`` -> full design tensor ``(B, 2, H, W)``."""
        return self.solve_dependent(torch.stack([k, torch.zeros_like(k)], dim=1), h)
