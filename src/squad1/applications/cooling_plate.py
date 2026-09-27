"""Liquid-cooling-plate application slice, rebuilt on solved physics.

Model (depth-averaged 2-D reduced-order model — **illustrative constants, not validated CFD**):

1. *Flow*: Darcy flow through the permeability layout ``k`` (channels = high ``k``), inlet at ``x_min``, outlet at
   ``x_max`` (same FV discretisation as :mod:`squad1.physics.darcy`). The pump delivers a fixed flow rate ``Q_pump`` so
   the
   required pressure drop is ``dp = dp_scale * Q_pump / Q_nd`` and the velocities scale by the same factor.
2. *Heat*: steady conduction–advection ``rho_c u.grad T = div(kappa grad T) + S`` (upwind advection, harmonic-mean
   conductivity, coolant enters at ``T_in`` on ``x_min``, insulated walls) with a uniform volumetric heat load ``S``.
3. Reported quantities — ``max_temperature_C``, ``pressure_drop_kPa``, ``relative_weight`` (solid fraction),
   ``flow_residual`` and ``energy_balance_error`` — are all computed from those solved fields, so they depend on the
   **layout**, not only on the channel area fraction (the earlier POC's algebraic formulas did not).

All solves are dense, differentiable torch operations (``H*W`` up to a few thousand) so layouts can be optimised
by gradient descent; :func:`optimize_layout` does that with a trust cap and a reject path.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from squad1.contracts.candidate import CandidateDesign
from squad1.contracts.channels import ChannelNormalizer, get_domain
from squad1.errors import ContractError, PhysicsError
from squad1.geometry.voxel_mesh import mesh_stats, voxels_to_mesh, write_binary_stl
from squad1.physics.darcy import DarcyBC, DarcyPhysics, darcy_residual, face_fluxes, solve_pressure

MAX_CELLS = 5000  # dense direct solves (differentiable); larger grids need an iterative solver


@dataclass(frozen=True)
class CoolingParams:
    plate_length_m: float = 0.1
    thickness_m: float = 0.005
    heat_flux_w_m2: float = 2.0e4  # areal heat load from the battery, W/m^2
    t_in_c: float = 25.0
    kappa_solid: float = 200.0  # aluminium, W/m/K
    kappa_fluid: float = (
        30.0  # effective transverse conductivity in channels (includes convective mixing; pure water is 0.6)
    )
    rho_c: float = 4.18e6  # volumetric heat capacity of the coolant, J/m^3/K
    q_pump: float = 1.0e-3  # imposed coolant flow rate per unit depth, m^2/s
    dp_scale_kpa: float = 1.4e4  # calibration: kPa per unit (Q_pump / Q_nd)
    t_target_c: float = 45.0
    dp_target_kpa: float = 10.0
    min_solid_fraction: float = 0.25  # structural minimum: at least this share of the plate must stay solid
    channel_threshold: float = 1.0  # k above this counts as coolant channel (geometric middle of [1e-2, 1e2])
    channel_softness: float = 0.5  # width of the smooth channel indicator in ln(k)

    def __post_init__(self) -> None:
        for name in (
            "plate_length_m",
            "thickness_m",
            "heat_flux_w_m2",
            "kappa_solid",
            "kappa_fluid",
            "rho_c",
            "q_pump",
            "dp_scale_kpa",
            "channel_threshold",
            "channel_softness",
        ):
            if not getattr(self, name) > 0:
                raise PhysicsError(f"{name} must be positive")
        if not 0 <= self.min_solid_fraction < 1:
            raise PhysicsError("min_solid_fraction must be in [0, 1)")
        if self.dp_target_kpa <= 0 or self.t_target_c <= self.t_in_c:
            raise PhysicsError("need dp_target > 0 and t_target > t_in")


def _channel_fraction(k: torch.Tensor, p: CoolingParams, hard: bool = False) -> torch.Tensor:
    z = (torch.log(k) - math.log(p.channel_threshold)) / p.channel_softness
    return (z > 0).to(k.dtype) if hard else torch.sigmoid(z)


class CoolingPlate:
    def __init__(self, params: CoolingParams | None = None, bc: DarcyBC | None = None):
        self.p = params or CoolingParams()
        self.bc = bc or DarcyBC()
        if self.bc.p_left <= self.bc.p_right:
            raise PhysicsError("flow must go from x_min (higher p) to x_max (lower p)")

    # -------------------------------------------------------------------------------- solves
    def solve(self, k: torch.Tensor, h_nd: float, smooth_max_k: float = 0.25) -> dict[str, torch.Tensor]:
        """``k (B, H, W)`` physical permeability -> fields and metrics (all per-sample ``(B,)`` tensors)."""
        if k.ndim != 3 or min(k.shape[1:]) < 8:
            raise ContractError(f"k must be (B, H>=8, W>=8), got {tuple(k.shape)}")
        if k.shape[1] * k.shape[2] > MAX_CELLS:
            raise ContractError(
                f"{k.shape[1] * k.shape[2]} cells exceed the dense-solve limit ({MAX_CELLS}); use a coarser grid"
            )
        if not torch.isfinite(k).all() or (k <= 0).any():
            raise PhysicsError("k must be finite and strictly positive")
        P = self.p
        _, H, W = k.shape
        dp_nd = self.bc.p_left - self.bc.p_right
        p = solve_pressure(k, h_nd, self.bc, method="dense")
        qx, qy = face_fluxes(k, p, h_nd)  # k * dp/dn
        q_nd = (-qx[:, 0, :]).sum(dim=1) * h_nd  # flow rate through the first face row (per unit depth)
        scale = P.q_pump / (q_nd.clamp_min(1e-30) * P.plate_length_m)  # physical / non-dimensional velocity
        ux, uy = -qx * scale.view(-1, 1, 1), -qy * scale.view(-1, 1, 1)  # m/s (velocity = -k grad p)
        h_m = P.plate_length_m / (H - 1)
        phi = _channel_fraction(k, P)
        kappa = P.kappa_solid * (1 - phi) + P.kappa_fluid * phi
        T = self._solve_temperature(kappa, ux, uy, h_m)
        interior = T[:, 1:, :]
        tmax = interior.flatten(1).max(dim=1).values
        beta = 1.0 / smooth_max_k  # log-sum-exp is an upper bound of the max, within ln(N)/beta
        tsmooth = torch.logsumexp(beta * interior.flatten(1), dim=1) / beta
        dp_kpa = P.dp_scale_kpa * scale * dp_nd
        hard = _channel_fraction(k, P, hard=True)
        # energy balance: enthalpy leaving through the outlet row - entering at the inlet == total heat source
        rho_c = P.rho_c
        q_out = (ux[:, -1, :].clamp_min(0) * (T[:, -1, :] - P.t_in_c)).sum(dim=1) * h_m * rho_c
        src = P.heat_flux_w_m2 / P.thickness_m
        total_src = src * h_m * h_m * (H - 1) * W
        # heat also leaves by conduction into the Dirichlet inlet row
        kap_face = 2 * kappa[:, 0, :] * kappa[:, 1, :] / (kappa[:, 0, :] + kappa[:, 1, :])
        q_cond = (kap_face * (T[:, 1, :] - P.t_in_c)).sum(dim=1)
        energy_err = (q_out + q_cond - total_src).abs() / total_src
        return {
            "p": p,
            "T": T,
            "ux": ux,
            "uy": uy,
            "max_temperature_C": tmax,
            "smooth_max_temperature_C": tsmooth,
            "pressure_drop_kPa": dp_kpa,
            "relative_weight": 1.0 - hard.flatten(1).mean(dim=1),
            "soft_relative_weight": 1.0 - phi.flatten(1).mean(dim=1),
            "flow_residual": (darcy_residual(k, p, h_nd) * h_nd * h_nd).flatten(1).pow(2).mean(dim=1).sqrt(),
            "energy_balance_error": energy_err,
            "flow_rate_nd": q_nd,
        }

    def _solve_temperature(self, kappa: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, h_m: float) -> torch.Tensor:
        P = self.p
        B, H, W = kappa.shape
        N = H * W
        idx = torch.arange(N, device=kappa.device).view(H, W)
        c = P.rho_c / h_m
        rows, cols, vals = [], [], []

        def add(r: torch.Tensor, cc: torch.Tensor, v: torch.Tensor) -> None:
            rows.append(r.reshape(-1))
            cols.append(cc.reshape(-1))
            vals.append(v.reshape(B, -1))

        for a, b, kap_a, kap_b, v in (
            (idx[:-1, :], idx[1:, :], kappa[:, :-1, :], kappa[:, 1:, :], ux),
            (idx[:, :-1], idx[:, 1:], kappa[:, :, :-1], kappa[:, :, 1:], uy),
        ):
            kd = 2 * kap_a * kap_b / (kap_a + kap_b) / h_m**2
            pos, neg = torch.relu(v), torch.relu(-v)
            add(a, a, kd + c * pos)
            add(a, b, -kd - c * neg)
            add(b, b, kd + c * neg)
            add(b, a, -kd - c * pos)
        add(idx[-1, :], idx[-1, :], c * torch.relu(ux[:, -1, :]))  # outflow through the outlet boundary
        r = torch.cat(rows)
        cc = torch.cat(cols)
        v = torch.cat(vals, dim=1)
        A = torch.zeros(B, N * N, dtype=kappa.dtype, device=kappa.device).index_add(1, r * N + cc, v).view(B, N, N)
        keep = torch.ones(N, dtype=kappa.dtype, device=kappa.device)
        keep[:W] = 0.0  # inlet row: Dirichlet T = T_in
        A = A * keep.view(1, N, 1) + torch.diag(1 - keep).unsqueeze(0)
        src = P.heat_flux_w_m2 / P.thickness_m
        rhs = (keep * src + (1 - keep) * P.t_in_c).unsqueeze(0).expand(B, N)
        d = (
            torch.diagonal(A, dim1=-2, dim2=-1).abs().clamp_min(1e-300)
        )  # Jacobi row scaling: entries O(1), better conditioned
        return torch.linalg.solve(A / d.unsqueeze(-1), (rhs / d).unsqueeze(-1)).squeeze(-1).view(B, H, W)

    # ------------------------------------------------------------------------------- reports
    def report(self, k: torch.Tensor, h_nd: float) -> list[dict[str, Any]]:
        with torch.no_grad():
            r = self.solve(k, h_nd)
        P = self.p
        out = []
        for b in range(k.shape[0]):
            t, dp = float(r["max_temperature_C"][b]), float(r["pressure_drop_kPa"][b])
            out.append(
                {
                    "max_temperature_C": t,
                    "pressure_drop_kPa": dp,
                    "relative_weight": float(r["relative_weight"][b]),
                    "flow_residual": float(r["flow_residual"][b]),
                    "energy_balance_error": float(r["energy_balance_error"][b]),
                    "temperature_ok": t <= P.t_target_c,
                    "pressure_drop_ok": dp <= P.dp_target_kpa,
                    "structure_ok": float(r["relative_weight"][b]) >= P.min_solid_fraction,
                    "constraints_satisfied": bool(
                        t <= P.t_target_c
                        and dp <= P.dp_target_kpa
                        and float(r["relative_weight"][b]) >= P.min_solid_fraction
                        and float(r["flow_residual"][b]) < 1e-8
                    ),
                    "targets": {"max_temperature_C": P.t_target_c, "pressure_drop_kPa": P.dp_target_kpa},
                }
            )
        return out


# --------------------------------------------------------------------------------------- optimisation
@dataclass
class LayoutConfig:
    steps: int = 120
    lr: float = 0.08
    margin: float = 0.05  # aim this far inside the limits
    w_temp: float = 1.0
    w_dp: float = 1.0
    w_weight: float = 0.05
    w_struct: float = 2.0
    max_correction_rms: float | None = None  # model-space RMS cap on the change of the k channel; None = no cap
    seed: int = 0


def optimize_layout(
    plate: CoolingPlate, candidate: CandidateDesign, cfg: LayoutConfig | None = None
) -> tuple[CandidateDesign, dict[str, Any]]:
    """Adjust the permeability layout until temperature and pressure-drop limits hold (with margin).

    The pressure channel is **re-solved from ``k``** at the end, so the returned design satisfies the flow physics
    exactly (``flow_residual ~ 0``) — the projection can no longer degrade flow consistency. Samples whose required
    change exceeds ``max_correction_rms`` (model space) are rejected and returned unchanged.
    """
    cfg = cfg or LayoutConfig()
    if candidate.domain not in ("cooling_plate", "darcy"):
        raise ContractError("optimize_layout needs a 'cooling_plate' (or 'darcy') candidate")
    t0 = time.perf_counter()
    P = plate.p
    spec = get_domain(candidate.domain)
    norm = ChannelNormalizer(spec.channels)
    phys = candidate.to_physical().tensor.double()
    k0 = phys[:, 0].clone()
    h = candidate.h
    lo, hi = spec.channels[0].lo, spec.channels[0].hi
    # unconstrained parameter: z in (-inf, inf) with model-space value tanh(z) in (-1, 1)
    m0 = norm.to_model(phys)[:, 0].clamp(-0.999, 0.999)
    z = torch.atanh(m0).clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=cfg.lr)
    t_lim, dp_lim = (
        P.t_in_c + (P.t_target_c - P.t_in_c) * (1 - cfg.margin),
        P.dp_target_kpa * (1 - cfg.margin),
    )

    def k_of(zv: torch.Tensor) -> torch.Tensor:
        m = torch.tanh(zv)
        return torch.exp(math.log(lo) + (m + 1) / 2 * (math.log(hi) - math.log(lo)))

    before = plate.report(k0, h)
    for _ in range(cfg.steps):
        r = plate.solve(k_of(z), h)
        struct_lim = P.min_solid_fraction * (1 + cfg.margin)
        # linear (not squared): a squared hinge has vanishing gradient right at the boundary, which stalls
        # Adam just short of the limit; the linear form keeps pushing until the constraint is actually met.
        loss = (
            cfg.w_temp * torch.relu(r["smooth_max_temperature_C"] - t_lim).pow(2) / (P.t_target_c - P.t_in_c) ** 2
            + cfg.w_dp * torch.relu(r["pressure_drop_kPa"] - dp_lim).pow(2) / P.dp_target_kpa**2
            + cfg.w_struct * torch.relu(struct_lim - r["soft_relative_weight"])
            + cfg.w_weight * r["soft_relative_weight"]
        )
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        k1 = k_of(z).detach()
    m1 = norm.to_model(torch.stack([k1, torch.zeros_like(k1)], 1))[:, 0]
    m0f = norm.to_model(torch.stack([k0, torch.zeros_like(k0)], 1))[:, 0]
    corr = (m1 - m0f).flatten(1).pow(2).mean(dim=1).sqrt()
    rejected = torch.zeros(k0.shape[0], dtype=torch.bool)
    reasons: list[str | None] = [None] * k0.shape[0]
    if cfg.max_correction_rms is not None:
        rejected = corr > cfg.max_correction_rms
        for b in range(len(reasons)):
            if bool(rejected[b]):
                reasons[b] = "excessive_correction"
    k_final = torch.where(rejected.view(-1, 1, 1), k0, k1)
    p_final = DarcyPhysics(plate.bc).solve(k_final, h)[:, 1]  # flow physics exact by construction
    out_phys = candidate.to_physical().replace(tensor=torch.stack([k_final, p_final], 1).to(candidate.tensor.dtype))
    out = out_phys.to_model() if candidate.representation == "model" else out_phys
    out = out.replace(tensor=torch.where(rejected.view(-1, 1, 1, 1), candidate.tensor, out.tensor))
    after = plate.report(k_final, h)
    info = {
        "method": "cooling_layout_gradient",
        "steps": cfg.steps,
        "runtime_s": time.perf_counter() - t0,
        "report_before": before,
        "report_after": after,
        "correction_rms": corr.tolist(),
        "rejected": rejected.tolist(),
        "reject_reason": reasons,
        "config": asdict(cfg),
    }
    return out, info


# ---------------------------------------------------------------------------------------- export
def plate_to_stl(
    k_phys: np.ndarray | torch.Tensor,
    path: str | Path,
    params: CoolingParams | None = None,
    layers: int = 4,
) -> dict[str, Any]:
    """Export the *solid plate with channels as voids* (one ``(H, W)`` sample) as a watertight binary STL in mm."""
    P = params or CoolingParams()
    k = k_phys.detach().cpu().numpy() if isinstance(k_phys, torch.Tensor) else np.asarray(k_phys)
    if k.ndim != 2 or min(k.shape) < 4:
        raise ContractError(f"expected one (H, W) permeability map, got {k.shape}")
    solid = k <= P.channel_threshold
    if not solid.any():
        raise PhysicsError("plate has no solid cells: nothing to export")
    H = k.shape[0]
    cell_mm = P.plate_length_m * 1e3 / (H - 1)  # same cell size as the thermal model (plate = (H-1) x W cells)
    depth_mm = P.thickness_m * 1e3
    occ = np.repeat(solid[:, :, None], layers, axis=2)
    mesh = voxels_to_mesh(occ)
    mesh.vertices[:, 0:2] *= cell_mm
    mesh.vertices[:, 2] *= depth_mm / layers
    st = mesh_stats(mesh)
    write_binary_stl(path, mesh)
    return {
        "path": str(path),
        "vertices": st.n_vertices,
        "faces": st.n_faces,
        "watertight": st.watertight,
        "volume_mm3": st.volume,
        "positive_volume": st.volume > 0,
        "components": st.components,
        "cell_mm": cell_mm,
        "thickness_mm": depth_mm,
    }
