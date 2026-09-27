"""PCFM pipeline: generator-space candidate -> physical space -> hard-constraint projection -> generator space."""

from __future__ import annotations

import time
from typing import Any

import torch

from squad1.contracts.candidate import CandidateDesign, ProjectionResult
from squad1.contracts.channels import MODEL_HI, MODEL_LO, ChannelNormalizer
from squad1.errors import ContractError, NonFiniteError
from squad1.physics import BiotPhysics, DarcyBC, DarcyPhysics, get_physics
from squad1.physics.base import Physics
from squad1.projection.projectors import Projector, get_projector, rms_correction


def physics_for(candidate: CandidateDesign) -> Physics:
    """Build the physics evaluator for a candidate from its ``boundary`` / ``physics`` metadata."""
    name = candidate.domain
    bc = candidate.boundary
    if name in ("darcy", "cooling_plate"):
        return DarcyPhysics(DarcyBC(float(bc.get("p_left", 1.0)), float(bc.get("p_right", 0.0))), **candidate.physics)
    if name == "darcy_biot":
        return BiotPhysics(bc=DarcyBC(float(bc.get("p_left", 1.0)), float(bc.get("p_right", 0.0))), **candidate.physics)
    return get_physics(name, **candidate.physics)


class ModelSpacePhysics(Physics):
    """Wrap a physics evaluator so the projector works in **generator (model) space** ``[-1, 1]``.

    The constraint is ``c(to_physical(z))``; Jacobians follow the chain rule automatically, so the minimum-norm
    Gauss-Newton step is the minimum change *in generator units*, log-scaled channels (permeability) are treated
    multiplicatively, positivity is automatic and the admissible box is uniform (``[-1, 1]``).
    """

    def __init__(self, inner: Physics):
        self.inner = inner
        self.domain_name = inner.domain_name
        self.norm = ChannelNormalizer(inner.spec.channels)

    def check(self, x: torch.Tensor) -> None:
        if x.ndim != 4 or x.shape[1] != self.spec.n_channels:
            raise ContractError(f"expected (B, {self.spec.n_channels}, H, W), got {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise NonFiniteError("model-space tensor contains NaN/Inf")

    def constraint_vector(self, x: torch.Tensor, h: float) -> torch.Tensor:
        return self.inner.constraint_vector(self.norm.to_physical(x), h)

    def bounds(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.full_like(x, MODEL_LO), torch.full_like(x, MODEL_HI)

    def clamp_feasible(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(MODEL_LO, MODEL_HI)

    def solve_dependent(self, x: torch.Tensor, h: float) -> torch.Tensor | None:
        out = self.inner.solve_dependent(self.norm.to_physical(x), h)
        return None if out is None else self.norm.to_model(out)


class PCFMPipeline:
    """Project batches of candidates onto ``c(x) = 0``.

    * the result is returned in the representation and dtype of the input;
    * projection runs **in model space** (see :class:`ModelSpacePhysics`): ``max_correction_rms`` is RMS per element in
      ``[-1, 1]`` units, so the threshold means the same for every channel and grid size;
    * rejected samples are returned unchanged (bit-identical) with a reason.
    """

    def __init__(
        self,
        method: str = "gauss_newton",
        max_correction_rms: float | None = None,
        compute_dtype: torch.dtype = torch.float64,
        fallback_solve: bool = True,
        physics: Physics | None = None,
        projector: Projector | None = None,
        **projector_config: Any,
    ):
        self.method = method
        self.compute_dtype = compute_dtype
        self.physics = physics
        cfg = dict(projector_config)
        if max_correction_rms is not None:
            cfg["max_correction_rms"] = max_correction_rms
        self.projector = projector or get_projector(method, **cfg)
        # the exact partial solve rewrites dependent channels, so it is off when the caller pinned channels
        self.fallback_solve = fallback_solve and method != "solve" and not self.projector.config.fixed_channels

    def _fallback(
        self, x: torch.Tensor, x0: torch.Tensor, info: dict[str, Any], physics: Physics, h: float
    ) -> list[bool]:
        """Samples the projector could not bring to the manifold get the physics' exact partial solve (Darcy: re-solve
        ``p`` for the given ``k``) when one exists; reported per sample in ``metadata['fallback_solve']``."""
        used = [False] * x.shape[0]
        if not self.fallback_solve:
            return used
        todo = ~info["converged"] & ~info["rejected"]
        if not bool(todo.any()):
            return used
        try:
            dep = physics.solve_dependent(x0, h)
        except Exception:
            return used
        if dep is None:
            return used
        dep = dep.to(x.dtype)
        with torch.no_grad():
            loss = physics.loss(dep, h)
            corr = rms_correction(dep, x0)
        cap = self.projector.config.max_correction_rms
        tol = self.projector.config.tol
        ok = todo & torch.isfinite(loss) & (loss.sqrt() <= tol) & (loss < info["loss_after"])
        for b in range(x.shape[0]):
            if not bool(ok[b]):
                continue
            used[b] = True
            x[b] = dep[b]
            info["loss_after"][b] = loss[b]
            info["correction"][b] = corr[b]
            if cap is not None and float(corr[b]) > cap:
                info["rejected"][b] = True
                info["reject_reason"][b] = "excessive_correction"
                x[b] = x0[b]
                info["loss_after"][b] = info["loss_before"][b]
                info["correction"][b] = 0.0
            else:
                info["converged"][b] = True
        return used

    def project(self, candidate: CandidateDesign) -> ProjectionResult:
        if not isinstance(candidate, CandidateDesign):
            raise ContractError(f"expected CandidateDesign, got {type(candidate).__name__}")
        t0 = time.perf_counter()
        in_dtype = candidate.tensor.dtype
        physics = ModelSpacePhysics(self.physics or physics_for(candidate))
        x0 = candidate.to_model().tensor.to(self.compute_dtype)
        x, info = self.projector.project(x0, physics, candidate.h, rms_correction)
        fallback = self._fallback(x, x0, info, physics, candidate.h)
        out_model = candidate.to_model().replace(tensor=x.to(in_dtype))
        out = out_model if candidate.representation == "model" else out_model.to_physical()
        # untouched (rejected) samples must come back bit-identical, not via a round trip
        rej = info["rejected"].view(-1, 1, 1, 1)
        out = out.replace(tensor=torch.where(rej, candidate.tensor, out.tensor))
        out = out.replace(
            boundary=dict(candidate.boundary),
            physics=dict(candidate.physics),
            generation={**candidate.generation, "projected_with": info["method"]},
        )
        return ProjectionResult(
            projected=out,
            loss_before=info["loss_before"],
            loss_after=info["loss_after"],
            residual_rms_before=info["loss_before"].sqrt(),
            residual_rms_after=info["loss_after"].sqrt(),
            correction_rms=info["correction"],
            iterations=int(info.get("iterations", 0)),
            converged=info["converged"],
            rejected=info["rejected"],
            reject_reason=info["reject_reason"],
            method=info["method"],
            runtime_s=time.perf_counter() - t0,
            metadata={
                **{k: v for k, v in info.items() if k in ("linear_solver",)},
                "fallback_solve": fallback,
            },
        )
