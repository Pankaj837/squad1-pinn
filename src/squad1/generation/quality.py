"""Generator output quality gate (shape / finite / range / physics residual) before hand-off to projection."""

from __future__ import annotations

from typing import Any

import torch

from squad1.contracts.candidate import CandidateDesign
from squad1.contracts.channels import MODEL_HI, MODEL_LO
from squad1.errors import ContractError
from squad1.physics.base import Physics


def quality_check(
    candidate: CandidateDesign,
    physics: Physics | None = None,
    *,
    range_tol: float = 1e-3,
    max_out_of_range_fraction: float = 0.02,
    max_residual_rms: float | None = None,
) -> dict[str, Any]:
    """Per-sample report. ``ok[i]`` is False when a sample is non-finite, mostly outside ``[-1, 1]``,
    or (if ``max_residual_rms`` is given) its physics residual is above the limit."""
    t = candidate.tensor
    B = t.shape[0]
    finite = torch.isfinite(t).flatten(1).all(dim=1)
    out = ((t < MODEL_LO - range_tol) | (t > MODEL_HI + range_tol)).flatten(1).float().mean(dim=1)
    ok = finite & (out <= max_out_of_range_fraction)
    report: dict[str, Any] = {
        "batch": B,
        "domain": candidate.domain,
        "shape": list(t.shape),
        "finite": finite.tolist(),
        "out_of_range_fraction": out.tolist(),
    }
    if physics is not None:
        phys = candidate.to_physical().tensor.double()
        try:
            physics.check(phys)
            r = physics.residual_rms(phys, candidate.h)
            report["residual_rms"] = r.tolist()
            if max_residual_rms is not None:
                ok = ok & (r <= max_residual_rms)
        except (ContractError, ValueError) as exc:
            report["physics_error"] = str(exc)
            ok = torch.zeros_like(ok)
    report["ok"] = ok.tolist()
    report["n_ok"] = int(ok.sum())
    return report
