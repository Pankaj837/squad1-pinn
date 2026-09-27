"""Generator -> physics/projection -> downstream contract objects."""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from squad1.contracts.channels import ChannelNormalizer, DomainSpec, get_domain
from squad1.errors import ContractError, NonFiniteError

REPRESENTATIONS = ("model", "physical")


def check_design_tensor(x: Any, spec: DomainSpec, *, min_size: int = 4) -> torch.Tensor:
    """Validate a design tensor against a domain spec (channel-first ``(B, C, H, W)``, floating, finite)."""
    if not isinstance(x, torch.Tensor):
        raise ContractError(f"design tensor must be a torch.Tensor, got {type(x).__name__}")
    if x.ndim != 4:
        raise ContractError(f"design tensor must be (B, C, H, W); got shape {tuple(x.shape)}")
    if not torch.is_floating_point(x):
        raise ContractError(f"design tensor must be floating point, got {x.dtype}")
    if x.shape[0] < 1:
        raise ContractError("empty batch (B = 0)")
    if x.shape[1] != spec.n_channels:
        raise ContractError(
            f"domain {spec.name!r} expects {spec.n_channels} channels {spec.names}; "
            f"got {x.shape[1]} (shape {tuple(x.shape)})"
        )
    if x.shape[2] < min_size or x.shape[3] < min_size:
        raise ContractError(f"spatial size must be >= {min_size}x{min_size}; got {tuple(x.shape[2:])}")
    if not torch.isfinite(x).all():
        raise NonFiniteError("design tensor contains NaN/Inf")
    return x


@dataclass
class CandidateDesign:
    """A batch of designs for one domain.

    ``representation`` is ``"model"`` (generator space, ``[-1, 1]``) or ``"physical"`` (units of the domain spec).
    ``h`` is the grid spacing (domain length / (H-1)); physics needs it, so it always travels with the tensor.
    """

    domain: str
    tensor: torch.Tensor
    representation: str = "model"
    h: float = 1.0
    boundary: dict[str, Any] = field(default_factory=dict)
    physics: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.representation not in REPRESENTATIONS:
            raise ContractError(f"representation must be one of {REPRESENTATIONS}, got {self.representation!r}")
        if isinstance(self.h, bool) or not isinstance(self.h, numbers.Real) or not self.h > 0:
            raise ContractError(f"h must be a positive number, got {self.h!r}")
        self.h = float(self.h)
        check_design_tensor(self.tensor, self.spec)

    @property
    def spec(self) -> DomainSpec:
        return get_domain(self.domain)

    @property
    def batch_size(self) -> int:
        return int(self.tensor.shape[0])

    def replace(self, **changes: Any) -> CandidateDesign:
        return replace(self, **changes)

    def to_physical(self) -> CandidateDesign:
        if self.representation == "physical":
            return self
        t = ChannelNormalizer(self.spec.channels).to_physical(self.tensor)
        return self.replace(tensor=t, representation="physical")

    def to_model(self) -> CandidateDesign:
        if self.representation == "model":
            return self
        t = ChannelNormalizer(self.spec.channels).to_model(self.tensor)
        return self.replace(tensor=t, representation="model")


@dataclass
class ProjectionResult:
    """Outcome of projecting a batch onto the physics constraint manifold.

    ``projected`` is returned **in the representation the input had**. For rejected samples the tensor is the
    unchanged input (finite, schema-valid) and ``reject_reason`` is set.
    """

    projected: CandidateDesign
    loss_before: torch.Tensor  # (B,)
    loss_after: torch.Tensor  # (B,)
    residual_rms_before: torch.Tensor  # (B,)
    residual_rms_after: torch.Tensor  # (B,)
    correction_rms: torch.Tensor  # (B,) RMS per element, in the space the physics lives in
    iterations: int
    converged: torch.Tensor  # (B,) bool
    rejected: torch.Tensor  # (B,) bool
    reject_reason: list[str | None]
    method: str
    runtime_s: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_converged(self) -> bool:
        return bool(self.converged.all())

    @property
    def any_rejected(self) -> bool:
        return bool(self.rejected.any())

    def to_dict(self) -> dict[str, Any]:
        def lst(t: torch.Tensor) -> list:
            return [float(v) if t.dtype.is_floating_point else bool(v) for v in t.detach().cpu().tolist()]

        return {
            "method": self.method,
            "iterations": self.iterations,
            "runtime_s": self.runtime_s,
            "loss_before": lst(self.loss_before),
            "loss_after": lst(self.loss_after),
            "residual_rms_before": lst(self.residual_rms_before),
            "residual_rms_after": lst(self.residual_rms_after),
            "correction_rms": lst(self.correction_rms),
            "converged": lst(self.converged),
            "rejected": lst(self.rejected),
            "reject_reason": list(self.reject_reason),
            "metadata": self.metadata,
        }
