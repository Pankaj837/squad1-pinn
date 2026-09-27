"""Channel registry and model-space <-> physical-space conversion.

Contract (see docs/INTERFACES.md):

* design tensors are ``torch`` floating tensors, channel-first ``(B, C, H, W)``;
* *model space* is the generator's working range, ``[-1, 1]`` per channel;
* *physical space* is described per channel by ``ChannelSpec`` (``lo``, ``hi``, ``scale``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from squad1.errors import ContractError, NormalizationError

MODEL_LO, MODEL_HI = -1.0, 1.0


@dataclass(frozen=True)
class ChannelSpec:
    """Physical meaning and range of one channel."""

    name: str
    lo: float
    hi: float
    scale: str = "linear"  # "linear" | "log"
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise NormalizationError("channel name must be non-empty")
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)) or not self.hi > self.lo:
            raise NormalizationError(f"channel {self.name!r}: need finite lo < hi, got ({self.lo}, {self.hi})")
        if self.scale not in ("linear", "log"):
            raise NormalizationError(f"channel {self.name!r}: scale must be 'linear' or 'log', got {self.scale!r}")
        if self.scale == "log" and self.lo <= 0:
            raise NormalizationError(f"channel {self.name!r}: log scale requires lo > 0")


@dataclass(frozen=True)
class DomainSpec:
    """A physics domain: ordered channels plus a human description."""

    name: str
    channels: tuple[ChannelSpec, ...]
    description: str = ""
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        names = [c.name for c in self.channels]
        if not names:
            raise ContractError(f"domain {self.name!r} has no channels")
        if len(set(names)) != len(names):
            raise ContractError(f"domain {self.name!r} has duplicate channel names: {names}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.channels)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise ContractError(f"domain {self.name!r} has no channel {name!r}; channels={self.names}") from None


_REGISTRY: dict[str, DomainSpec] = {}


def register_domain(spec: DomainSpec, overwrite: bool = False) -> DomainSpec:
    if spec.name in _REGISTRY and not overwrite and _REGISTRY[spec.name] != spec:
        raise ContractError(f"domain {spec.name!r} already registered with a different spec")
    _REGISTRY[spec.name] = spec
    return spec


def get_domain(name: str) -> DomainSpec:
    if name not in _REGISTRY:
        import squad1.physics.domains  # noqa: F401  (registers the standard domain library lazily)
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ContractError(f"unknown domain {name!r}; registered: {sorted(_REGISTRY)}") from None


def available_domains() -> list[str]:
    import squad1.physics.domains  # noqa: F401

    return sorted(_REGISTRY)


# ---- built-in Darcy family (the generator contract used by the DiT and the cooling-plate application) ----
_K = ChannelSpec("k", 1e-2, 1e2, "log", "1")
_P = ChannelSpec("p", 0.0, 1.0, "linear", "1")
register_domain(DomainSpec("darcy", (_K, _P), "Darcy flow: permeability k and pressure p, (B,2,H,W)"))
register_domain(
    DomainSpec(
        "darcy_biot",
        (_K, _P, ChannelSpec("ux", -1.0, 1.0), ChannelSpec("uy", -1.0, 1.0)),
        "Darcy + quasi-static poroelastic displacement (k, p, ux, uy)",
    )
)
register_domain(
    DomainSpec(
        "cooling_plate",
        (ChannelSpec("k", 1e-2, 1e2, "log", "1"), _P),
        "Cooling-plate topology: k = channel layout (high = coolant), p = coolant pressure",
    )
)


class ChannelNormalizer:
    """Convert ``(B, C, ...)`` tensors between model space ``[-1, 1]`` and physical space.

    * channel axis is **dim 1**;
    * dtype, device and autograd graph are preserved;
    * round-trip is exact up to floating point error.
    """

    def __init__(self, channels: Sequence[ChannelSpec]):
        self.channels = tuple(channels)
        if not self.channels:
            raise NormalizationError("at least one channel required")

    @classmethod
    def for_domain(cls, domain: str | DomainSpec) -> ChannelNormalizer:
        spec = get_domain(domain) if isinstance(domain, str) else domain
        return cls(spec.channels)

    def _check(self, x: torch.Tensor) -> None:
        if x.ndim < 2 or x.shape[1] != len(self.channels):
            raise ContractError(
                f"expected channel dim (dim 1) == {len(self.channels)} "
                f"({[c.name for c in self.channels]}), got shape {tuple(x.shape)}"
            )
        if not torch.is_floating_point(x):
            raise ContractError(f"expected a floating tensor, got {x.dtype}")

    def _bounds(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = [1] * x.ndim
        shape[1] = -1
        log = torch.tensor([c.scale == "log" for c in self.channels], device=x.device).view(shape)
        lo = torch.tensor(
            [math.log(c.lo) if c.scale == "log" else c.lo for c in self.channels],
            dtype=x.dtype,
            device=x.device,
        ).view(shape)
        hi = torch.tensor(
            [math.log(c.hi) if c.scale == "log" else c.hi for c in self.channels],
            dtype=x.dtype,
            device=x.device,
        ).view(shape)
        return log, lo, hi

    def to_physical(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        log, lo, hi = self._bounds(x)
        u = (x - MODEL_LO) / (MODEL_HI - MODEL_LO)  # [0, 1]
        v = lo + u * (hi - lo)
        return torch.where(log, torch.exp(v), v)

    def to_model(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        log, lo, hi = self._bounds(x)
        safe = torch.where(log, x.clamp_min(torch.finfo(x.dtype).tiny), x)
        v = torch.where(log, torch.log(safe), x)
        u = (v - lo) / (hi - lo)
        return MODEL_LO + u * (MODEL_HI - MODEL_LO)

    def clip_model(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        return x.clamp(MODEL_LO, MODEL_HI)
