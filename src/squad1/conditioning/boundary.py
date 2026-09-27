"""Boundary specification -> spatial conditioning field ``cond_field (B, Cc, H, W)``.

Deterministic rasteriser (the "boundary parser" back end): a validated :class:`BoundarySpec` becomes a fixed set of
named channels, the same spatial size as the design tensor. Sides use the design-tensor axes:
``x_min``/``x_max`` are rows ``i = 0`` / ``i = H-1``; ``y_min``/``y_max`` are columns ``j = 0`` / ``j = W-1``.

Channels (``Cc = 8``): dirichlet_mask, dirichlet_value, neumann_mask, neumann_flux, robin_mask, robin_coeff,
robin_ambient, source.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from squad1.errors import ConditioningError

COND_FIELD_CHANNELS: tuple[str, ...] = (
    "dirichlet_mask",
    "dirichlet_value",
    "neumann_mask",
    "neumann_flux",
    "robin_mask",
    "robin_coeff",
    "robin_ambient",
    "source",
)
SIDES = ("x_min", "x_max", "y_min", "y_max")
KINDS = ("dirichlet", "neumann", "robin")


@dataclass(frozen=True)
class BoundarySegment:
    side: str
    kind: str
    value: float | tuple[float, float]  # dirichlet: value; neumann: flux; robin: (coefficient, ambient)
    start: float = 0.0  # fraction along the side, 0..1
    end: float = 1.0

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise ConditioningError(f"side must be one of {SIDES}, got {self.side!r}")
        if self.kind not in KINDS:
            raise ConditioningError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if not (0.0 <= self.start < self.end <= 1.0):
            raise ConditioningError(f"need 0 <= start < end <= 1, got ({self.start}, {self.end})")
        vals = self.value if isinstance(self.value, tuple) else (self.value,)
        want = 2 if self.kind == "robin" else 1
        if len(vals) != want:
            raise ConditioningError(f"{self.kind} boundary needs {want} value(s), got {len(vals)}")
        if not all(math.isfinite(float(v)) for v in vals):
            raise ConditioningError("boundary values must be finite")
        if self.kind == "robin" and float(vals[0]) < 0:
            raise ConditioningError("Robin coefficient must be >= 0")


@dataclass(frozen=True)
class SourceRegion:
    """Rectangular source (fractions of the domain): ``x0 <= x <= x1`` (rows), ``y0 <= y <= y1`` (cols)."""

    x0: float
    x1: float
    y0: float
    y1: float
    value: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.x0 < self.x1 <= 1.0 and 0.0 <= self.y0 < self.y1 <= 1.0):
            raise ConditioningError("source region must satisfy 0 <= a0 < a1 <= 1 on both axes")
        if not math.isfinite(self.value):
            raise ConditioningError("source value must be finite")


@dataclass(frozen=True)
class BoundarySpec:
    segments: tuple[BoundarySegment, ...] = ()
    sources: tuple[SourceRegion, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [
                {
                    "side": s.side,
                    "kind": s.kind,
                    "value": list(s.value) if isinstance(s.value, tuple) else s.value,
                    "start": s.start,
                    "end": s.end,
                }
                for s in self.segments
            ],
            "sources": [asdict(r) for r in self.sources],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> BoundarySpec:
        segs = tuple(
            BoundarySegment(
                s["side"],
                s["kind"],
                tuple(s["value"]) if isinstance(s["value"], (list, tuple)) else s["value"],
                s.get("start", 0.0),
                s.get("end", 1.0),
            )
            for s in d.get("segments", [])
        )
        srcs = tuple(SourceRegion(**r) for r in d.get("sources", []))
        return cls(segs, srcs)


def _side_slices(side: str, H: int, W: int, start: float, end: float) -> tuple[Any, Any]:
    if side in ("x_min", "x_max"):
        n = W
        a, b = round(start * (n - 1)), round(end * (n - 1))
        return (0 if side == "x_min" else H - 1), slice(a, b + 1)
    n = H
    a, b = round(start * (n - 1)), round(end * (n - 1))
    return slice(a, b + 1), (0 if side == "y_min" else W - 1)


_PRIORITY = {"neumann": 0, "robin": 1, "dirichlet": 2}  # at shared corners the stronger condition wins


def rasterize(spec: BoundarySpec, height: int, width: int | None = None) -> torch.Tensor:
    """Return ``(Cc, H, W)`` float32.

    Two segments on the *same side* that overlap with different kinds are an error. Where different sides meet
    (corners) the stronger condition wins: Dirichlet > Robin > Neumann (standard finite-volume convention, e.g. the
    Darcy solver's Dirichlet rows at ``x_min``/``x_max`` with no-flow walls).
    """
    W = width or height
    if height < 4 or W < 4:
        raise ConditioningError("grid must be at least 4x4")
    f = torch.zeros(len(COND_FIELD_CHANNELS), height, W)
    ix = {n: i for i, n in enumerate(COND_FIELD_CHANNELS)}
    same_side: dict[str, torch.Tensor] = {}
    for seg in spec.segments:  # same-side conflicts
        r, c = _side_slices(seg.side, height, W, seg.start, seg.end)
        occ = same_side.setdefault(seg.side, torch.full((height, W), -1, dtype=torch.long))
        prev = occ[r, c]
        k = KINDS.index(seg.kind)
        if bool(((prev >= 0) & (prev != k)).any()):
            raise ConditioningError(f"segments on {seg.side} overlap with different kinds")
        occ[r, c] = k
    channels = {
        "dirichlet": ("dirichlet_mask", "dirichlet_value"),
        "neumann": ("neumann_mask", "neumann_flux"),
        "robin": ("robin_mask", "robin_coeff", "robin_ambient"),
    }
    for seg in sorted(spec.segments, key=lambda s: _PRIORITY[s.kind]):
        r, c = _side_slices(seg.side, height, W, seg.start, seg.end)
        for other, names in channels.items():  # weaker kinds are cleared where a stronger one is written
            if _PRIORITY[other] < _PRIORITY[seg.kind]:
                for n in names:
                    f[ix[n], r, c] = 0.0
        vals = seg.value if isinstance(seg.value, tuple) else (seg.value,)
        f[ix[channels[seg.kind][0]], r, c] = 1.0
        for name, v in zip(channels[seg.kind][1:], vals):
            f[ix[name], r, c] = float(v)
    for src in spec.sources:
        i0, i1 = round(src.x0 * (height - 1)), round(src.x1 * (height - 1))
        j0, j1 = round(src.y0 * (W - 1)), round(src.y1 * (W - 1))
        f[ix["source"], i0 : i1 + 1, j0 : j1 + 1] += src.value
    return f


def cond_field_batch(
    specs: Sequence[BoundarySpec] | BoundarySpec, height: int, batch: int | None = None
) -> torch.Tensor:
    """Stack rasterised specs into ``(B, Cc, H, W)`` (a single spec is repeated ``batch`` times)."""
    if isinstance(specs, BoundarySpec):
        if not batch:
            raise ConditioningError("give batch when passing a single BoundarySpec")
        specs = [specs] * batch
    if not specs:
        raise ConditioningError("empty batch")
    return torch.stack([rasterize(s, height) for s in specs])


def describe(field_: torch.Tensor) -> dict[str, Any]:
    """Human-readable summary of a rasterised field ``(Cc, H, W)`` (used in logs and round-trip tests)."""
    if field_.ndim != 3 or field_.shape[0] != len(COND_FIELD_CHANNELS):
        raise ConditioningError(f"expected ({len(COND_FIELD_CHANNELS)}, H, W), got {tuple(field_.shape)}")
    ix = {n: i for i, n in enumerate(COND_FIELD_CHANNELS)}
    return {
        "dirichlet_cells": int(field_[ix["dirichlet_mask"]].sum()),
        "neumann_cells": int(field_[ix["neumann_mask"]].sum()),
        "robin_cells": int(field_[ix["robin_mask"]].sum()),
        "source_total": float(field_[ix["source"]].sum()),
    }
