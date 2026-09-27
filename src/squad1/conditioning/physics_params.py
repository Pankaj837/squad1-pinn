"""Inferred physics parameters (output of the physics-inverse stage) — schema, validation and provider interface.

The inverse solver itself (BO-PINN etc.) lives outside this package; what Squad 1 needs from it is a *versioned,
named, unit-carrying* payload. Anything that satisfies :class:`ParameterProvider` can be plugged in.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from squad1.errors import ConditioningError, NonFiniteError


@dataclass(frozen=True)
class InferredParameters:
    names: tuple[str, ...]
    units: tuple[str, ...]
    values: tuple[float, ...]
    confidence: tuple[float, ...] | None = None  # in [0, 1] per parameter
    mask: tuple[bool, ...] | None = None  # False = value missing / not inferred
    version: str = "params_v1"
    source: str = ""

    def __post_init__(self) -> None:
        n = len(self.names)
        if n == 0 or len(set(self.names)) != n:
            raise ConditioningError("parameter names must be non-empty and unique")
        for label, seq in (
            ("units", self.units),
            ("values", self.values),
            ("confidence", self.confidence),
            ("mask", self.mask),
        ):
            if seq is not None and len(seq) != n:
                raise ConditioningError(f"{label} must have one entry per parameter ({n}), got {len(seq)}")
        mask = self.mask if self.mask is not None else (True,) * n
        for name, v, m in zip(self.names, self.values, mask):
            if m and not math.isfinite(v):
                raise NonFiniteError(f"parameter {name!r} is masked-in but not finite ({v})")
        if self.confidence is not None and any(not (0.0 <= c <= 1.0) for c in self.confidence):
            raise ConditioningError("confidence values must be in [0, 1]")

    @property
    def complete(self) -> bool:
        return self.mask is None or all(self.mask)

    def as_dict(self, require_complete: bool = True) -> dict[str, float]:
        """``{name: value}`` for the CondSpec encoder; masked-out values raise unless ``require_complete=False``."""
        if require_complete and not self.complete:
            missing = [n for n, m in zip(self.names, self.mask or ()) if not m]
            raise ConditioningError(f"inferred parameters incomplete, missing: {missing}")
        mask = self.mask if self.mask is not None else (True,) * len(self.names)
        return {n: v for n, v, m in zip(self.names, self.values, mask) if m}

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, float],
        units: Mapping[str, str],
        confidence: Mapping[str, float] | None = None,
        **kw: Any,
    ) -> InferredParameters:
        names = tuple(values)
        if set(units) != set(names):
            raise ConditioningError("units must be given for exactly the same parameters as values")
        return cls(
            names,
            tuple(units[n] for n in names),
            tuple(float(values[n]) for n in names),
            None if confidence is None else tuple(float(confidence[n]) for n in names),
            **kw,
        )


@runtime_checkable
class ParameterProvider(Protocol):
    """Anything that can turn a problem specification into :class:`InferredParameters`."""

    def infer(self, problem: Mapping[str, Any]) -> InferredParameters: ...


class StaticProvider:
    """Deterministic provider for tests / demos / hand-entered values."""

    def __init__(self, params: InferredParameters):
        self.params = params

    def infer(self, problem: Mapping[str, Any]) -> InferredParameters:
        return self.params


def check_against_spec_order(
    params: InferredParameters, expected_names: Sequence[str], expected_units: Sequence[str]
) -> None:
    """Refuse a payload whose names/units differ from what the conditioning spec was frozen with."""
    if tuple(params.names) != tuple(expected_names):
        raise ConditioningError(
            f"parameter names/order differ from spec: got {params.names}, expected {tuple(expected_names)}"
        )
    if tuple(params.units) != tuple(expected_units):
        raise ConditioningError(
            f"parameter units differ from spec: got {params.units}, expected {tuple(expected_units)}"
        )
