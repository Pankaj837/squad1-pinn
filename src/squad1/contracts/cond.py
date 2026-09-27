"""Conditioning-vector contract: named schema first, tensor derived.

Layout of the tensor (float32, ``(B, Dv)``): ``[composition | inferred parameters | target conditions]``.
``Dv`` is always computed from the spec (never hard-coded); the spec travels with every result so a vector can
be decoded again (round-trip tested).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from squad1.errors import ConditioningError, NonFiniteError

MAX_RAW_ABS = (
    1.0e3  # un-standardised inferred/target values above this are refused (encoder expects standardised inputs)
)
COMPOSITION_SUM_TOL = 1.0e-3


@dataclass(frozen=True)
class CondSpec:
    composition_order: tuple[str, ...]
    inferred_order: tuple[str, ...]
    inferred_units: tuple[str, ...]
    target_order: tuple[str, ...]
    target_units: tuple[str, ...]
    mean: tuple[float, ...] | None = None  # length Dv (composition entries should stay 0)
    std: tuple[float, ...] | None = None  # length Dv (composition entries should stay 1)
    version: str = "cond_v1"

    def __post_init__(self) -> None:
        for label, order in (
            ("composition_order", self.composition_order),
            ("inferred_order", self.inferred_order),
            ("target_order", self.target_order),
        ):
            if len(set(order)) != len(order):
                raise ConditioningError(f"{label} contains duplicates: {order}")
        if not self.composition_order:
            raise ConditioningError("composition_order must not be empty")
        if len(self.inferred_units) != len(self.inferred_order):
            raise ConditioningError("inferred_units must match inferred_order in length")
        if len(self.target_units) != len(self.target_order):
            raise ConditioningError("target_units must match target_order in length")
        if (self.mean is None) != (self.std is None):
            raise ConditioningError("mean and std must be given together")
        if self.mean is not None and self.std is not None:
            if len(self.mean) != self.dv or len(self.std) != self.dv:
                raise ConditioningError(f"mean/std must have length Dv={self.dv}")
            if not all(math.isfinite(v) for v in (*self.mean, *self.std)):
                raise NonFiniteError("mean/std must be finite")
            if any(s <= 0 for s in self.std):
                raise ConditioningError("std must be strictly positive")

    # ------------------------------------------------------------------ sizes
    @property
    def n_composition(self) -> int:
        return len(self.composition_order)

    @property
    def n_inferred(self) -> int:
        return len(self.inferred_order)

    @property
    def n_target(self) -> int:
        return len(self.target_order)

    @property
    def dv(self) -> int:
        return self.n_composition + self.n_inferred + self.n_target

    @property
    def groups(self) -> dict[str, int]:
        return {"composition": self.n_composition, "inferred": self.n_inferred, "target": self.n_target}

    @property
    def is_standardised(self) -> bool:
        return self.mean is not None

    # -------------------------------------------------------------- (de)coding
    def _stats(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mean is None or self.std is None:
            return np.zeros(self.dv), np.ones(self.dv)
        return np.asarray(self.mean, dtype=np.float64), np.asarray(self.std, dtype=np.float64)

    def encode(
        self,
        composition: Mapping[str, float],
        inferred: Mapping[str, float],
        target: Mapping[str, float],
    ) -> torch.Tensor:
        """Named dicts -> standardised float32 vector ``(Dv,)``. Raises on anything suspicious."""
        unknown = set(composition) - set(self.composition_order)
        if unknown:
            raise ConditioningError(f"unknown composition elements: {sorted(unknown)}")
        comp = np.array([float(composition.get(e, 0.0)) for e in self.composition_order], dtype=np.float64)
        if not np.isfinite(comp).all():
            raise NonFiniteError("composition contains NaN/Inf")
        if (comp < -1e-9).any():
            raise ConditioningError("composition fractions must be non-negative")
        if not math.isclose(float(comp.sum()), 1.0, abs_tol=COMPOSITION_SUM_TOL):
            raise ConditioningError(f"composition fractions must sum to 1 (got {comp.sum():.6f})")

        parts = [comp]
        for label, order, values in (
            ("inferred parameter", self.inferred_order, inferred),
            ("target condition", self.target_order, target),
        ):
            missing = [n for n in order if n not in values]
            extra = [n for n in values if n not in order]
            if missing or extra:
                raise ConditioningError(f"{label}s mismatch: missing={missing} unexpected={extra}")
            arr = np.array([float(values[n]) for n in order], dtype=np.float64)
            if not np.isfinite(arr).all():
                raise NonFiniteError(f"{label}s contain NaN/Inf: {dict(zip(order, arr.tolist()))}")
            if not self.is_standardised and (np.abs(arr) > MAX_RAW_ABS).any():
                bad = {n: float(v) for n, v in zip(order, arr) if abs(v) > MAX_RAW_ABS}
                raise ConditioningError(
                    f"{label}s look un-normalised (|value| > {MAX_RAW_ABS:g}) but the spec has no mean/std: {bad}"
                )
            parts.append(arr)
        raw = np.concatenate(parts)
        mean, std = self._stats()
        return torch.from_numpy(((raw - mean) / std).astype(np.float32))

    def encode_batch(self, items: Sequence[Mapping[str, Mapping[str, float]]]) -> torch.Tensor:
        """``items[i] = {"composition": {...}, "inferred": {...}, "target": {...}}`` -> ``(B, Dv)``."""
        if not items:
            raise ConditioningError("empty batch")
        return torch.stack([self.encode(it["composition"], it["inferred"], it["target"]) for it in items])

    def decode(self, vec: torch.Tensor) -> dict[str, dict[str, float]]:
        """Inverse of :meth:`encode` (un-standardises). ``vec`` is ``(Dv,)``."""
        if vec.ndim != 1 or vec.shape[0] != self.dv:
            raise ConditioningError(f"expected vector of length {self.dv}, got shape {tuple(vec.shape)}")
        mean, std = self._stats()
        raw = vec.detach().cpu().double().numpy() * std + mean
        a, b = self.n_composition, self.n_composition + self.n_inferred
        return {
            "composition": dict(zip(self.composition_order, raw[:a].tolist())),
            "inferred": dict(zip(self.inferred_order, raw[a:b].tolist())),
            "target": dict(zip(self.target_order, raw[b:].tolist())),
        }

    def check_tensor(self, cond_vec: torch.Tensor) -> torch.Tensor:
        if cond_vec.ndim != 2 or cond_vec.shape[1] != self.dv:
            raise ConditioningError(f"cond_vec must be (B, {self.dv}); got {tuple(cond_vec.shape)}")
        if not torch.is_floating_point(cond_vec):
            raise ConditioningError(f"cond_vec must be floating point, got {cond_vec.dtype}")
        if not torch.isfinite(cond_vec).all():
            raise NonFiniteError("cond_vec contains NaN/Inf")
        return cond_vec

    def with_normalization(self, inferred_samples: np.ndarray, target_samples: np.ndarray) -> CondSpec:
        """Fit per-entry mean/std of inferred + target columns (composition stays identity)."""
        inf = np.asarray(inferred_samples, dtype=np.float64)
        tgt = np.asarray(target_samples, dtype=np.float64)
        if inf.ndim != 2 or inf.shape[1] != self.n_inferred or tgt.ndim != 2 or tgt.shape[1] != self.n_target:
            raise ConditioningError("sample arrays must be (N, n_inferred) and (N, n_target)")
        if len(inf) < 2 or len(tgt) < 2:
            raise ConditioningError("need at least 2 samples to fit a standardisation")
        mean = np.concatenate([np.zeros(self.n_composition), inf.mean(0), tgt.mean(0)])
        std = np.concatenate(
            [np.ones(self.n_composition), np.maximum(inf.std(0), 1e-12), np.maximum(tgt.std(0), 1e-12)]
        )
        return CondSpec(
            self.composition_order,
            self.inferred_order,
            self.inferred_units,
            self.target_order,
            self.target_units,
            tuple(mean.tolist()),
            tuple(std.tolist()),
            self.version,
        )

    # -------------------------------------------------------------------- json
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "composition_order": list(self.composition_order),
            "inferred_order": list(self.inferred_order),
            "inferred_units": list(self.inferred_units),
            "target_order": list(self.target_order),
            "target_units": list(self.target_units),
            "mean": None if self.mean is None else list(self.mean),
            "std": None if self.std is None else list(self.std),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CondSpec:
        return cls(
            tuple(d["composition_order"]),
            tuple(d["inferred_order"]),
            tuple(d["inferred_units"]),
            tuple(d["target_order"]),
            tuple(d["target_units"]),
            None if d.get("mean") is None else tuple(d["mean"]),
            None if d.get("std") is None else tuple(d["std"]),
            d.get("version", "cond_v1"),
        )
