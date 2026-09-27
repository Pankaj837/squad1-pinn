"""Typed exceptions. Every boundary in Squad 1 fails loudly with one of these (never silently)."""

from __future__ import annotations


class Squad1Error(ValueError):
    """Base class for all Squad 1 validation failures."""


class ContractError(Squad1Error):
    """A tensor / metadata object violates the shared interface contract (shape, dtype, channels, units)."""


class NormalizationError(Squad1Error):
    """Missing, malformed or inconsistent normalisation metadata."""


class NonFiniteError(Squad1Error):
    """NaN / Inf detected where finite values are required."""


class PhysicsError(Squad1Error):
    """Invalid physical state (e.g. non-positive permeability) or unsupported physics configuration."""


class ConditioningError(Squad1Error):
    """Conditioning inputs (chemistry / inferred parameters / targets / boundary spec) are invalid or inconsistent."""


class UnsafeExpressionError(Squad1Error):
    """A rule/PDE string contains syntax outside the allow-list."""


class ConvergenceError(Squad1Error):
    """An iterative solver failed to converge within its budget."""
