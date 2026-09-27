"""Chemistry / material inverse discovery (dataset-driven baseline) — library form, no module-level globals.

Pipeline: load composition/property table -> fit multi-output Random-Forest surrogate (with out-of-bag error) ->
rank dataset rows by closeness to the target properties -> refine each seed with batched Dirichlet proposals on the
composition simplex -> physical sanity checks -> honest error reporting.

Honesty features (compared with the first baseline):
  * the score used for search *and* the reported ``target_error`` include the surrogate's out-of-bag variance, so the
    reported error is not the optimistic point estimate of a model that was just optimised against;
  * ``target_satisfied`` states explicitly whether every property is within ``target_tolerance_std`` dataset-stds;
  * ``surrogate_oob_rmse`` / ``surrogate_tree_std`` expose the model's own uncertainty;
  * the search space is the neighbourhood of dataset rows — a Random Forest cannot extrapolate, so targets outside the
    data hull come back with ``target_satisfied=False`` (never silently).
Thermodynamic stability (formation energy / hull distance) is **not** computed here; pass ``extra_checks`` to add it.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from squad1.errors import ConditioningError, NonFiniteError

try:  # optional extras: pip install squad1[chem]
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
except ImportError as exc:  # pragma: no cover
    pd = None
    _IMPORT_ERROR: ImportError | None = exc
else:
    _IMPORT_ERROR = None

ExtraChecks = Callable[[np.ndarray, Mapping[str, float]], Mapping[str, bool]]


def _require_extras() -> None:
    if _IMPORT_ERROR is not None:  # pragma: no cover
        raise ImportError(
            "chemistry module needs pandas and scikit-learn: pip install 'squad1[chem]'"
        ) from _IMPORT_ERROR


@dataclass
class ChemistryConfig:
    elements: tuple[str, ...]  # frozen composition vocabulary / order (e.g. the 30 generator elements)
    target_properties: tuple[str, ...]
    physical_limits: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    operating_temperature_k: float | None = None  # if set, "melting_point" must exceed it
    top_dataset_seeds: int = 10
    local_samples_per_seed: int = 500
    proposal_chunk: int = 50
    top_final_candidates: int = 10
    dirichlet_concentration: float = 80.0
    target_tolerance_std: float = 0.10
    rf_estimators: int = 300
    n_jobs: int = 1
    seed: int = 42
    min_rows: int = 5
    max_oob_std: float = 0.35  # surrogate is 'reliable' when oob RMSE <= this many dataset stds for every property

    def __post_init__(self) -> None:
        if not self.elements or len(set(self.elements)) != len(self.elements):
            raise ConditioningError("elements must be non-empty and unique")
        if not self.target_properties or len(set(self.target_properties)) != len(self.target_properties):
            raise ConditioningError("target_properties must be non-empty and unique")
        if self.proposal_chunk < 1 or self.local_samples_per_seed < 1 or self.top_dataset_seeds < 1:
            raise ConditioningError("search sizes must be >= 1")


@dataclass
class Candidate:
    rank: int
    composition: dict[str, float]
    predicted_properties: dict[str, float]
    target_error: float
    physical_checks: dict[str, bool]
    physically_valid: bool
    source: str
    target_satisfied: bool = False
    surrogate_reliable: bool = False
    surrogate_oob_rmse: dict[str, float] | None = None
    surrogate_tree_std: dict[str, float] | None = None


@dataclass
class ChemistryResult:
    candidates: list[Candidate]
    n_rows: int
    surrogate_oob_rmse: dict[str, float]

    @property
    def best(self) -> Candidate:
        return self.candidates[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "chemistry_inverse_v2",
            "n_rows": self.n_rows,
            "surrogate_oob_rmse": self.surrogate_oob_rmse,
            "candidates": [asdict(c) for c in self.candidates],
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class ChemistryInverse:
    def __init__(self, config: ChemistryConfig, extra_checks: ExtraChecks | None = None):
        _require_extras()
        self.cfg = config
        self.extra_checks = extra_checks
        self._model: Any = None
        self._scales: dict[str, float] = {}
        self._oob: dict[str, float] = {}
        self._df: Any = None

    # ------------------------------------------------------------------ data
    def load_dataset(self, source: str | Path | Any) -> Any:
        """Read + clean a table (path to .csv/.parquet or a DataFrame). Returns the cleaned frame."""
        cfg = self.cfg
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"dataset not found: {path}")
            suffix = path.suffix.lower()
            if suffix == ".parquet":
                df = pd.read_parquet(path)
            elif suffix in {".csv", ".txt"}:
                df = pd.read_csv(path)
            else:
                raise ConditioningError("dataset must be .parquet, .csv or .txt")
        else:
            df = source.copy()
        required = [*cfg.elements, *cfg.target_properties]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ConditioningError(f"dataset missing columns: {missing}")
        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=required)
        X = df[list(cfg.elements)].to_numpy(dtype=float)
        keep = np.isfinite(X).all(axis=1) & (X >= 0).all(axis=1) & (X.sum(axis=1) > 0)
        df = df.loc[keep].copy()
        X = df[list(cfg.elements)].to_numpy(dtype=float)
        df.loc[:, list(cfg.elements)] = X / X.sum(axis=1, keepdims=True)
        if len(df) < cfg.min_rows:
            raise ConditioningError(f"only {len(df)} usable rows (< {cfg.min_rows})")
        if len(df) < 20:
            warnings.warn("fewer than 20 usable dataset rows; results may be unstable", stacklevel=2)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------- fit
    def fit(self, source: str | Path | Any) -> ChemistryInverse:
        cfg = self.cfg
        df = self.load_dataset(source)
        X = df[list(cfg.elements)].to_numpy(dtype=float)
        y = df[list(cfg.target_properties)].to_numpy(dtype=float)
        self._model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=cfg.rf_estimators,
                random_state=cfg.seed,
                n_jobs=cfg.n_jobs,
                max_features="sqrt",
                min_samples_leaf=2,
                oob_score=True,
            )
        )
        self._model.fit(X, y)
        self._oob = {
            p: float(np.sqrt(np.nanmean((self._model.estimators_[j].oob_prediction_ - y[:, j]) ** 2)))
            for j, p in enumerate(cfg.target_properties)
        }
        self._scales = {p: max(float(df[p].std()), 1e-12) for p in cfg.target_properties}
        self._df = df
        return self

    # ------------------------------------------------------------ internals
    def _predict(self, comps: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(np.atleast_2d(comps)), dtype=float)

    def _tree_std(self, comp: np.ndarray) -> dict[str, float]:
        X = comp.reshape(1, -1)
        return {
            p: float(np.std([t.predict(X)[0] for t in self._model.estimators_[j].estimators_]))
            for j, p in enumerate(self.cfg.target_properties)
        }

    def _error(self, pred: Mapping[str, float], targets: Mapping[str, float]) -> float:
        return float(np.mean([((pred[p] - targets[p]) / self._scales[p]) ** 2 for p in self.cfg.target_properties]))

    def _checks(self, comp: np.ndarray, pred: Mapping[str, float]) -> tuple[bool, dict[str, bool]]:
        cfg = self.cfg
        checks = {
            "composition_sum": bool(np.isclose(comp.sum(), 1.0, atol=1e-5)),
            "nonnegative_composition": bool((comp >= -1e-10).all()),
        }
        for prop, lim in cfg.physical_limits.items():
            if prop not in pred:
                checks[f"{prop}_present"] = False
                continue
            if "min" in lim:
                checks[f"{prop}_min"] = pred[prop] >= lim["min"]
            if "max" in lim:
                checks[f"{prop}_max"] = pred[prop] <= lim["max"]
        for prop in ("density", "specific_heat", "thermal_conductivity"):
            if prop in pred:
                checks[f"{prop}_positive"] = pred[prop] > 0
        if cfg.operating_temperature_k is not None and "melting_point" in pred:
            checks["melting_above_operating_temperature"] = pred["melting_point"] > cfg.operating_temperature_k
        if self.extra_checks is not None:
            checks.update({str(k): bool(v) for k, v in self.extra_checks(comp, pred).items()})
        return bool(all(checks.values())), checks

    # ------------------------------------------------------------- discover
    def discover(self, targets: Mapping[str, float]) -> ChemistryResult:
        if self._model is None:
            raise ConditioningError("call fit() first")
        cfg = self.cfg
        missing = [p for p in cfg.target_properties if p not in targets]
        extra = [p for p in targets if p not in cfg.target_properties]
        if missing or extra:
            raise ConditioningError(f"targets mismatch: missing={missing} unexpected={extra}")
        if not all(math.isfinite(float(v)) for v in targets.values()):
            raise NonFiniteError(f"target values must be finite: {dict(targets)}")
        targets = {p: float(targets[p]) for p in cfg.target_properties}
        df = self._df
        rng = np.random.default_rng(cfg.seed)
        penalty = float(np.mean([(self._oob[p] / self._scales[p]) ** 2 for p in cfg.target_properties]))

        # seeds: dataset rows whose *measured* properties are closest to the targets
        y_all = df[list(cfg.target_properties)].to_numpy(dtype=float)
        t_vec = np.array([targets[p] for p in cfg.target_properties])
        sc_vec = np.array([self._scales[p] for p in cfg.target_properties])
        seed_err = (((y_all - t_vec) / sc_vec) ** 2).mean(axis=1)
        order = np.argsort(seed_err)[: cfg.top_dataset_seeds]

        def evaluate(
            batch: np.ndarray,
        ) -> list[tuple[np.ndarray, float, dict[str, float], bool, dict[str, bool]]]:
            preds = self._predict(batch)
            out = []
            for comp, row in zip(batch, preds):
                pred = dict(zip(cfg.target_properties, map(float, row)))
                valid, checks = self._checks(comp, pred)
                out.append((comp, self._error(pred, targets) + penalty, pred, valid, checks))
            return out

        def score(t: tuple[Any, ...]) -> float:
            return t[1] if t[3] else t[1] + 1e6

        reliable = all(self._oob[p] / self._scales[p] <= cfg.max_oob_std for p in cfg.target_properties)
        candidates: list[Candidate] = []
        n_chunks = max(1, cfg.local_samples_per_seed // cfg.proposal_chunk)
        for idx in order:
            seed = df.loc[idx, list(cfg.elements)].to_numpy(dtype=float)
            best = evaluate(seed[None, :])[0]
            for _ in range(n_chunks):
                alpha = np.maximum(best[0] * cfg.dirichlet_concentration, 1e-3)
                cand = min(evaluate(rng.dirichlet(alpha, size=cfg.proposal_chunk)), key=score)
                if score(cand) < score(best):
                    best = cand
            comp = np.where(best[0] > 1e-8, best[0], 0.0)
            comp = comp / comp.sum()
            pred = best[2]
            tol_ok = all(
                abs(pred[p] - targets[p]) / self._scales[p] <= cfg.target_tolerance_std for p in cfg.target_properties
            )
            candidates.append(
                Candidate(
                    rank=0,
                    composition={e: float(v) for e, v in zip(cfg.elements, comp) if v > 0},
                    predicted_properties=pred,
                    target_error=float(best[1]),
                    physical_checks=best[4],
                    physically_valid=bool(best[3]),
                    source="dataset_seed_plus_local_inverse_search",
                    target_satisfied=bool(tol_ok and best[3]),
                    surrogate_reliable=reliable,
                    surrogate_oob_rmse=dict(self._oob),
                    surrogate_tree_std=self._tree_std(comp),
                )
            )
        candidates.sort(key=lambda c: (not c.physically_valid, not c.target_satisfied, c.target_error))
        candidates = candidates[: cfg.top_final_candidates]
        for i, c in enumerate(candidates, 1):
            c.rank = i
        return ChemistryResult(candidates, len(df), dict(self._oob))


def composition_vector(composition: Mapping[str, float], elements: Sequence[str]) -> list[float]:
    """Dense composition vector in the frozen element order (raises on unknown elements)."""
    unknown = set(composition) - set(elements)
    if unknown:
        raise ConditioningError(f"unknown elements in composition: {sorted(unknown)}")
    return [float(composition.get(e, 0.0)) for e in elements]
