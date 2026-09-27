import numpy as np
import pandas as pd
import pytest
import torch

from squad1.conditioning import (
    BoundarySegment,
    BoundarySpec,
    ChemistryConfig,
    ChemistryInverse,
    InferredParameters,
    SourceRegion,
    StaticProvider,
    check_against_spec_order,
    composition_vector,
    cond_field_batch,
    describe,
    rasterize,
)
from squad1.conditioning.boundary import COND_FIELD_CHANNELS
from squad1.contracts import CondSpec
from squad1.errors import ConditioningError, NonFiniteError

ELEMS = tuple(f"E{i:02d}" for i in range(30))
PROPS = ("thermal_conductivity", "density", "specific_heat", "melting_point")
_rng = np.random.default_rng(0)
PV = np.abs(_rng.normal(size=(30, 4))) * np.array([100, 3000, 500, 1000]) + np.array([1, 500, 100, 300])


def truth(c):
    return c @ PV


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    X = np.zeros((300, 30))
    for r in range(300):
        a = rng.choice(30, 5, replace=False)
        X[r, a] = rng.dirichlet(np.ones(5))
    df = pd.DataFrame(X, columns=ELEMS)
    Y = truth(X)
    for j, p in enumerate(PROPS):
        df[p] = Y[:, j]
    return df, X


def cfg(**kw):
    base = dict(
        elements=ELEMS,
        target_properties=PROPS,
        local_samples_per_seed=100,
        top_dataset_seeds=4,
        rf_estimators=100,
    )
    base.update(kw)
    return ChemistryConfig(**base)


@pytest.fixture(scope="module")
def fitted(data):
    return ChemistryInverse(cfg()).fit(data[0])


def targets_of(X, i):
    return dict(zip(PROPS, map(float, truth(X[i : i + 1])[0])))


# ------------------------------------------------------------------ chemistry
def test_discovery_returns_valid_honest_candidates(fitted, data):
    df, X = data
    res = fitted.discover(targets_of(X, 7))
    b = res.best
    comp = np.array(composition_vector(b.composition, ELEMS))
    assert comp.sum() == pytest.approx(1.0) and (comp >= 0).all()
    assert b.physically_valid and b.rank == 1 and b.surrogate_oob_rmse and b.surrogate_tree_std
    scale = df[list(PROPS)].std().to_numpy()
    true_err = float(np.mean(((truth(comp) - truth(X[7])) / scale) ** 2))
    assert true_err < 5 * b.target_error  # reported error is conservative, never 38x optimistic
    assert res.n_rows == 300 and len(res.candidates) <= 10


def test_unreachable_target_is_flagged(fitted, data):
    df, X = data
    t = targets_of(X, 0)
    t["thermal_conductivity"] *= 10
    assert not fitted.discover(t).best.target_satisfied


def test_reachable_target_flagged_satisfied(fitted, data):
    df, X = data
    res = fitted.discover(targets_of(X, 7))
    assert any(c.target_satisfied for c in res.candidates)
    assert not res.best.surrogate_reliable  # 300 rows in 30-D: the surrogate itself is weak, and says so


def test_deterministic_and_saved_json(fitted, data, tmp_path):
    t = targets_of(data[1], 5)
    a, b = fitted.discover(t), fitted.discover(t)
    assert a.best.composition == b.best.composition
    p = tmp_path / "o" / "r.json"
    a.save(p)
    assert '"schema_version": "chemistry_inverse_v2"' in p.read_text()


def test_config_and_input_validation(data, tmp_path):
    with pytest.raises(ConditioningError):
        ChemistryConfig(elements=("A", "A"), target_properties=("p",))
    with pytest.raises(ConditioningError):
        ChemistryConfig(elements=("A",), target_properties=())
    with pytest.raises(ConditioningError):
        ChemistryConfig(elements=("A",), target_properties=("p",), proposal_chunk=0)
    with pytest.raises(ConditioningError):
        ChemistryInverse(cfg()).discover({})
    inv = ChemistryInverse(cfg(rf_estimators=20))
    with pytest.raises(FileNotFoundError):
        inv.fit(tmp_path / "missing.csv")
    (tmp_path / "x.json").write_text("{}")
    with pytest.raises(ConditioningError):
        inv.fit(tmp_path / "x.json")
    with pytest.raises(ConditioningError):
        inv.fit(data[0].drop(columns=[PROPS[0]]))
    tiny = data[0].head(3)
    with pytest.raises(ConditioningError):
        inv.fit(tiny)


def test_targets_validation(fitted, data):
    t = targets_of(data[1], 1)
    with pytest.raises(ConditioningError):
        fitted.discover({k: v for k, v in t.items() if k != "density"})
    with pytest.raises(ConditioningError):
        fitted.discover({**t, "extra": 1.0})
    with pytest.raises(NonFiniteError):
        fitted.discover({**t, "density": float("nan")})


def test_dataset_cleaning_csv_parquet_and_bad_rows(data, tmp_path):
    df = data[0].copy()
    df.loc[0, ELEMS[0]] = -1.0  # negative row dropped
    df.loc[1, PROPS[1]] = np.nan  # NaN row dropped
    df.loc[2, list(ELEMS)] = 0.0  # all-zero row dropped
    df.loc[3, ELEMS[1]] *= 2.0  # unnormalised row renormalised
    inv = ChemistryInverse(cfg(rf_estimators=10))
    clean = inv.load_dataset(df)
    assert len(clean) == 297 and np.allclose(clean[list(ELEMS)].sum(axis=1), 1.0)
    df.to_csv(tmp_path / "d.csv", index=False)
    df.to_parquet(tmp_path / "d.parquet")
    assert len(inv.load_dataset(tmp_path / "d.csv")) == 297 == len(inv.load_dataset(tmp_path / "d.parquet"))


def test_physical_limits_and_extra_checks(data):
    df, X = data
    inv = ChemistryInverse(
        cfg(physical_limits={"density": {"min": 1e9}}, operating_temperature_k=1.0),
        extra_checks=lambda comp, pred: {"custom": bool(comp.max() < 2.0)},
    ).fit(df)
    res = inv.discover(targets_of(X, 2))
    assert not res.best.physically_valid
    assert "density_min" in res.best.physical_checks and "custom" in res.best.physical_checks
    assert "melting_above_operating_temperature" in res.best.physical_checks


def test_composition_vector_rejects_unknown():
    assert composition_vector({"E01": 1.0}, ELEMS)[1] == 1.0
    with pytest.raises(ConditioningError):
        composition_vector({"nope": 1.0}, ELEMS)


# -------------------------------------------------------------- physics params
def test_inferred_parameters_schema():
    p = InferredParameters.from_mapping(
        {"a": 1.0, "b": 2.0}, {"a": "K", "b": "m"}, {"a": 0.9, "b": 0.5}, source="unit-test"
    )
    assert p.complete and p.as_dict() == {"a": 1.0, "b": 2.0}
    q = InferredParameters(("a", "b"), ("K", "m"), (1.0, float("nan")), mask=(True, False))
    assert not q.complete and q.as_dict(require_complete=False) == {"a": 1.0}
    with pytest.raises(ConditioningError):
        q.as_dict()
    with pytest.raises(NonFiniteError):
        InferredParameters(("a",), ("K",), (float("nan"),))
    with pytest.raises(ConditioningError):
        InferredParameters(("a", "a"), ("K", "K"), (1.0, 2.0))
    with pytest.raises(ConditioningError):
        InferredParameters(("a",), ("K", "m"), (1.0,))
    with pytest.raises(ConditioningError):
        InferredParameters(("a",), ("K",), (1.0,), confidence=(1.5,))
    with pytest.raises(ConditioningError):
        InferredParameters.from_mapping({"a": 1.0}, {"b": "K"})
    check_against_spec_order(p, ("a", "b"), ("K", "m"))
    with pytest.raises(ConditioningError):
        check_against_spec_order(p, ("b", "a"), ("m", "K"))
    with pytest.raises(ConditioningError):
        check_against_spec_order(p, ("a", "b"), ("K", "cm"))
    assert StaticProvider(p).infer({}) is p


def test_params_feed_condspec_end_to_end(fitted, data):
    p = InferredParameters.from_mapping({"t0": 300.0, "t1": 5.0}, {"t0": "K", "t1": "W/mK"})
    res = fitted.discover(targets_of(data[1], 4))
    spec = CondSpec(ELEMS, ("t0", "t1"), ("K", "W/mK"), ("T_op",), ("K",))
    v = spec.encode(res.best.composition, p.as_dict(), {"T_op": 500.0})
    assert v.shape == (33,) and torch.isfinite(v).all()


# --------------------------------------------------------------------- boundary
def test_rasterize_sides_and_channels():
    spec = BoundarySpec(
        (
            BoundarySegment("x_min", "dirichlet", 1.0),
            BoundarySegment("x_max", "dirichlet", 0.0),
            BoundarySegment("y_min", "neumann", 0.0),
            BoundarySegment("y_max", "robin", (5.0, 300.0), 0.25, 0.75),
        ),
        (SourceRegion(0.4, 0.6, 0.4, 0.6, 2.0),),
    )
    f = rasterize(spec, 16)
    ix = {n: i for i, n in enumerate(COND_FIELD_CHANNELS)}
    assert f.shape == (8, 16, 16) and f.dtype == torch.float32
    assert f[ix["dirichlet_mask"], 0].sum() == 16 and f[ix["dirichlet_value"], 0].eq(1.0).all()
    assert f[ix["dirichlet_value"], 15].eq(0.0).all() and f[ix["dirichlet_mask"], 15].eq(1.0).all()
    assert f[ix["neumann_mask"], :, 0].sum() == 14  # both corners belong to the Dirichlet sides
    rob = f[ix["robin_mask"], :, 15]
    assert (
        0 < rob.sum() < 16 and f[ix["robin_coeff"], :, 15].max() == 5.0 and f[ix["robin_ambient"], :, 15].max() == 300.0
    )
    assert f[ix["source"]].max() == 2.0 and f[ix["source"]].sum() > 0
    d = describe(f)
    assert d["dirichlet_cells"] == 32 and d["source_total"] > 0


def test_boundary_validation_and_roundtrip():
    with pytest.raises(ConditioningError):
        BoundarySegment("left", "dirichlet", 1.0)
    with pytest.raises(ConditioningError):
        BoundarySegment("x_min", "wall", 1.0)
    with pytest.raises(ConditioningError):
        BoundarySegment("x_min", "dirichlet", (1.0, 2.0))
    with pytest.raises(ConditioningError):
        BoundarySegment("x_min", "robin", 1.0)
    with pytest.raises(ConditioningError):
        BoundarySegment("x_min", "robin", (-1.0, 2.0))
    with pytest.raises(ConditioningError):
        BoundarySegment("x_min", "dirichlet", float("inf"))
    with pytest.raises(ConditioningError):
        BoundarySegment("x_min", "dirichlet", 1.0, 0.6, 0.4)
    with pytest.raises(ConditioningError):
        SourceRegion(0.5, 0.4, 0, 1, 1.0)
    with pytest.raises(ConditioningError):
        rasterize(BoundarySpec(), 2)
    clash = BoundarySpec(
        (BoundarySegment("x_min", "dirichlet", 1.0), BoundarySegment("x_min", "neumann", 0.0, 0.5, 1.0))
    )
    with pytest.raises(ConditioningError):  # same side, different kinds, overlapping
        rasterize(clash, 8)
    corner = rasterize(
        BoundarySpec((BoundarySegment("y_min", "neumann", 0.0), BoundarySegment("x_min", "dirichlet", 1.0))),
        8,
    )
    assert corner[0, 0, 0] == 1.0 and corner[2, 0, 0] == 0.0 and corner[2, 1, 0] == 1.0  # Dirichlet wins at the corner
    spec = BoundarySpec(
        (BoundarySegment("x_min", "robin", (2.0, 3.0), 0.1, 0.9),), (SourceRegion(0.1, 0.2, 0.1, 0.2, 1.0),)
    )
    assert BoundarySpec.from_dict(spec.to_dict()).segments == spec.segments
    assert torch.equal(rasterize(BoundarySpec.from_dict(spec.to_dict()), 8), rasterize(spec, 8))


def test_cond_field_batch():
    spec = BoundarySpec((BoundarySegment("x_min", "dirichlet", 1.0),))
    assert cond_field_batch(spec, 8, 3).shape == (3, 8, 8, 8)
    assert cond_field_batch([spec, BoundarySpec()], 8).shape == (2, 8, 8, 8)
    with pytest.raises(ConditioningError):
        cond_field_batch(spec, 8)
    with pytest.raises(ConditioningError):
        cond_field_batch([], 8)
    with pytest.raises(ConditioningError):
        describe(torch.zeros(3, 4, 4))
