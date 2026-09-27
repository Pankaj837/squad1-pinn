"""Phase 3: the real pipeline against the synthetic data files in data/ (not in-memory fixtures).

Skips (not fails) if the data hasn't been generated yet, with a message pointing at the generator script.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "scripts"))

from squad1.conditioning import ChemistryConfig, ChemistryInverse, composition_vector
from squad1.contracts import CondSpec
from squad1.encoding import ConditioningToTokens
from squad1.geometry import load_material

pytestmark = pytest.mark.skipif(
    not (DATA / "chemistry" / "chemistry_train.parquet").exists(),
    reason="synthetic data not generated; run `python scripts/generate_synthetic_data.py` first",
)


@pytest.fixture(scope="module")
def ground_truth():
    from generate_synthetic_data import ELEMENTS_30, TARGET_PROPERTIES, true_properties

    gt = json.loads((DATA / "chemistry" / "_ground_truth_DO_NOT_USE_AS_FEATURES.json").read_text())
    return {
        "elements": ELEMENTS_30,
        "props": TARGET_PROPERTIES,
        "basis": np.array(gt["element_basis"]),
        "fn": true_properties,
    }


@pytest.fixture(scope="module")
def fitted_inverse(ground_truth):
    cfg = ChemistryConfig(
        elements=tuple(ground_truth["elements"]), target_properties=tuple(ground_truth["props"]), top_dataset_seeds=6
    )
    return ChemistryInverse(cfg).fit(DATA / "chemistry" / "chemistry_train.parquet")


def test_train_file_loads_and_cleans_edge_cases(fitted_inverse):
    # the generator injects ~1% NaN rows, ~1% unnormalised rows, ~1% near-duplicates into ~2828 raw train rows
    assert 2600 < fitted_inverse._df.shape[0] <= 2828
    assert np.allclose(fitted_inverse._df[list(fitted_inverse.cfg.elements)].sum(axis=1), 1.0, atol=1e-6)


def test_discovery_on_real_test_rows_is_conservative_not_optimistic(fitted_inverse, ground_truth):
    test_df = pd.read_parquet(DATA / "chemistry" / "chemistry_test.parquet").dropna()
    scales = {p: max(float(test_df[p].std()), 1e-9) for p in ground_truth["props"]}
    worse_or_equal = 0
    for _, row in test_df.sample(n=8, random_state=1).iterrows():
        targets = {p: float(row[p]) for p in ground_truth["props"]}
        best = fitted_inverse.discover(targets).best
        comp = np.array(composition_vector(best.composition, ground_truth["elements"]))
        true_pred = ground_truth["fn"](comp[None], ground_truth["basis"])[0]
        true_err = float(
            np.mean([((true_pred[i] - targets[p]) / scales[p]) ** 2 for i, p in enumerate(ground_truth["props"])])
        )
        # the reported error must not be dramatically more optimistic than reality (the defect found in
        # the original chemistry_inverse.py: reported 38x lower than the true error)
        if best.target_error >= true_err * 0.3:
            worse_or_equal += 1
    assert worse_or_equal >= 6  # allow a little sampling noise, but this must hold for most rows


def test_materials_lookup_filters_synthetic_invalid_rows():
    path = DATA / "materials" / "mp_elasticity_labels_synthetic.parquet"
    raw = pd.read_parquet(path)
    assert ((raw.E <= 0) | (raw.nu <= -1) | (raw.nu >= 0.5)).sum() > 0  # the generator does inject bad rows
    m = load_material(path)
    assert m["E_gpa"] > 0 and -1 < m["nu"] < 0.5
    m2 = load_material(path, target_E_gpa=200.0)
    assert abs(m2["E_gpa"] - 200.0) < abs(m["E_gpa"] - 200.0)


def test_discovered_composition_flows_through_condspec_and_encoder(fitted_inverse, ground_truth):
    train_df = pd.read_parquet(DATA / "chemistry" / "chemistry_train.parquet").dropna()
    target = {p: float(train_df[p].iloc[0]) for p in ground_truth["props"]}
    best = fitted_inverse.discover(target).best

    rng = np.random.default_rng(0)
    inferred_names = ("stub_a", "stub_b")
    spec = CondSpec(
        tuple(ground_truth["elements"]), inferred_names, ("u", "u"), tuple(target), ("W/mK", "kg/m3", "J/kgK", "K")
    )
    spec = spec.with_normalization(rng.normal(size=(len(train_df), 2)), train_df[ground_truth["props"]].to_numpy())
    cond_vec = spec.encode(best.composition, {"stub_a": 0.1, "stub_b": -0.4}, target).unsqueeze(0)
    assert cond_vec.shape == (1, 36) and torch.isfinite(cond_vec).all()

    torch.manual_seed(0)
    model = ConditioningToTokens(
        {"composition": 30, "inferred": 2, "target": 4},
        vocab_size=16,
        embed_dim=32,
        max_seq_len=6,
        num_heads=4,
        num_layers=1,
        ff_dim=32,
    )
    emb, ids, logits = model(cond_vec)
    assert emb.shape == (1, 6, 32) and ids.shape == (1, 6)
