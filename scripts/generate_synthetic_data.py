"""Generate synthetic chemistry/materials data to exercise `squad1.conditioning` end-to-end.

WHY SYNTHETIC: no real chemistry dataset or Materials-Project elasticity export was provided to
Squad 1. `ChemistryInverse.fit()` and `geometry.load_material()` both need a real file on disk to
run at all — without one, those two code paths are untested by construction. This script builds a
disk-realistic stand-in so the pipeline can be exercised end-to-end (see scripts/verify_end_to_end.py
and tests/integration/test_synthetic_data_pipeline.py), and documents exactly how it was made so it
is never mistaken for real data.

HOW THE CHEMISTRY GROUND TRUTH WORKS (read before trusting any number that comes out of it)
    1. A frozen 30-element vocabulary is picked (ceramics/alloy-relevant; see ELEMENTS_30 below —
       Squad 1's real vocabulary from Swaraj/the generator team is still open, see docs/DECISIONS.md D-5).
    2. Each element gets a fixed, seeded "base property" vector (rng seed SEED=20260927) for the four
       target properties. This is an arbitrary deterministic assignment, NOT physical/DFT/experimental data.
    3. A composition's *true* property is: rule-of-mixtures (composition-weighted average of element bases)
       plus a small nonlinear interaction term (so it is not perfectly linear, like real alloys) — see
       `true_properties()`. This function is exact and noise-free; it is the "ground truth" used only for
       *verification* (comparing a surrogate's reported error against the true error) and is written to
       `data/chemistry/_ground_truth_DO_NOT_USE_AS_FEATURES.json` — the model under test never sees it.
    4. The saved train/val/test files add heteroscedastic Gaussian measurement noise (~3% of scale + a
       floor) on top of the true properties, mimicking real (noisy) measurements.

EDGE CASES INJECTED (to exercise `ChemistryInverse.load_dataset`'s cleaning step, not just the happy path):
    - ~1% of rows have a NaN in a target-property column (must be dropped).
    - ~1% of rows have composition fractions that do not sum to 1 before cleaning (must be renormalised).
    - ~1% of rows are near-duplicates (same active elements, slightly different fractions).
    - a handful of rows are property outliers (valid but extreme) to check the surrogate does not choke.

MATERIALS TABLE: synthetic (formula_pretty, E, nu, density_g_cm3, material_id) rows in the schema
`squad1.geometry.load_material()` expects, replacing the missing `mp_elasticity_labels.parquet`
referenced (but never delivered) in the original Topology report. ~3% of rows are deliberately invalid
(E <= 0 or nu outside (-1, 0.5)) to exercise load_material's own filtering.

Regenerate with:  python scripts/generate_synthetic_data.py [--out data] [--seed 20260927]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEED = 20260927

ELEMENTS_30: list[str] = [
    "Al", "Si", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zn", "Zr", "Nb", "Mo", "Hf", "Ta", "W", "Re", "Ru", "Rh",
    "Pd", "Ag", "Sn", "Y", "La", "Ce", "B", "C", "N", "O",
]  # fmt: skip
assert len(ELEMENTS_30) == 30

TARGET_PROPERTIES: list[str] = ["thermal_conductivity", "density", "specific_heat", "melting_point"]
# realistic order-of-magnitude ranges used to scale the seeded per-element base values
PROP_RANGE = {
    "thermal_conductivity": (1.0, 420.0),  # W/m/K  (aerogel-ish .. silver-ish)
    "density": (1800.0, 19300.0),  # kg/m^3 (light ceramic .. tungsten)
    "specific_heat": (120.0, 900.0),  # J/kg/K
    "melting_point": (500.0, 4200.0),  # K (low-melting alloy .. HfC-ish)
}


def _element_base_properties(rng: np.random.Generator) -> np.ndarray:
    """(30, 4) seeded, deterministic per-element property basis (log-uniform within PROP_RANGE)."""
    out = np.zeros((len(ELEMENTS_30), len(TARGET_PROPERTIES)))
    for j, prop in enumerate(TARGET_PROPERTIES):
        lo, hi = PROP_RANGE[prop]
        out[:, j] = np.exp(rng.uniform(np.log(lo), np.log(hi), size=len(ELEMENTS_30)))
    return out


def true_properties(composition: np.ndarray, element_basis: np.ndarray, interaction: float = 0.12) -> np.ndarray:
    """Exact, noise-free ground truth for a batch of compositions ``(N, 30) -> (N, 4)``.

    rule-of-mixtures (linear) + a nonlinear "mixing entropy"-like interaction term scaled by how many
    elements are active and how evenly they are mixed (real alloys/ceramics are not perfectly linear).
    """
    composition = np.atleast_2d(composition)
    linear = composition @ element_basis
    n_active = (composition > 1e-9).sum(axis=1, keepdims=True).clip(min=1)
    evenness = 1.0 - (composition**2).sum(axis=1, keepdims=True)  # 0 = pure element, -> 1 = evenly mixed
    nonlinear = interaction * evenness * np.log1p(n_active) * linear
    return linear + nonlinear


def _sample_composition(rng: np.random.Generator, n: int, min_active: int = 2, max_active: int = 6) -> np.ndarray:
    out = np.zeros((n, len(ELEMENTS_30)))
    for i in range(n):
        k = rng.integers(min_active, max_active + 1)
        idx = rng.choice(len(ELEMENTS_30), size=k, replace=False)
        out[i, idx] = rng.dirichlet(np.full(k, 2.0))
    return out


def make_chemistry_dataset(rng: np.random.Generator, n: int, element_basis: np.ndarray) -> pd.DataFrame:
    comp = _sample_composition(rng, n)
    truth = true_properties(comp, element_basis)
    scales = np.array([PROP_RANGE[p][1] - PROP_RANGE[p][0] for p in TARGET_PROPERTIES])
    noise = rng.normal(0, 1, size=truth.shape) * (0.03 * truth + 0.005 * scales)
    measured = np.clip(truth + noise, 1e-6, None)

    df = pd.DataFrame(comp, columns=ELEMENTS_30)
    for j, prop in enumerate(TARGET_PROPERTIES):
        df[prop] = measured[:, j]

    # --- edge cases -----------------------------------------------------------------------------
    n_nan = max(1, int(0.01 * n))
    nan_rows = rng.choice(n, n_nan, replace=False)
    nan_cols = rng.choice(TARGET_PROPERTIES, n_nan)
    for r, c in zip(nan_rows, nan_cols):
        df.loc[r, c] = np.nan

    n_unnorm = max(1, int(0.01 * n))
    unnorm_rows = rng.choice(n, n_unnorm, replace=False)
    df.loc[unnorm_rows, ELEMENTS_30] = df.loc[unnorm_rows, ELEMENTS_30] * rng.uniform(0.5, 1.8, size=(n_unnorm, 1))

    n_dup = max(1, int(0.01 * n))
    dup_src = rng.choice(n, n_dup, replace=False)
    dup_rows = df.loc[dup_src].copy()
    dup_rows[ELEMENTS_30] = dup_rows[ELEMENTS_30] * rng.uniform(0.97, 1.03, size=(n_dup, 1))
    dup_rows[ELEMENTS_30] = dup_rows[ELEMENTS_30].div(dup_rows[ELEMENTS_30].sum(axis=1), axis=0)

    n_outlier = max(1, int(0.005 * n))
    outlier_rows = rng.choice(n, n_outlier, replace=False)
    df.loc[outlier_rows, "melting_point"] = df.loc[outlier_rows, "melting_point"] * 1.6  # valid but extreme

    df = pd.concat([df, dup_rows], ignore_index=True)
    df["row_id"] = [f"synth-chem-{i:06d}" for i in range(len(df))]
    return df.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)


_MATERIAL_ROOTS = [
    "Al", "Fe", "Ti", "Ni", "Cu", "Zr", "Hf", "Ta", "W", "Nb", "Mo", "Cr", "Co", "SiC", "TiC", "HfC", "ZrB2", "TiN",
]  # fmt: skip


def make_materials_dataset(rng: np.random.Generator, n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i in range(n):
        a, b = rng.choice(_MATERIAL_ROOTS, 2, replace=False)
        frac = round(float(rng.uniform(0.1, 0.9)), 2)
        formula = a if frac >= 0.98 else f"{a}{frac:g}{b}{round(1 - frac, 2):g}"
        e_gpa = float(np.exp(rng.normal(np.log(150.0), 0.6)))  # ~ 10 .. 900 GPa, median 150
        nu = float(np.clip(rng.normal(0.28, 0.06), -0.3, 0.48))
        density = float(np.exp(rng.normal(np.log(6.0), 0.5)))  # g/cm3, ~1..25
        rows.append(
            {
                "material_id": f"synth-mat-{i:06d}",
                "formula_pretty": formula,
                "E": e_gpa,
                "nu": nu,
                "density_g_cm3": density,
            }
        )
    df = pd.DataFrame(rows)

    n_invalid = max(1, int(0.03 * n))
    bad = rng.choice(n, n_invalid, replace=False)
    half = n_invalid // 2
    df.loc[bad[:half], "E"] = -np.abs(df.loc[bad[:half], "E"])  # non-physical: negative stiffness
    df.loc[bad[half:], "nu"] = rng.uniform(0.55, 0.9, size=len(bad[half:]))  # outside (-1, 0.5)

    n_dup = max(1, int(0.02 * n))
    dup_src = rng.choice(n, n_dup, replace=False)
    dup = df.loc[dup_src].copy()
    dup["material_id"] = [f"synth-mat-dup-{i:04d}" for i in range(len(dup))]
    dup["E"] = dup["E"] * rng.uniform(0.95, 1.05, size=len(dup))  # same formula, slightly different measurement
    df = pd.concat([df, dup], ignore_index=True)
    return df.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)


def split(df: pd.DataFrame, rng: np.random.Generator, fracs=(0.70, 0.15, 0.15)) -> dict[str, pd.DataFrame]:
    n = len(df)
    idx = rng.permutation(n)
    a = int(fracs[0] * n)
    b = a + int(fracs[1] * n)
    return {
        "train": df.iloc[idx[:a]].reset_index(drop=True),
        "val": df.iloc[idx[a:b]].reset_index(drop=True),
        "test": df.iloc[idx[b:]].reset_index(drop=True),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-chemistry", type=int, default=4000)
    ap.add_argument("--n-materials", type=int, default=600)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / "chemistry").mkdir(parents=True, exist_ok=True)
    (out / "materials").mkdir(parents=True, exist_ok=True)

    basis = _element_base_properties(np.random.default_rng(args.seed))  # separate stream, ground truth is fixed
    chem = make_chemistry_dataset(rng, args.n_chemistry, basis)
    for name, part in split(chem, rng).items():
        part.drop(columns=["row_id"]).to_parquet(out / "chemistry" / f"chemistry_{name}.parquet", index=False)
        part.drop(columns=["row_id"]).to_csv(out / "chemistry" / f"chemistry_{name}.csv", index=False)
    (out / "chemistry" / "_ground_truth_DO_NOT_USE_AS_FEATURES.json").write_text(
        json.dumps(
            {
                "note": "Ground truth used ONLY to score verification runs; never feed this to a model.",
                "seed": args.seed,
                "elements_30": ELEMENTS_30,
                "target_properties": TARGET_PROPERTIES,
                "element_basis": basis.tolist(),
                "interaction_coefficient": 0.12,
                "formula": (
                    "true = comp @ element_basis + 0.12 * (1 - sum(comp^2)) * log1p(n_active) * (comp @ element_basis)"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    mats = make_materials_dataset(rng, args.n_materials)
    for name, part in split(mats, rng).items():
        part.to_parquet(out / "materials" / f"materials_{name}.parquet", index=False)
        part.to_csv(out / "materials" / f"materials_{name}.csv", index=False)
    # load_material() consumes a single table (it is a lookup/selection table, not something trained on)
    mats.to_parquet(out / "materials" / "mp_elasticity_labels_synthetic.parquet", index=False)

    print(f"chemistry: {len(chem)} rows -> {out / 'chemistry'} (train/val/test + ground truth json)")
    print(f"materials: {len(mats)} rows -> {out / 'materials'} (train/val/test + full lookup table)")


if __name__ == "__main__":
    main()
