"""Phase 3 verification: run the real pipeline against the synthetic data on disk and report results.

This is not a mock of the pipeline — it calls the actual `squad1.conditioning.ChemistryInverse`,
`squad1.geometry.load_material`, `squad1.contracts.CondSpec` and `squad1.encoding.ConditioningToTokens`
against the files in `data/`. It prints a human-readable report and writes the same data to
`reports/phase3_verification.json` (machine-readable) and `reports/phase3_verification.md` (narrative).

Run:  python scripts/verify_end_to_end.py
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))  # for true_properties() from the generation script
sys.path.insert(0, str(ROOT / "src"))

from generate_synthetic_data import ELEMENTS_30, TARGET_PROPERTIES, true_properties  # noqa: E402

from squad1.conditioning import ChemistryConfig, ChemistryInverse, composition_vector  # noqa: E402
from squad1.contracts import CondSpec  # noqa: E402
from squad1.encoding import ConditioningToTokens  # noqa: E402
from squad1.geometry import load_material  # noqa: E402

DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def step(name: str, fn):
    t0 = time.perf_counter()
    try:
        result = fn()
        return {"step": name, "status": "ok", "seconds": time.perf_counter() - t0, **result}
    except Exception as exc:
        return {
            "step": name,
            "status": "error",
            "seconds": time.perf_counter() - t0,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=6),
        }


def chemistry_fit_and_discover() -> dict[str, Any]:
    gt = json.loads((DATA / "chemistry" / "_ground_truth_DO_NOT_USE_AS_FEATURES.json").read_text())
    basis = np.array(gt["element_basis"])

    cfg = ChemistryConfig(elements=tuple(ELEMENTS_30), target_properties=tuple(TARGET_PROPERTIES), top_dataset_seeds=8)
    inv = ChemistryInverse(cfg).fit(DATA / "chemistry" / "chemistry_train.parquet")

    test_df = pd.read_parquet(DATA / "chemistry" / "chemistry_test.parquet").dropna()
    rows = test_df.sample(n=5, random_state=0)
    scales = {p: max(float(test_df[p].std()), 1e-9) for p in TARGET_PROPERTIES}

    cases = []
    for _, row in rows.iterrows():
        targets = {p: float(row[p]) for p in TARGET_PROPERTIES}
        result = inv.discover(targets)
        best = result.best
        comp = np.array(composition_vector(best.composition, ELEMENTS_30))
        true_pred = true_properties(comp[None], basis)[0]
        true_err = float(
            np.mean([((true_pred[i] - targets[p]) / scales[p]) ** 2 for i, p in enumerate(TARGET_PROPERTIES)])
        )
        cases.append(
            {
                "target": targets,
                "reported_target_error": best.target_error,
                "true_target_error_vs_ground_truth": true_err,
                "target_satisfied": best.target_satisfied,
                "surrogate_reliable": best.surrogate_reliable,
                "physically_valid": best.physically_valid,
            }
        )
    return {
        "train_rows_used": inv._df.shape[0] if inv._df is not None else None,  # after load_dataset cleaning
        "surrogate_oob_rmse": {k: round(v, 4) for k, v in result.surrogate_oob_rmse.items()},
        "cases": cases,
    }


def materials_lookup() -> dict[str, Any]:
    path = DATA / "materials" / "mp_elasticity_labels_synthetic.parquet"
    stiffest = load_material(path)
    near_200 = load_material(path, target_E_gpa=200.0)
    by_name = load_material(path, formula=stiffest["formula"])
    raw = pd.read_parquet(path)
    return {
        "raw_rows": len(raw),
        "invalid_rows_present": int(((raw.E <= 0) | (raw.nu <= -1) | (raw.nu >= 0.5)).sum()),
        "stiffest_material": stiffest,
        "nearest_to_200GPa": near_200,
        "lookup_by_formula_matches_stiffest": by_name["material_id"] == stiffest["material_id"],
    }


def conditioning_and_encoder(chem_result: dict[str, Any]) -> dict[str, Any]:
    """Take a real discovered composition through CondSpec -> ConditioningToTokens (the C02 -> C03 boundary)."""
    if chem_result["status"] != "ok" or not chem_result["cases"]:
        return {"skipped": "chemistry step did not produce a candidate"}
    case = chem_result["cases"][0]
    inferred_names = ("physics_inverse_param_1", "physics_inverse_param_2")  # stub: C01 not delivered
    inferred = {"physics_inverse_param_1": 0.4, "physics_inverse_param_2": -0.2}
    raw_spec = CondSpec(
        tuple(ELEMENTS_30),
        inferred_names,
        ("dimensionless", "dimensionless"),
        tuple(case["target"].keys()),
        ("W/mK", "kg/m3", "J/kgK", "K"),
    )
    # CondSpec.encode refuses raw (un-normalised) inferred/target values above MAX_RAW_ABS by design
    # (see squad1.contracts.cond) -- fit a standardisation from the training data first, as documented
    # in docs/INTEGRATION.md, instead of feeding it raw physical units.
    train_df = pd.read_parquet(DATA / "chemistry" / "chemistry_train.parquet").dropna()
    rng = np.random.default_rng(1)
    inferred_samples = rng.normal(0, 1, size=(len(train_df), len(inferred_names)))
    target_samples = train_df[TARGET_PROPERTIES].to_numpy()
    spec = raw_spec.with_normalization(inferred_samples, target_samples)

    # re-run discover once more to get an actual composition dict (not just the summary stored above)
    cfg = ChemistryConfig(elements=tuple(ELEMENTS_30), target_properties=tuple(TARGET_PROPERTIES), top_dataset_seeds=8)
    inv = ChemistryInverse(cfg).fit(DATA / "chemistry" / "chemistry_train.parquet")
    result = inv.discover(case["target"])
    comp = result.best.composition
    cond_vec = spec.encode(comp, inferred, case["target"]).unsqueeze(0)

    torch.manual_seed(0)
    model = ConditioningToTokens(
        {"composition": 30, "inferred": 2, "target": 4},
        vocab_size=16,
        embed_dim=64,
        max_seq_len=8,
        num_heads=4,
        num_layers=1,
        ff_dim=64,
    )
    with torch.no_grad():
        emb, ids, _logits = model(cond_vec)
    return {
        "cond_vec_shape": list(cond_vec.shape),
        "cond_vec_finite": bool(torch.isfinite(cond_vec).all()),
        "token_embeddings_shape": list(emb.shape),
        "token_ids_shape": list(ids.shape),
    }


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    results = []
    r1 = step("chemistry_fit_and_discover (data/chemistry/*.parquet)", chemistry_fit_and_discover)
    results.append(r1)
    results.append(step("materials_lookup (data/materials/*.parquet)", materials_lookup))
    results.append(
        step("conditioning_to_tokens (CondSpec -> ConditioningToTokens)", lambda: conditioning_and_encoder(r1))
    )

    (REPORTS / "phase3_verification.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Phase 3 verification run log",
        "",
        "Generated by `scripts/verify_end_to_end.py`. Real code, real files on disk.",
        "",
    ]
    ok_count = sum(1 for r in results if r["status"] == "ok")
    for r in results:
        mark = "PASS" if r["status"] == "ok" else "FAIL"
        lines.append(f"## [{mark}] {r['step']} ({r['seconds']:.2f}s)")
        if r["status"] == "ok":
            lines.append("```json")
            lines.append(
                json.dumps(
                    {k: v for k, v in r.items() if k not in ("step", "status", "seconds")}, indent=2, default=str
                )
            )
            lines.append("```")
        else:
            lines.append(f"Error: `{r['error']}`")
            lines.append("```")
            lines.append(r["traceback"])
            lines.append("```")
        lines.append("")
    (REPORTS / "phase3_verification.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"{ok_count}/{len(results)} steps OK -> reports/phase3_verification.{{json,md}}")
    for r in results:
        print(f"  [{r['status'].upper():5}] {r['step']} ({r['seconds']:.2f}s)")
    if ok_count != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
