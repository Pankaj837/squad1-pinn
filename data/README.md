# `data/` — synthetic data

**Everything in this folder is synthetic.** No real chemistry, materials, or experimental data was
provided to Squad 1; these files exist only to exercise and verify the pipeline end-to-end (see
`reports/phase3_verification.md`). Do not use them for anything beyond that.

Regenerate at any time (deterministic, seed `20260927`):
```bash
python scripts/generate_synthetic_data.py [--n-chemistry 4000] [--n-materials 600] [--seed 20260927]
```
Full generation method, the exact ground-truth formula, and every assumption made are documented in the
script's module docstring (`scripts/generate_synthetic_data.py`) — read that before trusting any number
derived from this data.

## `chemistry/` — composition → property dataset (for `squad1.conditioning.ChemistryInverse`)

| File | Rows | Purpose |
|---|---|---|
| `chemistry_train.parquet` / `.csv` | ~2828 | fit the Random-Forest surrogate |
| `chemistry_val.parquet` / `.csv` | ~606 | held out, not used by any script here (available for your own tuning) |
| `chemistry_test.parquet` / `.csv` | ~606 | target values used in `scripts/verify_end_to_end.py` |
| `_ground_truth_DO_NOT_USE_AS_FEATURES.json` | — | the exact noise-free property function + per-element basis used to *generate* the data; used only to score verification runs (compare reported vs. true error) |

**Schema**: 30 element-fraction columns (`Al, Si, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Nb, Mo, Hf, Ta, W, Re,
Ru, Rh, Pd, Ag, Sn, Y, La, Ce, B, C, N, O` — a placeholder vocabulary; Squad 1's real element order is still
open, see `docs/DECISIONS.md` D-5) + `thermal_conductivity` (W/m/K) + `density` (kg/m³) + `specific_heat`
(J/kg/K) + `melting_point` (K). Rows are **not** cleaned: `ChemistryInverse.load_dataset()` does that (drops
NaN rows, drops negative/all-zero compositions, renormalises). Deliberately injected so the cleaning step has
something to do: ~1% NaN-valued rows, ~1% rows whose composition doesn't sum to 1, ~1% near-duplicate rows,
a few extreme-but-valid property outliers.

## `materials/` — elasticity lookup table (for `squad1.geometry.load_material`)

| File | Rows | Purpose |
|---|---|---|
| `mp_elasticity_labels_synthetic.parquet` | 612 | the full lookup table (this is what `load_material()` reads — it's a selection table, not something trained on) |
| `materials_train/val/test.parquet` / `.csv` | 428 / 92 / 92 | the same rows split three ways, in case you want to hold out a portion for your own analysis |

**Schema**: `material_id, formula_pretty, E` (GPa), `nu` (Poisson ratio), `density_g_cm3`. Stands in for the
`mp_elasticity_labels.parquet` referenced but never delivered in the original Topology report. ~3% of rows
are deliberately invalid (negative `E`, or `nu` outside `(-1, 0.5)`) to exercise `load_material`'s filtering;
a few duplicate formulas with slightly different measurements are included too.
