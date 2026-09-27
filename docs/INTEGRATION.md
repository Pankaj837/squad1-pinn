# Integration guide

## 0. Order of operations

1. Freeze the interfaces in `INTERFACES.md` (sign-off list in `DECISIONS.md`).
2. Plug the real components into the four seams below (each is a ≤ 20-line adapter).
3. Run `squad1 selftest`, then `python -m pytest`, then `squad1 gpu-suite` on the GPU machine.
4. Replace the toy generator with the trained DiT and run the **central validation** (section 5).

## 1. Chemistry → conditioning vector (Swaraj)

```python
from squad1.conditioning import ChemistryConfig, ChemistryInverse
from squad1.contracts import CondSpec

cfg = ChemistryConfig(elements=ELEMENTS_30, target_properties=("thermal_conductivity", "density", "specific_heat", "melting_point"),
                      physical_limits={"density": {"min": 0.1}}, operating_temperature_k=2773.0)
inv = ChemistryInverse(cfg).fit("chemistry_dataset.parquet")           # DataFrame or csv/parquet path
result = inv.discover({"thermal_conductivity": 18.0, "density": 4200.0, ...})
best = result.best                                                     # .target_satisfied, .surrogate_reliable, .surrogate_oob_rmse
spec = CondSpec(ELEMENTS_30, INFERRED_NAMES, INFERRED_UNITS, TARGET_NAMES, TARGET_UNITS).with_normalization(inferred_samples, target_samples)
cond_vec = spec.encode(best.composition, inferred_params.as_dict(), targets)
```
* Thermodynamic checks (formation energy, hull distance, charge neutrality) are **not** computed: pass `extra_checks=fn(comp, pred) -> {name: bool}`.
* Refuse to condition on unreliable results: gate on `best.target_satisfied and best.surrogate_reliable`.

## 2. Physics inverse → `InferredParameters` (Shrot)

Implement `ParameterProvider.infer(problem) -> InferredParameters` with **fixed names, units, order, confidence and mask**;
`StaticProvider` is available for tests. `check_against_spec_order(params, spec.inferred_order, spec.inferred_units)` guards the seam.

## 3. Boundary parser → `cond_field` (Data team)

Emit a `BoundarySpec` (segments + optional source regions). `cond_field_batch(spec, H, batch)` produces the `(B, 8, H, W)` tensor;
`darcy_bc_from_boundary(spec)` forwards `p_left`/`p_right` to the physics. If the parser needs more than Dirichlet/Neumann/Robin/source
channels, extend `COND_FIELD_CHANNELS` (and the DiT's `field_channels`) — the pipeline checks the count.

## 4. Generator (Pankaj / Navneet)

`squad1.generation.DiT` reproduces the constructor used by the cooling POC (`DiT(img_size, patch_size, in_channels, hidden_size, depth, num_heads)`)
plus `cond_dim` / `field_channels`. To use the team's own DiT, satisfy the same `forward(x, t, cond_vec, cond_field, drop_cond)` signature
and expose `.img_size`, `.in_channels`, `.cond_dim`, `.field_channels` — nothing else in the package depends on DiT internals.
Checkpoints: `torch.save({"model": model.state_dict()}, path)`; load with `weights_only=True`.

```python
from squad1.pipeline import Squad1Pipeline, E2EConfig
pipe = Squad1Pipeline(model, scheduler, cond_spec, E2EConfig(domain="darcy", n=64, steps=50, cfg_scale=1.5, method="gauss_newton",
                                                             max_correction_rms=0.5, seed=0))
pkg = pipe.run(cond_items, boundary=BoundarySpec(...))
pkg.save("out/")           # package.json + candidate_raw.npy + design_final.npy
```

## 5. The central validation (do this first on GPU)

Generate with `cfg_scale=0` (imperfect on purpose), compare `pkg.result.residual_rms_before/after`, `correction_rms`,
`converged`, `rejected` and `metadata["fallback_solve"]`. With a fixed `k`, Darcy projection equals the exact flow solve; the informative
experiment is the joint `(k, p)` projection on real DiT samples (`method="gauss_newton"`) versus `method="gradient"`.

## 6. Migrating from the old code

| Old | New |
|---|---|
| `pcfm.physics.darcy.DarcyPhysics` (rank-2 `[N,2]`, residual `* 0.0`) | `squad1.physics.DarcyPhysics` (`(B,2,H,W)`, FV residual, verified) |
| `pcfm.interface.normalization.ChannelAwareNormalizer` | `squad1.contracts.ChannelNormalizer` |
| `pcfm.projection.Projector / PIRFProjector`, `scm_pinn` | `get_projector("gauss_newton" \| "solve" \| "gradient" \| "residual_weighted")` |
| `pcfm.pipeline.project(candidate, domain)` | `PCFMPipeline(method).project(candidate)` |
| `physics_l.py`, `projector_cl.py`, `run_cl.py` | `squad1.applications.cooling_plate` (`CoolingPlate`, `optimize_layout`, `plate_to_stl`) |
| `chemistry_inverse.py` (module globals) | `squad1.conditioning.ChemistryInverse(ChemistryConfig(...))` |
| Afnan `tokenizer.py` / `model.py` | `squad1.encoding.SmilesTokenizer / FormulaTokenizer / ConditioningToTokens` |
| `training/pinn_trainer.py`, `sympy_loss_generator.py` | `squad1.pinn.train`, `squad1.pinn.compile_rule` |

Out of scope (not ported, still in `PINN_Squad1_Organized/03_components/C11-C12`): the auto-PINN factory / simulation generator (2 200 lines,
6 legacy test files that cannot import) and the Gemini-routed screening pipeline. They are a different subsystem from Squad 1's contracts.
