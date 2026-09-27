# Interface contracts (as implemented and enforced)

Every rule below is checked at runtime and covered by tests. Violations raise a subclass of `squad1.errors.Squad1Error`
(`ContractError`, `NormalizationError`, `NonFiniteError`, `PhysicsError`, `ConditioningError`, `UnsafeExpressionError`,
`ConvergenceError`) — nothing is silently coerced.

## 1. Design tensor

| Property | Rule |
|---|---|
| type / layout | `torch.Tensor`, floating, **channel-first `(B, C, H, W)`**, `H, W ≥ 4` |
| axes | dim 2 = `i` = *x* (flow direction), dim 3 = `j` = *y* |
| channels | named by a registered `DomainSpec`; `C` must equal the domain's channel count |
| finite | NaN/Inf rejected at construction of `CandidateDesign` |
| spacing | `h = L / (H − 1)` travels with the tensor (`CandidateDesign.h`) |
| model space | per-channel `[-1, 1]` (generator working range) |
| physical space | per-channel `[lo, hi]`, `linear` or `log` scale (`ChannelSpec`) |
| conversion | `ChannelNormalizer` — channel axis **1**, dtype/device/autograd preserved, exact round-trip |

Registered domains (`available_domains()`): `darcy [k, p]`, `darcy_biot [k, p, ux, uy]`, `cooling_plate [k, p]`,
`laplace_heat [T]`, `stress_equilibrium [sxx, sxy, syy]`, `reaction_diffusion [u]`, `diffusion_decay [c]`,
`thermal_advection [T]`, `navier_stokes [u, v, p]`. Add a domain with `register_domain(DomainSpec(...))` and a `Physics`
subclass. **The "18-channel design tensor" mentioned in a Squad-C update is not defined anywhere**; nothing here assumes 18 —
any channel count works once registered (see `DECISIONS.md` D-3).

Darcy defaults: `k ∈ [1e-2, 1e2]` (log), `p ∈ [0, 1]`; Dirichlet `p_left` at `x_min`, `p_right` at `x_max`, no-flow walls.

## 2. `CandidateDesign` and `ProjectionResult`

```python
CandidateDesign(domain, tensor, representation="model"|"physical", h,
                boundary={...}, physics={...}, generation={...})     # .to_model() / .to_physical() / .replace(...)

ProjectionResult(projected: CandidateDesign,          # SAME representation and dtype as the input
                 loss_before, loss_after,             # (B,)  mean squared scaled constraint
                 residual_rms_before, residual_rms_after,   # (B,)
                 correction_rms,                      # (B,)  RMS per element, model-space units
                 iterations, converged (B,) bool, rejected (B,) bool, reject_reason [str|None]*B,
                 method, runtime_s, metadata)         # metadata["fallback_solve"] per sample
```
* `rejected[i]` ⇒ `projected.tensor[i]` is **bit-identical to the input** and `reject_reason[i]` is set
  (`excessive_correction`, `non_finite_result`).
* `converged[i]` ⇒ physics residual RMS ≤ `tol` (default `1e-8`, scaled units, float64).
* Downstream code must treat `converged == False and rejected == False` as "improved but not on the manifold".

## 3. Conditioning vector (`CondSpec`)

Named schema first, tensor derived. Layout `[composition | inferred | target]`, `float32 (B, Dv)`,
`Dv = n_composition + n_inferred + n_target` (**never hard-coded**).

```python
spec = CondSpec(composition_order=("Al", "Cu", ...30 names...), inferred_order=(...), inferred_units=(...),
                target_order=(...), target_units=(...), mean=None, std=None, version="cond_v1")
vec = spec.encode(composition_dict, inferred_dict, target_dict)      # standardised float32 (Dv,)
spec.decode(vec) -> {"composition":…, "inferred":…, "target":…}      # exact inverse
spec.with_normalization(inferred_samples, target_samples)            # fit mean/std (composition stays identity)
```
Rules: composition ≥ 0 and Σ = 1 (±1e-3); inferred/target dictionaries must match the frozen order *exactly* (missing or
extra ⇒ error); NaN/Inf ⇒ `NonFiniteError`; if no mean/std is set, any |value| > 1000 is refused (the encoder expects
standardised inputs). The spec is JSON-serialisable and is written into every `DesignPackage`.

## 4. Inferred physics parameters (from the physics-inverse stage)

`InferredParameters(names, units, values, confidence?, mask?, version, source)` — validated schema.
`ParameterProvider` protocol: `infer(problem: Mapping) -> InferredParameters`. `check_against_spec_order` refuses a payload
whose names/units differ from what the `CondSpec` was frozen with. **Open: the parameter count/order/units from Shrot.**

## 5. Boundary field (`cond_field`)

`rasterize(BoundarySpec, H, W) -> (Cc=8, H, W)`; batch via `cond_field_batch` → `(B, 8, H, W)`.
Channels: `dirichlet_mask, dirichlet_value, neumann_mask, neumann_flux, robin_mask, robin_coeff, robin_ambient, source`.
Sides `x_min/x_max` = rows 0 / H−1, `y_min/y_max` = columns 0 / W−1. Same-side overlaps of different kinds ⇒ error; at
corners Dirichlet > Robin > Neumann. Full-side Dirichlet on `x_min`/`x_max` sets the Darcy `p_left`/`p_right`.

## 6. Generator ⇄ pipeline

`DiT(img_size, patch_size, in_channels, hidden_size, depth, num_heads, cond_dim=0, field_channels=0)`;
`forward(x, t, cond_vec=None, cond_field=None, drop_cond=None) -> eps`. `Squad1Pipeline` verifies at construction that
`model.cond_dim == CondSpec.dv` and `model.field_channels ∈ {0, 8}` — a mismatch with the frozen interface raises
immediately (the Week-4 checklist's "conditioning-dim mismatch guard").

## 7. `DesignPackage` (`package.json`, schema `squad1_design_package_v1`)

`domain, representation, tensor_shape, channels, h, config, quality_gate_raw (per-sample finite / range / residual / ok),
projection (ProjectionResult.to_dict), accepted [bool], cond_spec, provenance {squad1_version, seed, config_hash, python,
torch, numpy, scipy, cuda, device, git_commit}` plus `candidate_raw.npy`, `design_final.npy`.
`accepted[i] = converged[i] and not rejected[i]`.

## 8. Cooling-plate contract (application layer)

`CoolingPlate.report(k, h)` → per sample `max_temperature_C, pressure_drop_kPa, relative_weight, flow_residual,
energy_balance_error, temperature_ok, pressure_drop_ok, constraints_satisfied, targets`. `constraints_satisfied` is true only if
both limits hold **and** `flow_residual < 1e-8`; `optimize_layout` returns a design whose pressure channel is the exact flow
solution of its layout (it cannot degrade flow consistency) and never reports success it did not achieve.
