# Decisions, assumptions and open items

Status: **A** = adopted here (change is one commit if the team disagrees) · **O** = open, needs an owner's answer.
Nothing marked **O** blocks the tests; each has a safe default in the code.

| # | Topic | Decision / default | Status |
|---|---|---|---|
| D-1 | "PIRF" and "sCM-PINN" | The project documents give no formal definition (the PCFM report says so itself); the shipped code implemented both as the baseline step `x ← x − η∇L`. Not implemented under those names — `get_projector("pirf" \| "scm_pinn")` raises `NotImplementedError` with an explanation. `residual_weighted` is an explicit residual-feedback variant (weights ∝ \|c\|^β). Note: "PIRF" is also used in `Training Strategies.pdf` for training-time collocation re-weighting — a different thing. Consistency models (sCM) are a *sampler*, not a corrector. | **O** owner to supply definitions or retire the names |
| D-2 | PCFM meaning | Team docs expand PCFM as *Physics-Constrained Feature Manifold*; the published method (Utkarsh et al. 2025) is *Physics-Constrained Flow Matching* (hard constraints at inference by constraint projection). This package implements hard-constraint projection either way. | **O** confirm intended meaning |
| D-3 | "18-channel design tensor" | Mentioned once (Swaraj's update, "from Shourya"); not defined in any document. Any channel count works via `register_domain`. | **O** Shourya to define channel list |
| D-4 | `cond_vec` format | Named schema (`CondSpec`) is the source of truth; tensor is derived, standardised, `float32 (B, Dv)`, layout `[composition \| inferred \| target]`. Resolves the schema-first vs positional conflict. | **A** (needs sign-off) |
| D-5 | Counts / order / units | 30 element order, the *n* inferred parameters (8?) and 4 target conditions are **not frozen**; `Dv` is computed, never hard-coded. | **O** Swaraj / Shrot / Navneet |
| D-6 | Chemistry token target | SMILES is for molecules; both `SmilesTokenizer` (strict) and `FormulaTokenizer` (elements, numbers, `()`, `- / : .`; e.g. `HfC-SiC`) are provided. The 768-D encoder is agnostic. | **O** choose per use case (formula recommended for ceramics/alloys) |
| D-7 | Encoder training data | Paired `(cond_vec → string)` data and objective do not exist; trainability is verified by memorising a mock set only. | **O** dataset |
| D-8 | Boundary channels | `Cc = 8` (Dirichlet/Neumann/Robin masks+values, source). Cell-centred rasterisation, Dirichlet > Robin > Neumann at corners. | **A** (needs Data-team sign-off) |
| D-9 | Darcy boundary convention | Dirichlet `p_left`/`p_right` on `x_min`/`x_max`, no-flow on `y` walls, `h = L/(H−1)`. | **A** |
| D-10 | Domain library | Biology (Turing) and Maxwell are omitted: no steady residual formulation is specified (the earlier residuals were not consistent with their own data). Add via `Physics` + `register_domain`. | **O** spec |
| D-11 | Projection space | Projection runs in **model space** so the minimum-norm step means "smallest change in generator units", log-permeability is multiplicative, positivity is automatic, and `max_correction_rms` is channel/grid independent. | **A** |
| D-12 | Failure policy | Unconverged samples get the physics' exact partial solve when one exists (Darcy), reported in `metadata["fallback_solve"]`; otherwise they are returned improved-but-flagged. Rejected samples are returned bit-identical. | **A** |
| D-13 | Default tolerances | `tol = 1e-8` (scaled float64 residual RMS), `max_iters = 25`, LM damping 1e-12 → 1e8 adaptive. float32 storage floors the *returned* residual at ~1e-6. | **A** |
| D-14 | Cooling-plate model | Reduced-order, depth-averaged (Darcy flow + conduction–advection). Constants (κ, ρc, pump flow, `dp_scale_kpa`, limits 45 °C / 10 kPa) are **illustrative**; `dp_scale_kpa` is a calibration constant. Not validated CFD. | **O** real numbers / CFD calibration |
| D-15 | Generator | `squad1.generation.DiT` is a reference implementation with the project's constructor signature; the team's trained DiT can replace it (see INTEGRATION §4). | **O** provide DiT + checkpoint |
| D-16 | Thermodynamic checks | Not computed by `ChemistryInverse` (formation energy / hull distance / charge neutrality). Hook: `extra_checks`. Reports that claim "thermodynamic consistency" overstate the earlier code. | **O** data source |
| D-17 | Scope | The auto-PINN factory / simulation generator and the Gemini-routed screening pipeline are not part of this package (different subsystem; legacy tests cannot import). The injection-safe rule compiler **is** included for use by the LLM team. | **A** |
| D-18 | Secrets & VCS | A `.env` (Gemini key) was shipped inside the source archives → rotate the key; this repo has a `.gitignore` excluding `.env`, `.venv`, caches. Initialise git before sharing. | **O** action for the team lead |
| D-19 | Python / platforms | Executed on Python 3.13, torch 2.12 CPU, Windows. CI matrix (3.10–3.13, Linux) is configured but has **not** been run. | **O** run CI once pushed |
