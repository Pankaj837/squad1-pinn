# squad1 — PRANA-G Squad 1 (Generative & Inverse Core + Conditioning)

One installable, tested Python package that replaces the scattered Squad 1 code (chemistry inverse, token encoder,
generators, PCFM projection, cooling-plate POC, topology/lattice, PINN training) with a single set of **contracts**,
**physics that is actually verified**, and a **hard-constraint projection** that provably lands on the physics manifold.

```
 chemistry ──┐                          boundary spec ──rasterize──► cond_field (B,Cc,H,W)
 inferred    ├─► CondSpec ─► cond_vec (B,Dv) ─┐
 params ─────┤   (named schema, versioned)    │
 targets ────┘                                ▼
                                    ┌──────────────────────┐   CandidateDesign (model space, (B,C,H,W))
                                    │  DiT + DDIM sampler  │ ─────────────────────────────────────────┐
                                    └──────────────────────┘                                          ▼
                                                              quality_check ─► PCFMPipeline (Gauss–Newton hard projection,
                                                                               active-set bounds, LM damping, exact-solve fallback)
                                                                                         │
                                                              ProjectionResult ◄─────────┘  (per-sample: converged / rejected + reason)
                                                                    │
                                                                    ▼
                                          DesignPackage (JSON + .npy + provenance) ──► Squad B / Squad D / cooling-plate application
```

## Install

```bash
pip install -e ".[dev]"                      # torch, numpy, scipy + pandas/sklearn/pyarrow/trimesh/pytest/ruff/mypy
squad1 selftest                              # 6 fast sanity checks, no data needed
python scripts/generate_synthetic_data.py    # writes data/chemistry, data/materials (synthetic — see data/README.md)
python scripts/verify_end_to_end.py          # runs the real pipeline against that data -> reports/
squad1 demo --out demo_out                   # trains a toy generator, samples, projects, writes a design package
python -m pytest                             # full suite (≈ 4–6 min on CPU, 97 % line coverage)
```
Python ≥ 3.10. Verified here on Python 3.13 / torch 2.12 (CPU). GPU code paths are exercised by `squad1 gpu-suite` (see
`docs/INTEGRATION.md` §5) — they have **not** been run on a GPU yet.

## Repository layout

```
squad1_repo/
├─ src/squad1/       source code, organised by module (contracts, physics, projection, generation,
│                    conditioning, encoding, applications, geometry, pinn, pipeline, utils)
├─ tests/            pytest suite: unit/ (per-module) + integration/ (end-to-end, CLI, docs, synthetic data)
├─ data/             synthetic chemistry + materials datasets and how they were generated (data/README.md)
├─ reports/          generated verification run output (reports/README.md)
├─ scripts/          generate_synthetic_data.py, verify_end_to_end.py, collect_evidence.py, run_gpu_suite.py
├─ docs/             INTERFACES / INTEGRATION / DECISIONS / VERIFICATION / LITERATURE
├─ archive/          nothing orphaned in this repo (clean rewrite); points to where the superseded
│                    original code is kept for traceability (archive/README.md)
├─ .github/workflows/ci.yml   lint + mypy + tests (3.10-3.13) + optional self-hosted GPU job
└─ README.md / CHANGELOG.md / CONTRIBUTING.md / pyproject.toml / .pre-commit-config.yaml
```

## What is inside

| Package | What it provides | Replaces |
|---|---|---|
| `squad1.contracts` | `ChannelSpec`/`DomainSpec` registry, `ChannelNormalizer` (channel dim 1, log scales, dtype/autograd-safe), `CandidateDesign`, `ProjectionResult`, `CondSpec` | scattered shape/units conventions |
| `squad1.physics` | Darcy (finite-volume, harmonic-mean faces, dense + CG solvers), Biot poroelasticity (verified by manufactured solution), domain library (Laplace, stress equilibrium, KPP, diffusion-decay, advection-diffusion, Navier–Stokes/Kovasznay) with **on-manifold** data generators | `pcfm/physics/*`, `generative_models.py` |
| `squad1.projection` | `PCFMPipeline`, Gauss–Newton (dense & matrix-free), solve, gradient and residual-weighted projectors, model-space projection, rejection path | `pcfm/projection`, `pipeline.py` |
| `squad1.generation` | Reference DiT (adaLN-Zero, cond_vec + cond_field + CFG), cosine DDIM scheduler, sampler with optional physics guidance, training loop with EMA, quality gate | missing `common/dit`, `common/diffusion` |
| `squad1.conditioning` | `ChemistryInverse` (honest surrogate error, feasibility flags), `InferredParameters` provider interface, boundary rasteriser (`cond_field`, Cc = 8) | `chemistry_inverse.py`, Shrot/Data-team interfaces |
| `squad1.encoding` | strict `SmilesTokenizer` and `FormulaTokenizer`, `ConditioningToTokens` (non-degenerate cross-attention, 768-D) | Afnan's encoder + tokenizer |
| `squad1.applications` | cooling plate on **solved** flow + heat physics, differentiable layout optimiser, watertight STL export | `physics_l.py`, `projector_cl.py`, `run_cl.py` |
| `squad1.geometry` | graph-canonical lattices, voxel-union density, closed mesh + STL I/O, 2-D SIMP topology optimiser, material loader | Topology/Lattice report code |
| `squad1.pinn` | MLP (Fourier features), problems with exact solutions, Adam→L-BFGS trainer (resampling, RAD, gradient-norm balancing), injection-safe rule compiler | `training/pinn_trainer.py`, `sympy_loss_generator.py` |
| `squad1.pipeline` | end-to-end pipeline + `DesignPackage`, toy generator, benchmark suite G1–G8 | Week-4 integration scaffold |

## Quick example

```python
import torch
from squad1.contracts import CandidateDesign
from squad1.physics.darcy import solve_pressure
from squad1.projection import PCFMPipeline

k = torch.exp(0.5 * torch.randn(2, 16, 16, dtype=torch.float64))
p = solve_pressure(k, 1 / 15) + 0.05 * torch.randn(2, 16, 16, dtype=torch.float64)  # imperfect "generated" field
cand = CandidateDesign("darcy", torch.stack([k, p], 1).float(), "physical", h=1 / 15).to_model()

result = PCFMPipeline().project(cand)  # returned in model space, float32, like the input
print(result.residual_rms_before, result.residual_rms_after, result.converged, result.reject_reason)
```

## Guarantees enforced by tests
* **Physics is not zero**: Darcy/Biot/all domains have analytic or manufactured-solution tests, gradient checks (autograd == finite difference) and negative controls (a wrong field must give a large residual).
* **Projection is hard**: Gauss–Newton reaches residual ≲ 1e-9 where gradient descent stalls ~1e-3; results stay inside the registered channel ranges; rejected samples return **bit-identical input**.
* **Interfaces fail loudly**: NaN/Inf, wrong shapes, wrong channel counts, un-normalised values, mismatched `Dv`/`Cc`, unknown tokens, unsafe expressions — each raises a typed error (`squad1.errors`).
* **Reproducible**: seeded everywhere, config hash + versions in every output JSON.

## Known limitations (read before integrating)
* The DiT here is a **reference implementation** with the project's constructor signature; swap in the team's trained DiT (see `docs/INTEGRATION.md`). Quality of *real* generated samples is untested.
* PIRF / sCM-PINN have no formal definition in the project documents → not implemented under those names (`residual_weighted` is an explicit residual-feedback variant; see `docs/DECISIONS.md`).
* Physics-inverse (Shrot) and the boundary parser (Data team) are **interfaces + validators**, not solvers.
* Cooling-plate constants are illustrative (reduced-order model), not validated CFD.
* Dense solves cap the differentiable cooling model at ≈ 5000 cells; matrix-free projection needs preconditioning for large grids (measure with G2).
* Biology (Turing) and Maxwell domains are not included: no steady formulation is specified.

See `docs/`: [`INTERFACES.md`](docs/INTERFACES.md) (contracts), [`INTEGRATION.md`](docs/INTEGRATION.md) (plug in real components),
[`DECISIONS.md`](docs/DECISIONS.md) (assumptions/open items), [`VERIFICATION.md`](docs/VERIFICATION.md) (reproducible evidence),
[`LITERATURE.md`](docs/LITERATURE.md) (design choices vs. published methods).
