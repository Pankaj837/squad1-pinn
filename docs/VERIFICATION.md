# Verification summary (final gate, reproducible)

Environment: Python 3.13.13, torch 2.12.1 CPU, Windows. Not yet run on GPU or on CI (see `DECISIONS.md` D-19).

```bash
pip install -e ".[dev]"
squad1 selftest                     # 6/6 PASS
ruff format --check src tests       # 65 files already formatted
ruff check src tests                # All checks passed!
mypy                                 # Success: no issues found in 47 source files
python -m pytest --cov=squad1       # 245 passed, 97% coverage (85% required)
```

## Headline evidence (regenerate with `python scripts/collect_evidence.py`, raw numbers in `docs/evidence.json`)

**Darcy physics is real and correct** (`darcy_analytic`):
| Case | Result |
|---|---|
| linear `p`, uniform `k` | max residual `6.1e-14` (should be 0) |
| `p = x²`, `k = 1` | mean residual `2.0000` (exact analytic value is 2) |
| `k = 1+x` exact log solution | max scaled residual `1.5e-6` (2nd-order truncation only) |

**Domain library residuals** — on-manifold data vs. random noise of the same scale (`domain_library_residuals`):
| Domain | data residual RMS | noise residual RMS | ratio |
|---|---|---|---|
| diffusion_decay | 1.8e-8 | 0.191 | 1.1e-7 |
| laplace_heat | 3.1e-4 | 1.18 | 2.7e-4 |
| navier_stokes (Kovasznay) | 7.2e-5 | 0.180 | 4.0e-4 |
| reaction_diffusion | 7.8e-9 | 0.0785 | 1.0e-7 |
| stress_equilibrium | 6.6e-5 | 0.0147 | 4.5e-3 |
| thermal_advection | 7.8e-9 | 0.0847 | 9.2e-8 |

**Projection: Gauss–Newton vs. gradient descent vs. exact solve**, imperfect Darcy candidates, 16×16, B=4 (`projection_comparison_16x16`):
| noise | loss before | gradient | residual_weighted | **gauss_newton** | solve (k fixed) |
|---|---|---|---|---|---|
| 0.02 | 9.5e-3 | 3.6e-7 (rel-L2 p 0.43%, 0.76s) | 1.1e-6 (0.48%, 0.63s) | **1.0e-18 (0.02%, 0.04s)** | 1.3e-31 (0.00%) |
| 0.10 | 2.3e-1 | 9.1e-6 (2.1%, 0.62s) | 3.3e-5 (2.4%, 0.52s) | **5.4e-23 (0.09%, 0.08s)** | 1.3e-31 (0.00%) |
| 0.30 | 2.2e0 | 1.0e-4 (8.2%, 0.74s) | 3.0e-4 (9.3%, 0.74s) | **7.5e-26 (0.87%, 0.22s)** | 1.3e-31 (0.00%) |
Reading: GN reaches machine-zero residual and the smallest deviation from the true field, faster than 300 steps of gradient descent, at every noise level.

**Cooling plate: layout-sensitive, energy-conserving, honestly gated** (`cooling_plate`):
* Random layouts: max energy-balance error `7.0e-14` (should be ≈0).
* Optimizer repair of a violating 2-channel design: before `{T=41.9°C, Δp=16.1 kPa, weight=0.83, satisfied=False}` → after `{T=37.4°C, Δp=4.8 kPa, weight=0.26, satisfied=True}`, flow residual stays `< 1e-8` throughout (never degrades, unlike the original POC's contract.json).
* Same channel area, different layout: straight channels Δp = **8.04 kPa**; the *same pixels shuffled* → Δp = **3405.79 kPa** (layout matters; the original algebraic model gave identical numbers for both).

**Geometry** (`geometry`):
* Lattice (BCC+edges, 2×2×2): 118 unique struts, relative density (voxel union) **0.227** vs the naive concatenated-cylinder sum **0.275** (+21% overstatement from double-counted joints) — a single watertight body.
* Topology (SIMP, MBB 60×20, vf 0.5): compliance converges to **218.8** (matches the published 88-line-code benchmark range), volume error `1.9e-7`.
* Cantilever vs. Timoshenko beam theory: compliance ratio **0.993** (within 1%).

**PINN training vs. the framework it replaces** (`pinn`, heat equation, Adam→L-BFGS):
* This trainer: **relative L2 error 0.00097** against the exact solution.
* The original framework path on the same problem: **relative L2 error 0.970**, reported as "PASSED VALIDATION" (physics residual alone was checked, not the answer).

**Trained-generator behaviour** (`tests/integration/test_generation_quality.py`, small DiT trained ~20 s CPU):
* Conditioning is respected: log-permeability spread at the high-contrast condition is `> 2×` the spread at the low-contrast condition.
* Classifier-free guidance sharpens that gap further.
* Physics guidance during sampling: raw (pre-projection) residual drops `> 3×` at guidance scale 0.2 vs. 0.

## What is *not* claimed
* No result here uses the team's real trained DiT or real chemistry/materials data — synthetic/manufactured data only (see `docs/DECISIONS.md`, `README.md` "Known limitations").
* GPU behaviour is untested; `squad1 gpu-suite` defines the experiments but has only run in CPU smoke mode.
* Literature citations in `docs/LITERATURE.md` were not looked up live this session.
