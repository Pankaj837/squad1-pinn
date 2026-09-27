# Design choices vs the literature

> **Not live-verified.** Web access was unavailable while this package was built; the references are cited from the
> author's knowledge (authors/year/title only). Treat them as review leads and confirm details before quoting.

| Area | What this package does | Basis |
|---|---|---|
| Hard-constraint projection at inference | Gauss–Newton minimum-norm steps onto `c(x) = 0` (+ active set, LM damping, exact-solve fallback) instead of a fixed-step penalty descent | Utkarsh et al. 2025, *Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints* |
| Why penalties stall | gradient descent on a Laplacian-type residual is ill-conditioned; measured here: GD ≈ 1e-3…1e-4 vs GN ≲ 1e-9 | Krishnapriyan et al. 2021, *Characterizing possible failure modes in PINNs*; Rathore et al. 2024, *Challenges in training PINNs: a loss landscape perspective* |
| DiT backbone | adaLN-Zero transformer blocks, patch embedding, sin-cos positions | Peebles & Xie 2023, *Scalable Diffusion Models with Transformers* |
| Noise schedule / sampler / guidance | cosine schedule, deterministic DDIM, classifier-free guidance with a learned null condition | Nichol & Dhariwal 2021; Song, Meng & Ermon 2021; Ho & Salimans 2022 |
| Physics-guided sampling (optional) | gradient of the physics loss of the predicted clean sample nudges each step (scale 0 by default) | Chung et al. 2023 (DPS); Huang et al. 2024 (DiffusionPDE) |
| Consistency models | not used as a corrector (they are samplers) | Lu & Song 2024, *Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models* |
| Conditioning tokens | one token per scalar (value·W + b + slot + group embedding) so cross-attention has something to select | Gorishniy et al. 2021, *Revisiting Deep Learning Models for Tabular Data* (FT-Transformer) |
| PINN training | Adam → L-BFGS, resampling every step, RAD refinement, gradient-norm loss balancing, Fourier features, input non-dimensionalisation, validation by relative L2 against an exact solution | Wang, Teng & Perdikaris 2021; Wang, Yu & Perdikaris 2022; Wang, Wang & Perdikaris 2023 (*An expert's guide to training PINNs*); Wu et al. 2023 (RAD/RAR); Tancik et al. 2020 |
| PINN benchmark hygiene | error against reference solutions, several seeds | Hao et al. 2023 (PINNacle) |
| Finite-volume Darcy | harmonic-mean face permeability for high-contrast media; conservative form | standard reservoir/porous-media practice (e.g. Aziz & Settari) |
| Biot poroelasticity | steady quasi-static Navier–Cauchy + Darcy, verified by manufactured solutions | Biot 1941; Roache 2002 (MMS) |
| Navier–Stokes check | Kovasznay flow (exact steady solution) | Kovasznay 1948 |
| Topology optimisation | SIMP, density filter, optimality criteria; MBB half-beam with symmetry BCs; compliance 218.8 for 60×20, vf 0.5 is in the range reported for the 88-line code | Sigmund 2001; Andreassen et al. 2011 |
| Materials inverse design | RF surrogate + simplex search is a **baseline**; uncertainty-aware / generative approaches are the direction | Zeni et al. 2025 (MatterGen); GP/ensemble Bayesian optimisation |
| Rule-string safety | allow-list AST evaluation, no `eval`/`exec` of untrusted text | standard secure-evaluation practice |
