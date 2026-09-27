# Changelog

## 1.0.0 — first integrated release

Consolidates the Squad 1 components into one package (see README for the module map). Compared with the code it replaces:

**Fixed (defects found while auditing the uploaded code)**
* Darcy residual was multiplied by `0.0`; evaluator crashed for N > 2 and rejected the generator's `(B,C,H,W)` layout → new finite-volume physics verified against analytic/manufactured solutions.
* PIRF / sCM-PINN were copies of the baseline step; benchmarks compared identical algorithms on a zero residual → removed; hard Gauss–Newton projection added.
* Normaliser scaled the last axis, ignored named channels, returned numpy float64, and the pipeline never converted results back to model space → `ChannelNormalizer` + model-space projection.
* PINN trainer certified a network with 97 % error (no IC/BC for plain inputs; validation = residual ≤ 1) → explicit IC/BC, exact-solution validation, seeding, resampling.
* `sympify` + `exec` on rule strings (arbitrary code execution) → allow-list compiler.
* Tokenizer silently dropped `Hf/Ti/Zr/Mo/Ta/W`, lost `<EOS>` on truncation → strict tokenizers.
* Cross-attention over a single context token (softmax ≡ 1) → per-scalar context tokens.
* Chemistry: reported error 38× optimistic, NaN and raw-scale values passed through → out-of-bag-honest scoring, guards.
* Cooling-plate metrics depended only on channel area fraction; flow residual worsened after "projection"; STL had inverted normals → solved-physics model, exact-flow projection, closed positively-oriented meshes.
* Lattice: duplicated struts (up to 2.25×), 160 disconnected bodies, density +54 %; MBB BCs non-standard → graph-canonical lattice, voxel-union density, standard BCs.
* Generators' synthetic data violated their own PDE residuals → on-manifold data (analytic or Newton-solved).

**Added**: contracts, `DesignPackage`, boundary rasteriser, `CondSpec`, generator checkpoints, SIMP optimiser, benchmark suite G1–G8, CLI, CI configuration.

**Not included / open**: see `docs/DECISIONS.md`.
