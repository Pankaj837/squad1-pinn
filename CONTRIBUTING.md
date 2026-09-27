# Contributing

1. `pip install -e ".[dev]"` and `pre-commit install` (ruff + mypy).
2. Every behaviour change needs a test. Physics needs **an analytic / manufactured-solution test and a negative control**
   (a wrong field must produce a large residual) — the earlier code passed 28/28 tests with a residual hard-wired to zero.
3. Add a new domain: `register_domain(DomainSpec(...))`, a `Physics` subclass (`constraint_vector`, h²-scaled, zero iff valid),
   an on-manifold `sample()` if you want it in the data generators, then a row in `tests/unit/test_projection_domains.py`.
4. Fail loudly: raise a `squad1.errors` subclass; never return a silently altered value.
5. Keep seeds explicit (`seed_everything`) and write provenance into every output.
6. CI gates: `ruff check`, `ruff format --check`, `mypy`, `pytest` (coverage ≥ 85 %).
