# `reports/` — verification run output

Generated artefacts, not hand-written. Produced by `scripts/verify_end_to_end.py`, which runs the real
`squad1` code (not a mock) against the files in `data/`:

* `phase3_verification.md` — human-readable run log (one section per pipeline step, PASS/FAIL, timing, output)
* `phase3_verification.json` — the same data, machine-readable

Regenerate:
```bash
python scripts/generate_synthetic_data.py   # if data/ doesn't exist yet
python scripts/verify_end_to_end.py
```
The chat summary given after this work ("Phase 3 — Verification") is a condensed narrative of exactly these
two files — nothing in that summary was invented separately from what these scripts actually printed.
