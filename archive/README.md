# `archive/` — deprecated / superseded / not-sure-yet files

Empty by design: `squad1_repo` was built as a clean rewrite (see `CHANGELOG.md`), so every file
currently in the repo is in active use — nothing here needed to be shelved.

**The material this repo replaces was not deleted; it lives outside this repo** and is kept for
traceability:

| Where | What it is |
|---|---|
| `../PINN_Squad1_Organized/90_reference_legacy/` | the original uploaded Squad-1 files (chemistry_inverse.py, Afnan's token-encoder code, the PCFM package, cooling-plate POC, etc.) exactly as received, plus the pre-restructure "PRANA-G" framework and older per-person reports |
| `../PINN_Squad1_Organized/03_components/*/fixes/` | the intermediate, per-component bugfix versions written while auditing the original code (kept so each fix can be traced back to the defect it corrects — see `../PINN_Squad1_Organized/05_reports/STEP2_verification_report.md` and `STEP3_fix_register.md`) |
| `../PINN_Squad1_Organized/00_INDEX.md` | the defect-by-defect audit of the original code that this rewrite was built to fix |

If something later turns out to be dead code *inside* this repo, move it here — don't delete it — and
add a line to this file noting what it was and why it moved.
