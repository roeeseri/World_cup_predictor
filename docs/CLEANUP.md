# Cleanup verification — 2026-10-03

The cleanup preserves active prediction behavior while reducing the repository's surface area.

| Area | Before | After |
| --- | ---: | ---: |
| Notebooks | 18 | 8 |
| Python files under src (including package markers) | 64 | 46 |
| Python scripts | 26 | 6 |
| Saved goal model artifacts | 6 | 3 |

## Changes

Removed superseded experiments, incomplete/peripheral notebooks, unused alternate simulation paths, old model artifacts, redundant local backups, and generated audit materials. Cleared historical notebook outputs. Removed the unused dashboard and its eager data reads. Simplified the model factory, extracted pure bracket presentation, shared flag data, removed unused imports, and documented the retained responsibilities.

Application, development, and notebook dependencies are separated and pinned to the tested Python 3.11 environment. A GitHub Actions test workflow is included; it has not run remotely yet.

## Verified locally

- 63 automated tests passed, including saved-model and bracket regressions.
- Nine real predictions (three team pairs across V4/V5/V6) match the pre-cleanup snapshot.
- All 54 retained data/model files are byte-identical to the pre-cleanup snapshot.
- All three application pages load for all three model versions in isolated copies.
- Single-match prediction succeeds for V4/V5/V6.
- Full tournament progression through the final succeeds for V4/V5/V6.
- All 137 retained notebook code cells parse; no unresolved explicit internal imports were found.
- Installed dependency checks and Git whitespace checks pass.

Notebook syntax checks do not mean all training cells were executed. Full retraining, a fresh independent benchmark, a clean-machine dependency install, and hosted CI execution were not performed.

## Recovery and publication

A complete working-tree snapshot and Git bundle were saved outside the repository before removal, including previously uncommitted work. The old study PDFs were moved alongside the repository into `study_materials`; their file inventory is historical. The new [study route](STUDY_HE.md) matches the cleaned tree.

No remote branch was pushed or rewritten. Known scientific limitations remain documented. The previously exposed API credential remains in Git history and needs owner-side revocation/rotation before treating the history as safe to publish.
