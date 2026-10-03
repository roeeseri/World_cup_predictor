# Architecture

## One application entry point

`src/app/streamlit_app.py::main` loads the historical data and squad values, selects a model version, and passes its feature builder and score policy to the selected page. The application does not run notebooks or retrain models.

1. `data/load_results.py` reads yearly Elo match files and reconstructs pre-match values.
2. `features/build_features.py` assembles an ordered feature row using form, ratings, squad values, rest, and tournament context.
3. The saved ensemble predicts expected goals for A and B.
4. `models/score_conversion.py` converts these into a score and probabilities.
5. `state/live_state.py` records completed matches; `state/elo.py` updates Elo. `state/tournament_calibration.py` adapts prediction calibration using observed results.
6. `tournament/group_standings.py` and `build_knockout.py` determine standings and qualification.

## Version boundaries

| Version | Features | Goal model | Score selection |
| --- | --- | --- | --- |
| V4 (default) | `FEATURE_COLS` | Saved ensemble | `most_likely_score` |
| V5 | `FEATURE_COLS_V5_PROD` | Saved ensemble | `most_likely_score_v5` |
| V6 | Same feature set as V5 | Corrected mirror classes in `goal_models_v6.py` | `most_likely_score_v6`, configuration-driven drawband |

The serialized models depend on their original Python class import paths. Preserve these paths when refactoring. The V6 mirror correction and all score policies are intentionally unchanged by cleanup.

## Research versus inference

Baseline, Poisson, tree, Optuna, weighting, and evaluation modules remain because the retained notebooks and training routes use them. They explain how models were compared and selected; they are not all executed for a prediction. Removing them would erase the retained research workflow.

The obsolete generic trainer was removed; there is no verified exact recipe for reproducing the saved V4 artifact. `scripts/train_v5.py` and `train_v6.py` retain the later training procedures. Their historical validation protocol has documented time-ordering limitations.

## State and side effects

Feature builders read history and the supplied state. Live result submission persists CSV files under `data/raw/world_cup_updates/`. Simulation uses session state; it is not a Monte Carlo estimate of championship probabilities. Training and collection write artifacts explicitly; launch them from the repository root.

## Scope of cleanup

Removed unreachable alternate simulation/evaluation paths, obsolete model artifacts V1–V3, exploratory scripts, ten redundant or peripheral notebooks, old reports, and local backup copies. Removed the unused legacy dashboard and its unnecessary data reads. No active model bytes, score rules, fixture results, or numerical feature logic were changed. Earlier study PDFs describe the pre-cleanup tree and are retained outside this repository.

`app/bracket_view.py` owns pure bracket HTML and slot resolution. It is separate from live persistence and session-state handling. Shared flag data avoids duplicate team maps.
