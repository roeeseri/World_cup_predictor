# WC 2026 Predictor v5 — Calibrated Score Grid + Leak-Free Evaluation

## Context

The production v4 model (90% LightGBM + 10% XGBoost ensemble predicting goal lambdas, 21 features, `models/production_model_v4.joblib`) scores ~30% exact on WC 2022, but:
- 2022 was a tuning CV fold AND the 0.9 score-threshold was selected on 2022 → the 30% is optimistic.
- Tuning objective used `round()`, production uses `floor(λ+0.1)` — mismatch.
- 2026 simulation pathology: ~5 draws in 72 group games, almost no 2-1/3-1-type scores, knockouts mostly draws. Root cause is mostly the **decision rule**, not the model: `floor(λ+0.1)` converts each team's lambda independently, structurally under-producing joint outcomes (1-1, 2-1), and the independent Poisson grid underestimates draws (no Dixon-Coles correction).
- User is scored on **exact final score (incl. extra time)** primarily, W/D/L secondarily. Verified: the dataset already records knockout scores **after 120 min, penalties excluded** (Argentina–France 2022 = 3-3; Croatia–England 2018 = 2-1), so training targets already match the objective; only the prediction-time grid needs an ET mixture.

Budget: 24–30 hours; WC 2026 starts June 11. Rollout: **v5 side-by-side** — v4 artifacts, `score_conversion.py` defaults, and the daily pipeline stay untouched; app flips to v5 only after it wins a held-out comparison.

## Verified facts (by exploration agents, spot-checked)

- Competition strings in `model_dataset.csv`: `World Cup` (2014/2018/2022 = 64 rows each), `European Championship` (2024 = 51 rows), `Copa America` (2024 = 32 rows). All candidate folds exist; Euro/Copa 2021 available as extras.
- Knockout rows identifiable via `(team_a_tournament_matches_played >= 3) | (team_b... >= 3)` — yields exactly 16 KO matches per WC.
- Base rates: group-stage draws ~25% (2014–2024 tournaments); KO 120-min draw (→pens) 20–38%; v4's 5/72 is far below.
- All consumers (`streamlit_app.py:21`, `scripts/simulate_2026_world_cup.py`, `match_simulation.py`, `live_state.py:simulate_forward`, `backtest.py`) import from `score_conversion.py` and hardcode the v4 model path → v5 isolation = new modules + never editing `score_conversion.py` defaults.
- `create_wc_cv_splits` (`src/models/world_cup_utils.py:232`) trains each fold on everything else **including WC 2022** — cannot be reused as-is.
- `metrics.exact_score_accuracy` anomaly-weights blowouts (0.3) — NOT suitable as the harness's primary metric (the real game scores all matches equally). Reuse `rps_batch` as-is.
- `mirror_features` works by column name on DataFrames → a changed v5 feature list works if we always pass DataFrames.

## User-confirmed constraints

- **Dataset**: all v5 training/evaluation runs on `data/processed/updated_model_dataset.csv` (the current dataset incl. latest results), not the older `model_dataset.csv`. Verify at Block 0 that the fold counts (WC 2014/2018/2022, Euro 2024, Copa 2024) and competition strings hold on this file.
- **Decay reference**: temporal decay is always relative to the year of the tournament being predicted — fold year during tuning (2014/2018/2024), 2026 for production.
- **Tournament-context scope**: the in-tournament goals-for/against-per-match features AND the in-tournament lambda adaptation apply **only** to `World Cup`, `European Championship`, `Copa America` rows (a `MAJOR_TOURNAMENTS` constant). All other rows get the features zeroed (same convention as existing tournament-state features) and no lambda blending.

## Evaluation protocol (the leak fix — build FIRST)

```
TUNING_FOLDS = [(World Cup, 2014), (World Cup, 2018),
                (European Championship, 2024), (Copa America, 2024)]
HOLDOUT      = (World Cup, 2022)   # evaluated ONCE per final candidate
```

- Per tuning fold i: train = all rows except fold_i **and except WC 2022**; validate on fold_i. WC 2022 appears in no training set and no tuning loop.
- This answers the user's concern directly: Euro/Copa 2024 are *closer in time to 2026 than 2022 is*, so tuning sees modern-football signal without burning the 2022 test.
- All calibration params are fit on **out-of-fold (OOF) lambdas** with cross-fitting: scoring fold i uses params fit on folds j≠i. For holdout/production: fit on all 4 folds' OOF lambdas.
- Primary metric: **raw unweighted exact-score accuracy**; secondary: outcome accuracy, predicted-vs-actual draw rate, scoreline distribution, RPS, grid log-loss.
- Holdout discipline: every WC-2022 evaluation logged to `outputs/evaluation/v5/holdout_log.json`.
- Final production model afterwards retrains on ALL data (incl. 2022 + 2026 results so far).

## New files (v4 untouched)

| Path | Contents |
|---|---|
| `src/evaluation/protocol.py` | Fold/holdout splits, OOF lambda generation, raw metrics, holdout logging, chronological consistency check (wraps existing `run_chronological_backtest`) |
| `src/models/score_grid.py` | Dixon-Coles joint grid, lambda calibration, decision rule, knockout ET mixture |
| `src/models/predictor_v5.py` | `ScorePredictorV5` — bundles ensemble + calibration params + decision policy; same output dict shape as `predict_match_from_features` |
| `scripts/evaluate_v5.py` | CLI to run the protocol for any candidate config → JSON/CSV in `outputs/evaluation/v5/` |
| `scripts/train_v5.py` | Final retrain on all data + calibration fit → `models/production_model_v5.joblib` + `production_config_v5.json` (template: `train_production.py`) |
| `scripts/simulate_2026_world_cup_v5.py` | Copy of v4 simulate script pointed at v5, ET-aware knockouts |
| `tests/test_score_grid.py` | Grid sums to 1, rho sign behavior, ET mixture mass conservation, argmax sanity |

Small additive edits (v4 path remains default):
- `src/features/feature_columns.py`: add `FEATURE_COLS_V5` + `_DIFF_COLS_V5` (don't touch existing constants).
- `src/app/streamlit_app.py`: sidebar "Model version: v4 (default) / v5" → loads v5 artifact, dispatches to `ScorePredictorV5`.
- `src/tournament/simulate_world_cup.py` + `match_simulation.py`: optional `predictor=None` param; `None` → existing v4 path unchanged; `ScorePredictorV5` → new path with `knockout=True` in KO rounds.

## Core method: calibrated joint score grid (`src/models/score_grid.py`)

No GBM refitting needed for any of this — it operates on the lambdas.

1. **Lambda calibration** (fit on OOF tournament lambdas by Poisson MLE): scale `λ' = c·λ`, and optionally affine in log space `log λ' = a + b·log λ` (b<1 widens/narrows spread — addresses "elo_diff dominates → λa≈λb in knockouts"). Keep whichever is stable across folds.
2. **Dixon-Coles grid**: independent Poisson outer product + standard τ correction on (0,0),(0,1),(1,0),(1,1); renormalize. Fit rho by MLE (`scipy.optimize.minimize_scalar`) on tournament OOF rows. Negative rho inflates 0-0/1-1 → directly fixes the missing draws.
3. **Decision rule**: `pick_score(grid, alpha)` = argmax of `P_exact(i,j) + alpha·P(outcome of (i,j))`. alpha=0 is optimal for pure exact score; tune alpha ∈ {0, 0.05, 0.1, 0.2} on folds for the secondary W/D/L value. Replaces `floor(λ+0.1)` entirely in v5. W/D/L probs come from the same DC grid.
4. **Knockout ET mixture — conditional, not blanket** (targets are 120-min scores): extra time is applied **only to the probability mass where the game is drawn after 90'**. Games the model sees as likely decided in 90' are essentially unchanged. Concretely: `G_final = non-draw cells of G90 (unchanged) + Σ_k G90[k,k] · shift(GET, k)` where `GET = poisson_grid(λa·s, λb·s)`, s = et_scale ≈ 30/90 (tune ∈ {0.25, 0.33, 0.40} on the ~55 KO matches across folds; prefer the 30/90 prior given small n). So a match with P(draw@90)=15% gets only 15% of its mass ET-extended, while a tight match with P(draw@90)=35% can shift its modal final score from 1-1 to 2-1. Residual draw mass at 120 is legitimate (pens-bound games); advancement winner = larger win mass (matches existing tie-break in `simulate_match`).
5. **Per-fold diagnostics**: predicted vs actual draw share, **count and accuracy of "non-basic" scorelines (2-1, 3-1, 3-2, 2-2)** vs base rates, modal-score distribution. The lambda calibration (log-affine, b≠1) plus DC grid argmax is the mechanism that lets 2-1 become the modal pick when λa≈1.8, λb≈1.1 — track that it actually happens on folds.

## Feature changes (kept only if folds improve)

`FEATURE_COLS_V5`:
- **Drop** `team_a/b_matches_played_before` (career counts ~300–400 = noise, per user).
- **Add** `competition_importance` (numeric from `weighting.COMPETITION_WEIGHTS` via the existing `competition` column — currently the model can't tell a WC game from a friendly at inference; tournaments are systematically lower-scoring; constant 4.0 at 2026 prediction time). Plumb into `protocol.py`/`train_v5.py` (training) and `build_pre_match_features`/`build_knockout_feature_row` (inference).
- **Add** `rest_diff` = days_since_last_match diff (sign-flips on mirror); test replacing vs keeping the raw pair.
- **Add in-tournament attack/defense context**: `tournament_goals_for_per_match_diff` and `tournament_goals_against_per_match_diff` (team_states already track goals_for/goals_against/matches — currently only the *net* goal-diff and points reach the model). This gives the model "how is this team actually scoring/conceding in THIS tournament", separate from pre-tournament ELO. Plumb into `compute_tournament_state_features` output + v5 diff-col list; v4's function output is additive (extra keys ignored by v4's FEATURE_COLS).

**Temporal decay weighting (explicit experiment, not an afterthought):** matches closer to 2026 should count more. Sweep through the harness using existing `apply_temporal_decay` / `apply_combined_weighting` (`src/models/weighting.py`) with `reference_year=2026`: decay_rate ∈ {0.85, 0.90, 0.95, 1.0=off} × blend ratio (competition vs decay) ∈ {70/30 (current default), 50/50}. Per-fold caveat: when validating on WC 2014/2018, reference_year = that fold's year (decay must be relative to the tournament being predicted, otherwise the 2014 fold would upweight 2024 data it shouldn't emphasize).

**In-tournament lambda adaptation (calibration layer, addresses "context weight during the tournament"):** beyond features, blend the model's lambda with the team's *observed* in-tournament scoring rate once it has played ≥2 games: `λ_final = w·λ_model + (1−w)·(tournament goals_for / matches, shrunk toward λ_model)`. Fit w by MLE on fold knockout+matchday-3 rows (where tournament context exists). This is exactly "how do the lambdas/means change during the tournament" — it lives in `ScorePredictorV5.predict_lambdas`, costs nothing to v4, and is tuned on folds like every other calibration param. Keep only if folds improve; risk is small-n overfitting, so w is a single global scalar.

Hyperparameters: keep current LGBM/XGB defaults by default (calibration layer absorbs most of the round/floor mismatch damage). If time remains: 50-trial Optuna sweep on the 4 clean folds with objective = **calibrated-grid-argmax exact accuracy** (matching production).

## Stretch (only if ≥4h remain)

1. Multiclass scoreline classifier (LightGBM over top ~20 scorelines + "other", same protocol); promote only if it beats v5 on all folds.
2. WC-only lambda re-scaler (meta-calibration fit on WC rows only).
Skip NN and shots/possession scraping — not enough time/data.

## Time-boxed schedule (24–30h)

| Block | Hrs | Work | Exit criterion |
|---|---|---|---|
| 0 | 1 | Copy this plan into the repo as `docs/PLAN_V5.md` (visible/trackable in VS Code + git). Sanity: reproduce v4's 2022 backtest number through new code path | Baseline reproduced |
| 1 | 4 | `protocol.py` + `evaluate_v5.py` (folds, OOF lambdas, metrics, holdout log, chronological check) | floor(λ+0.1) and plain-Poisson-argmax both scored on 4 clean folds (honest baselines) |
| 2 | 7 | `score_grid.py`: DC grid, rho MLE, lambda calibration, decision rule, ET mixture, unit tests; tune (c/a,b), rho, alpha, et_scale with cross-fitting | Fold metrics + draw-rate diagnostics; v5 conversion config frozen |
| 2b | (within 2) | In-tournament lambda adaptation (blend weight w) tuned on fold matchday-3/KO rows | Kept only if folds improve |
| 3 | 5 | `FEATURE_COLS_V5` + competition_importance + tournament for/against-per-match plumbing; temporal-decay × competition weights sweep; optional matched-objective Optuna | Best feature/weight combo on folds; v5 candidate frozen |
| 4 | 2 | **One-shot holdout**: train candidate on all-except-WC-2022, score 2022 once; same for v4 as comparator | Go/no-go: v5 beats v4 on exact score without big outcome regression |
| 5 | 4 | `train_v5.py` (retrain on ALL data, save artifacts), `simulate_2026_world_cup_v5.py`, app version selector, `predictor` seam; run v4 vs v5 full-2026 simulations side by side | v5 artifacts + side-by-side simulation outputs |
| Buffer | 2–7 | Stretch models, extra folds (Euro/Copa 2021), daily-pipeline `--model-version v5` flag | — |

## Verification

- `tests/test_score_grid.py`: grids sum to 1, DC correction behaves, ET mixture conserves mass.
- Block 0 reproduces v4's known 2022 numbers → new harness is trustworthy.
- Chronological consistency check: stored-row fold evaluation ≈ `run_chronological_backtest` lambdas (validates the live-state path matches what we calibrated).
- Block 4 one-shot holdout = the unbiased v4-vs-v5 comparison.
- Block 5: run both full-2026 simulations; check v5's draw count and 2-1/3-1 share against base rates (~18 draws expected in 72 group games); confirm streamlit app still works with v4 selected (default) and with v5.

## Safety guarantees

- `score_conversion.py`, `production_model_v4.joblib`, daily pipeline, live-state code: never modified.
- All v5 logic in new files; edits to shared files are additive-only with v4 as default behavior.
- WC 2022 enters no training/tuning until the final production retrain.
