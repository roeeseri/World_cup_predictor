# Project Summary: WC Score Predictor

## What This Is
A supervised ML system that predicts exact scores of football matches, optimized for World Cup 2026 prediction. Outputs: (goals_a, goals_b) as expected goals → converted to discrete score via Poisson probability grid.

## Two-Partner Structure

| Partner | Owns | Responsible For |
|---|---|---|
| Models/Evaluation | src/models, src/evaluation, src/prediction, notebooks 03-04, 06_optimization | Training models, evaluation, score conversion, hyperparameter tuning |
| Data/Features/State | src/data, src/features, src/state, notebooks 01-02, 05, 06_scrape | Data loading, feature engineering, tournament state for live use |

---

## What's Done

### Data (Complete for training)
- 26 yearly Elo CSV files (2001–2026): ~24k international match results with pre-match ratings
- Transfermarkt squad market values scraped and cleaned (2004–2026, ~85% coverage)
- `data/processed/model_dataset.csv`: 21,539 matches × 43 columns, all features pre-computed, no leakage
- Feature dictionary documented

### Models (Mostly complete)
- Implemented: AverageGoals, EloHeuristic, PoissonRegression, RandomForest, XGBoost, LightGBM, Ensemble
- Sample weighting: competition importance (WC=5×, Friendly=1×) × temporal decay (0.95/year)
- Hyperparameter tuning: Optuna (60 trials LightGBM, 50 XGBoost)
- Score conversion: Poisson probability grid → most_likely_score
- Evaluation: 7 metrics (exact score %, result %, goal MAE, winner-aware error, etc.)

### Evaluation (Complete)
- Chronological backtest framework
- WC-specific evaluation with expanding-window splits
- 2022 WC backtest completed as reference

### Prediction (Implemented, needs live feature integration)
- `score_conversion.py`: functional, used in all experiments
- `predict_match.py`: implemented but depends on feature building stubs

---

## Current Best Results (2022 WC, 64 matches)
- Baseline (average goals): ~15% exact score
- Poisson (WC-only training): ~29%
- LightGBM (all data + weights + Optuna): ~30–35%
- Ensemble (tuned LGB + XGB + Poisson): ~30–38%
- Target: 40% exact score

Note: 64-match test set has high variance. Numbers fluctuate across runs.

---

## Current Training Strategy
Train on ALL historical matches (2004–2026, ~20k) with competition-based sample weights and temporal decay. Use expanding-window WC splits for honest evaluation. Best model: LightGBM (Optuna-tuned) or ensemble. Features: feat_a (27 columns from prepare_feature_sets).

---

## Known Issues (Fixed)
- **Data leakage**: `standardize_goal_columns()` renamed target_goals_a → goals_a, but `prepare_feature_sets()` exclusion list didn't update → target inside features. Fixed in `world_cup_utils.py`.
- **NaN crash on 2006**: 0 training matches → np.average([]) = NaN → Poisson grid crash. Fixed by skipping years with no training data.
- **Optuna refit non-determinism**: `study.best_params` doesn't include hardcoded `random_state`. Fixed by passing `random_state=42` explicitly on refit.

---

## Next Steps: Models/Evaluation Partner
1. Complete Phase 7 experiments (notebook 06): evaluate NN and stacking results
2. Run full cross-year evaluation (all 4 WC years) with best config to confirm generalization
3. Determine best ensemble weights that generalize across years (not overfit to 2022)
4. Save production model: pkl file + feat_a list + scaler (if NN) for live prediction
5. Implement `predict_match.py` to accept a feature row and return prediction + probabilities
6. Define final model interface that data/features partner must target

## Next Steps: Data/Features/State Partner
1. Implement `src/data/load_fixtures.py`: load 2026 WC fixtures from `data/raw/fifa_2026.txt`
2. Implement `src/data/load_elo.py`: load current Elo ratings for all 2026 WC teams
3. Implement `src/features/pre_tournament_features.py`: compute all 27 feat_a features for any (team_a, team_b, date) pair
4. Implement `src/features/tournament_state_features.py`: compute points_diff, goal_diff_diff from live state
5. Implement `src/state/team_state.py`: initialize and persist tournament state dict
6. Implement `src/state/group_table.py`: group standings, points_diff and goal_diff_diff helpers
7. Validate that output of build_pre_match_features() matches feat_a column list exactly

---

## Live Tournament Prediction Flow (Target Architecture)
```
Tournament start:
  team_states = initialize_team_states(wc_teams)
  elo_ratings = load_current_elo_ratings()
  fixtures    = load_tournament_fixtures()

Before each match:
  features  = build_pre_match_features(team_a, team_b, date, team_states, elo_ratings, ...)
  y_pred    = model.predict(features)           # → [[λ_a, λ_b]]
  score     = convert_expected_goals_to_scores(y_pred)[0]
  probs     = outcome_probabilities(λ_a, λ_b)

After match result:
  team_states = update_state_after_match(team_states, result)
```

---

## Interface Contract Between Partners
- Models partner defines: `feat_a` column list (27 columns, exact names and order)
- Features partner must produce: a DataFrame row with exactly those columns, no extras, no targets
- Get definitive feat_a: `prepare_feature_sets(df)[0]` from any loaded df
- NaN is allowed for missing market values (LightGBM handles natively)
- Column order matters: use `.values` indexing in model, not by name
