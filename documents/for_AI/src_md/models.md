# src/models/ — Model Implementations & Training Pipeline

## Status: Implemented and functional.

---

## base.py — Dataset & Feature Utilities
Core utilities used by all models.

- `load_model_dataset(path, standardize)` → pd.DataFrame
  - Loads model_dataset.csv; optionally renames target_goals_a → goals_A
- `infer_feature_columns(df)` → list[str]
  - Returns all numeric columns excluding targets, metadata, and leakage columns
- `coerce_goal_array(y)` → ndarray (n, 2)
  - Accepts DataFrame, ndarray, list → always returns (n_matches, 2) float array
- `ensure_non_negative(predictions)` → ndarray
  - Clips all values to [0, ∞) — required before score conversion
- `build_sample_weight(df)` → ndarray | None
  - Reads `competition_weight` column, normalizes to mean=1.0

---

## baseline.py — Reference Baselines
All implement `.fit(X, y, sample_weight=None)` → `.predict(X)` → ndarray (n, 2).

| Class | Logic |
|---|---|
| AverageGoalsBaseline | Predicts mean(goals_a), mean(goals_b) from training set for every match |
| EloHeuristicBaseline | Shifts mean goals by tanh(elo_diff/400) × scale; uses only elo_diff feature |
| ConstantScoreBaseline | Predicts fixed goals; sanity check |

---

## poisson_model.py — Poisson Regression
`PoissonGoalModel(alpha=1.0, max_iter=1000)`
- Two separate sklearn PoissonRegressor instances (one for goals_a, one for goals_b)
- Log-linear: λ = exp(X @ β); predicts expected goals per team
- alpha: L2 regularization. High α → predictions collapse toward mean. Low α → responsive to features.
- Tested range: 0.001–50; optimal on WC data typically 0.01–0.1 (see notebook 06 Phase 7)

---

## tree_model.py — Random Forest
`TreeGoalModel(n_estimators=100, max_depth=15)`
- Two separate RandomForestRegressor instances
- `.home_model.feature_importances_` and `.away_model.feature_importances_` for analysis
- Best production config: n_estimators=300, max_depth=12

---

## ensemble.py — Weighted Ensemble
`EnsembleGoalModel(models: list, weights: list[float] | None)`
- `predict()`: weighted average of each model's (λ_a, λ_b) predictions before score conversion
- weights=None → equal weighting
- Best WC ensemble: ~0.5 LightGBM + 0.2–0.3 XGBoost + 0.1–0.2 tuned Poisson (from grid search in notebook 06)

---

## weighting.py — Sample Weights
```
COMPETITION_WEIGHTS = {
    'FIFA World Cup': 5.0, 'European Championship': 4.0, 'Copa America': 3.5,
    'World Cup qualifier': 3.0, 'African Nations Cup': 3.0, 'Friendly': 1.0, ...
}

apply_combined_weighting(df, apply_decay=False, decay_rate=0.95, reference_year=2024)
  → ndarray of per-sample weights, normalized to mean=1.0
  Combines competition importance × temporal decay
```

---

## optuna_tuning.py — Hyperparameter Optimization
- `PoissonTuner(X_train, y_train, X_val, y_val)` → `.optimize(n_trials=50)` → best_params, best_model
- `TreeTuner(...)` → same interface
- Optimizes: exact_score_accuracy on validation set
- LightGBM and XGBoost tuned inline in notebook 06 (not via this module)

---

## world_cup_utils.py — WC-Specific Utilities

Key functions:
- `prepare_feature_sets(df)` → (feat_a, feat_b)
  - feat_a: 27 features, no tournament state
  - feat_b: 31 features, includes tournament state columns
  - Excludes: goals_a, goals_b, target_goal_diff, target_total_goals (leakage fix)
- `create_expanding_window_splits(df, years)` → dict[year → (train_df, test_df)]
  - Skips years with 0 prior training matches (2006)
- `evaluate_world_cup_only(y_true, y_pred_scores, y_pred_expected)` → metrics dict
  - Returns exact_score_accuracy, result_accuracy, rounded_score_mae, goal_mae

---

## train_production.py — CLI
```bash
python -m src.models.train_production --model-type tree --temporal-decay
```
- Trains on full dataset with competition weights
- Saves to `models/saved/production_model.pkl` + `model_config.json`

---

## Best Training Configuration (from notebook 06 experiments)
- Data: all historical matches (2004–2026, ~20k)
- Weights: competition_weight × temporal decay (0.95/year)
- Model: LightGBM (Optuna-tuned, 60 trials) or weighted ensemble
- Features: feat_a (27 columns from prepare_feature_sets)
- Score conversion: Poisson grid (most_likely_score)
