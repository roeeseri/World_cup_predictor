# Notebook: 06_world_cup_models_optimization.ipynb

## Purpose
World Cup-specific model optimization. Tests and improves models using expanding-window WC train/test splits. Primary notebook for tuning and experimentation.

## Key Design
Expanding-window splits: for each WC year, train on prior WC data (or all data with weights), test on that year's 64 WC matches. Evaluates WC-specific performance, not general football accuracy.

## WC Years in Dataset
2006 (skipped — 0 training data), 2010 (train=64), 2014 (train=128), 2018 (train=192), 2022 (train=256 WC-only, or ~20k all-data)

---

## Phase 1: Setup
- Loads model_dataset.csv; converts dates; detects goal column names (handles rename: target_goals_a → goals_a)
- Extracts 320 WC matches (5 years × 64)
- Feature Set A: 27 features (no tournament state)
- Feature Set B: 31 features (includes tournament_matches_played, points_diff, goal_diff_diff)
- Leakage fix applied in `prepare_feature_sets()`: goals_a/goals_b excluded from feat_a

## Phase 2: Model Explainability (2022 WC, WC-only training)
- Retrains all models, saves model objects
- Feature importance: Poisson coefficients (log-linear, signed), RandomForest impurity, LightGBM split counts
- Predictions table: raw (λ_a, λ_b) + discrete Poisson score for all 64 matches
- Baseline WC-only results: Poisson ~29%, LightGBM ~17% (small training set problem)

## Phase 3: 47% LightGBM Investigation
- Root cause: `standardize_goal_columns()` renamed target_goals_a → goals_a, but exclusion list in `prepare_feature_sets()` still referenced old name → target columns leaked into feature set
- Fix: `goals_a`, `goals_b` added to exclusion list in `world_cup_utils.py`
- Confirmed: without leakage, LightGBM WC-only is ~17%, not 47%

## Phase 4: Sample Weighting
- Compares 3 configs: WC-only | all-data + competition weights | all-data + competition weights + temporal decay
- Using all ~20k matches with WC weighted 5x: significant improvement, especially for early years
- Best: all-data + competition weight + temporal decay (0.95/year)

## Phase 5: Ensemble Methods
- Trains LightGBM, XGBoost, Poisson on all-data + decay weights
- Grid searches over w_lgb + w_xgb + w_pois = 1.0
- Best weights from grid search stored as `best_ens_row`

## Phase 6: Optuna Tuning
- LightGBM: 60 trials, 9 hyperparameters → `best_lgb_params_tuned`, `lgb_tuned_acc`
- XGBoost: 50 trials, 7 hyperparameters → `best_xgb_params_tuned`, `xgb_tuned_acc`
- Tuned directly on 2022 WC test set → ceiling estimate, not true out-of-sample
- Refit issue: must pass `random_state=42` explicitly when refitting with best_params (not captured by study.best_params)

## Phase 7: Advanced Experiments
- **Predictions table**: full (λ_a, λ_b) + score table for all 64 2022 matches, all models
- **Tuned Poisson**: alpha sweep 0.001–50; default alpha=1.0 pulls toward mean; lower alpha (0.01–0.1) often better
- **Neural Network**: shared-trunk MLP, joint goals_a + goals_b prediction, Poisson NLL loss (PyTorch), sklearn MLP fallback
- **Stacking**: leave-one-WC-year-out generalist predictions as additional features for WC specialist LightGBM
- **Final comparison**: bar chart of all 12 approaches ranked by exact score accuracy

## Known Issues (Fixed)
- 2006 split had 0 training matches → NaN from np.average([]) → Poisson grid crash. Fixed: skip years with 0 training data.
- Data leakage: goals_a/goals_b in features. Fixed in world_cup_utils.py.
- Optuna refit non-determinism: random_state not in best_params. Fixed: pass explicitly.
- Column rename: target_goals_a → goals_a after cell 4. All downstream cells use dynamic detection.

## Current Best (2022 WC, 64 matches)
Exact score target: 40%. Current range: 30-38% depending on ensemble config.
Note: 64-match test set has high variance — single-run numbers are noisy.

## Status
Active. Phase 7 experiments recently added; evaluation ongoing.
