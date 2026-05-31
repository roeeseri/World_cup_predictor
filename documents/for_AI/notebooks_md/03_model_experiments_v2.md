# Notebook: 03_model_experiments_v2.ipynb

## Purpose
Production-grade model training pipeline with Optuna tuning, tournament weighting, and feature importance analysis. Current general training reference.

## What's Done

### 1. Data Setup
- Loads model_dataset.csv (21,539 matches, 43 columns)
- Chronological train/test split (roughly 80/20 by date)

### 2. Sample Weighting
- Uses `competition_weight` column (World Cup=5.0, Friendly=1.0)
- Models trained with these weights so WC-type matches count more

### 3. Optuna Hyperparameter Tuning (30 trials each)
- `PoissonGoalModel`: tunes alpha (regularization strength)
- `TreeGoalModel`: tunes n_estimators, max_depth
- Metric optimized: exact score accuracy on validation set

### 4. Feature Importance
- Extracts tree model feature importances
- Identifies top predictive features

### 5. Error Analysis
- Where models fail: group stage vs knockout, strong vs weak teams, away vs home
- Anomaly detection: matches with goal_diff > 4 flagged and down-weighted

## Best Results (on general test set)
- TreeGoalModel: 18.36% exact score, 84.53% result accuracy, 0.799 goal MAE
- PoissonGoalModel: lower exact score but better calibrated probabilities
- RandomForest outperforms Poisson significantly on exact score

## Top Features (by tree importance)
rank_diff, elo_diff, rating_a_before, rating_b_before, form_diff_last5, market_value diffs

## Status
Complete. Current general model performance reference. Note: these results are on the full general test set, not WC-specific — see notebook 06 for WC-focused numbers.
