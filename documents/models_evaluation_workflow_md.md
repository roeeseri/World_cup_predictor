# Models & Evaluation Workflow — World Cup Score Predictor

## Purpose

This document defines the workflow for the modeling/evaluation branch of the `WC_Predictor` project.

The project is a dynamic supervised World Cup score-prediction system. It is **not** reinforcement learning. Each historical match should eventually become a **pre-match feature snapshot**, meaning the row contains only information that would have been known before kickoff.

The model/evaluation owner should build infrastructure that can run before the final feature-engineering pipeline is complete. Until the real feature table exists, use a mock feature table with the same expected shape.

---

## 1. Git Branch Workflow

Work on a separate branch:

```bash
git checkout main
git pull origin main
git checkout -b feature/models-evaluation
```

Commit small, logical changes:

```bash
git add .
git commit -m "Add model evaluation metrics"
git push origin feature/models-evaluation
```

Before merging:

```bash
git checkout main
git pull origin main
git checkout feature/models-evaluation
git merge main
```

Resolve conflicts if needed, run tests, then push again.

---

## 2. Folder Ownership Rules

The goal is to avoid merge conflicts and accidental overwriting of another person's work.

### Person A — Models/Evaluation Owner

Allowed to freely edit:

```text
src/models/
src/evaluation/
src/prediction/score_conversion.py
notebooks/03_model_experiments.ipynb
notebooks/04_2022_backtest.ipynb
models/saved/
models/model_reports/
outputs/evaluation/
outputs/predictions/
tests/test_metrics.py
tests/test_score_conversion.py
```

Allowed to create:

```text
src/models/base.py
src/evaluation/evaluate.py
src/evaluation/mock_data.py
src/evaluation/demo_evaluation.py
tests/test_models_smoke.py
outputs/evaluation/model_comparison.csv
outputs/predictions/mock_predictions.csv
```

Do not edit without coordination:

```text
src/data/
src/features/
src/state/
notebooks/01_data_exploration.ipynb
notebooks/02_feature_engineering.ipynb
notebooks/05_build_model_dataset.ipynb
data/raw/
data/interim/
data/processed/
scrape_elo.py
```

### Person B — Data/Features/State Owner

Allowed to freely edit:

```text
src/data/
src/features/
src/state/
notebooks/01_1_elo_matches_results.ipynb
notebooks/01_data_exploration.ipynb
notebooks/02_feature_engineering.ipynb
notebooks/05_build_model_dataset.ipynb
data/raw/
data/interim/
data/processed/
data/external/
scrape_elo.py
tests/test_feature_leakage.py
tests/test_state_update.py
```

Do not edit without coordination:

```text
src/models/
src/evaluation/
src/prediction/score_conversion.py
notebooks/03_model_experiments.ipynb
notebooks/04_2022_backtest.ipynb
```

### Shared / Coordinate Before Editing

These files are shared. Edit only after agreeing or in a small separate PR:

```text
README.md
TODOs.md
requirements.txt
.env.example
.gitignore
src/config.py
src/prediction/predict_match.py
src/app/streamlit_app.py
outputs/figures/
```

Suggested rule:

- If a file is shared, say in the group chat before editing it.
- If both people need it, split the change into two commits or one person edits it.
- Avoid editing the same notebook at the same time. Notebook conflicts are annoying.

---

## 3. Responsibility Boundary: Who Builds the Pre-Match State?

The data/features/state owner should build the actual pre-match feature table.

That includes:

```text
- separating matches by tournament
- deciding which tournaments are training/test tournaments
- creating tournament chronological order
- using the year/two years before the tournament as pre-tournament form
- initializing team state before tournament start
- updating team state after each historical match
- producing one feature row per match using only pre-match information
```

The models/evaluation owner should **not** manually calculate these features.

The models/evaluation owner should:

```text
- define the expected input schema
- build models that consume a feature table
- build evaluation functions
- build chronological backtest logic that can call partner-owned state/feature functions later
- validate that target columns and feature columns exist
- compare models
- save model reports
```

The correct interface is:

```python
feature_df = build_training_dataset(matches_df, history_df, elo_df)
```

Then the modeling side uses:

```python
X = feature_df[FEATURE_COLUMNS]
y = feature_df[["goals_A", "goals_B"]]
```

Until the real `feature_df` exists, use mock data.

---

## 4. Expected Feature Table Contract

The modeling pipeline expects a pandas DataFrame with one row per pre-match snapshot.

Required metadata columns:

```text
match_id
tournament
date
stage
team_A
team_B
```

Required target columns:

```text
goals_A
goals_B
```

Model feature columns should be numeric only.

Recommended starting feature columns:

```text
elo_A
elo_B
elo_diff

elo_change_365_A
elo_change_365_B
elo_change_365_diff

recent_goals_for_A
recent_goals_for_B
recent_goals_for_diff

recent_goals_against_A
recent_goals_against_B
recent_goals_against_diff

recent_goal_diff_A
recent_goal_diff_B
recent_goal_diff_diff

recent_win_rate_A
recent_win_rate_B
recent_win_rate_diff

avg_opponent_elo_last_10_A
avg_opponent_elo_last_10_B
avg_opponent_elo_last_10_diff

tournament_matches_played_A
tournament_matches_played_B
tournament_matches_played_diff

tournament_points_A
tournament_points_B
tournament_points_diff

tournament_goals_for_A
tournament_goals_for_B
tournament_goals_for_diff

tournament_goals_against_A
tournament_goals_against_B
tournament_goals_against_diff

tournament_goal_diff_A
tournament_goal_diff_B
tournament_goal_diff_diff

stage_encoded
is_group_stage
is_knockout
matchday_in_group
rest_days_A
rest_days_B
rest_days_diff
```

The real feature set can be smaller at first. The model code should not assume every optional column exists unless listed in `FEATURE_COLUMNS`.

---

## 5. Team IDs and Team Names in the Model

Team names and team IDs should be kept as metadata, not model features.

Use them for:

```text
- joins
- display
- match selection
- grouping
- state updates
- debugging
```

Do not feed them into the model in the MVP.

Reason:

```text
- team IDs cause memorization
- the model may learn historical team identity instead of football strength
- it reduces generalization to weaker/newer 2026 teams
- it is dangerous with a small dataset
```

Allowed metadata:

```text
team_A
team_B
team_A_id
team_B_id
```

Forbidden MVP features:

```text
team_A_id encoded as number
team_B_id encoded as number
one-hot team names
label-encoded teams
```

Possible later experiment:

Use team identity only as a separate optional model, compare against the no-ID model, and keep it only if it improves chronological test performance without hurting generalization.

---

## 6. Modeling Tasks

Implement models with a common interface:

```python
model.fit(X_train, y_train)
y_pred_expected = model.predict(X_test)
```

Prediction shape must be:

```text
(n_samples, 2)
```

Column 0:

```text
expected_goals_A
```

Column 1:

```text
expected_goals_B
```

Models should output continuous expected goals, not rounded scores.

Rounding and score conversion happen in `src/prediction/score_conversion.py`.

### Required models

Implement:

```text
ConstantScoreBaseline
AverageGoalsBaseline
EloHeuristicBaseline
TreeGoalModel
PoissonGoalModel
```

Optional later:

```text
EnsembleGoalModel
GradientBoosting model
LightGBM/XGBoost model if dependency is approved
```

---

## 7. Score Conversion Tasks

Implement in:

```text
src/prediction/score_conversion.py
```

Required functions:

```python
round_expected_goals(lambda_a, lambda_b) -> tuple[int, int]
poisson_score_grid(lambda_a, lambda_b, max_goals=6) -> pd.DataFrame
most_likely_score(lambda_a, lambda_b, max_goals=6) -> tuple[int, int]
outcome_probabilities(lambda_a, lambda_b, max_goals=6) -> dict
convert_expected_goals_to_scores(y_pred_expected, method="poisson") -> np.ndarray
```

Use Poisson conversion as the preferred method and simple rounding as a fallback.

---

## 8. Evaluation Tasks

Implement in:

```text
src/evaluation/metrics.py
src/evaluation/evaluate.py
```

Required metrics:

```python
goal_mae_raw(y_true, y_pred_expected)
goal_rmse_raw(y_true, y_pred_expected)
rounded_score_mae(y_true, y_pred_scores)
exact_score_accuracy(y_true, y_pred_scores)
result_accuracy(y_true, y_pred_scores)
goal_difference_mae(y_true, y_pred_scores)
winner_aware_error(y_true, y_pred_scores, alpha=0.5)
```

Required helper:

```python
result_label(goals_a, goals_b)
```

Required evaluation wrapper:

```python
evaluate_predictions(y_true, y_pred_expected, y_pred_scores, alpha=0.5) -> dict
```

Required model comparison wrapper:

```python
compare_models(models, X_test, y_test, score_method="poisson") -> pd.DataFrame
```

---

## 9. Mock Data Requirement

Create:

```text
src/evaluation/mock_data.py
```

Purpose:

Use mock data only to test infrastructure while feature engineering is unfinished.

Implement:

```python
generate_mock_feature_table(n_matches=600, random_state=42) -> pd.DataFrame
```

The mock targets should be somewhat related to the mock features, but do not tune models based on synthetic results.

Mock data is only for checking:

```text
- code runs
- shapes are correct
- metrics work
- models fit and predict
- score conversion works
```

---

## 10. Demo Script

Create:

```text
src/evaluation/demo_evaluation.py
```

It should be runnable with:

```bash
python -m src.evaluation.demo_evaluation
```

It should:

```text
1. Generate mock data.
2. Split train/test.
3. Train baseline and model classes.
4. Predict expected goals.
5. Convert expected goals to scores.
6. Compare all models.
7. Print a clean evaluation table.
```

---

## 11. Tests

Add/update:

```text
tests/test_metrics.py
tests/test_score_conversion.py
tests/test_models_smoke.py
```

Tests should check:

```text
- metric values on tiny known examples
- result labels
- winner-aware error
- Poisson grid probabilities are valid
- score conversion returns non-negative integers
- all models fit and return shape (n_samples, 2)
- all model predictions are non-negative
```

Run:

```bash
pytest
```

---

## 12. Integration With Real Feature Table Later

When the real feature pipeline is ready, the mock line:

```python
df = generate_mock_feature_table(n_matches=600)
```

should be replaceable with:

```python
df = pd.read_csv("data/processed/model_dataset.csv")
```

Then use:

```python
X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMNS]
```

The modeling code should not depend on how the features were created.

---

## 13. Chronological Backtest Responsibility

The modeling/evaluation owner can implement a generic chronological backtest wrapper in:

```text
src/evaluation/backtest.py
```

But the actual feature/state calculations should call partner-owned functions.

Expected future flow:

```python
for match in matches_in_chronological_order:
    feature_row = build_match_features(match, team_state, pre_tournament_data)
    prediction = model.predict(feature_row)
    save_prediction(prediction)
    team_state = update_state_after_match(team_state, actual_result)
```

The backtest wrapper can define this flow, but should not duplicate feature-engineering logic.

---

## 14. Do Not Do These Yet

Do not implement:

```text
- scraping
- advanced injury features
- xG scraping
- team ID features
- automatic 2026 live updates
- complex app UI
- full tournament simulation
- heavy hyperparameter tuning on mock data
```

---

## 15. Definition of Done for This Branch

This branch is successful if:

```text
1. `python -m src.evaluation.demo_evaluation` runs successfully.
2. `pytest` passes.
3. At least 4 models can be trained and compared on mock data.
4. Score conversion works using both rounding and Poisson grid.
5. Metrics return a clean comparison table.
6. The modeling code can later consume `data/processed/model_dataset.csv` without major rewriting.
```

---

## 16. Implementation Request

Please implement the models/evaluation workflow described in this document.

Important:

- Stay inside the allowed files/folders for the models/evaluation owner.
- Do not modify partner-owned data/feature/state files.
- Use clean Python and simple interfaces.
- Do not over-engineer.
- Keep all model predictions as expected goals.
- Convert expected goals to exact scores only in the score-conversion layer.
- Use mock data only for infrastructure testing.
- After implementation, list all files changed and explain how to run the demo and tests.

