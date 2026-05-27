# Team Integration Guide — Data/Features ↔ Models/Evaluation

## Purpose of This Document

This document explains how the two parts of the World Cup score predictor connect:

1. **Data / Features / State** — partner’s side
2. **Models / Evaluation / Prediction** — modeling side

The goal is to make sure both people can work separately without blocking each other, while still producing code that connects cleanly.

The most important idea:

> The data/features side produces a clean pre-match feature table.  
> The modeling side consumes that table, trains models, evaluates them, and predicts future matches.

---

## 1. Big Picture Flow

The system has two main phases:

1. **Training / historical evaluation phase**
2. **Actual 2026 prediction phase**

They are similar, but not identical.

---

## 2. Training / Historical Evaluation Phase

In training, we use historical tournaments.

For every historical match, we want one row that represents what was known **before the match started**.

Example:

```text
Before Argentina vs France:
    Argentina Elo before match
    France Elo before match
    Argentina tournament points before match
    France tournament points before match
    Argentina recent form before tournament
    France recent form before tournament
    Match stage
    Matchday
    Rest days
    etc.

Target:
    Argentina goals
    France goals
```

So the training table should look like:

```text
match_id | tournament | date | team_A | team_B | features... | goals_A | goals_B
```

The model does **not** calculate these features itself. The model receives them already prepared.

---

## 3. Actual 2026 Prediction Phase

During the 2026 World Cup, the logic is similar.

Before a selected match:

```text
1. Current tournament state is loaded.
2. Current pre-match features are created for the selected match.
3. The trained model predicts expected goals.
4. Expected goals are converted to an exact score.
```

After the real match happens:

```text
1. The actual result is entered or loaded.
2. Team state is updated.
3. Group table is updated.
4. Future predictions use the updated state.
```

---

## 4. Responsibility Split

## Partner: Data / Features / State Owner

The partner owns the code that creates correct model-ready data.

Main responsibility:

> Produce one clean feature row per match, using only information available before kickoff.

### Partner should handle:

```text
- loading raw match results
- loading Elo/ranking data
- loading tournament fixtures
- cleaning team names
- creating team IDs for joins/display
- separating matches by tournament
- sorting matches chronologically inside each tournament
- calculating pre-tournament form
- initializing team state before each tournament
- updating team state after each match
- building pre-match feature rows
- saving the final model dataset
```

### Partner-owned folders/files:

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
scrape_elo.py
```

---

## Modeling Side: Models / Evaluation / Prediction Owner

The modeling side owns model training, score conversion, metrics, and comparison.

Main responsibility:

> Consume the feature table, train models, evaluate them, and produce predictions.

### Modeling side should handle:

```text
- defining model input columns
- defining target columns
- implementing baseline models
- implementing ML/statistical models
- converting expected goals to exact score
- calculating evaluation metrics
- comparing models
- saving trained models
- saving evaluation reports
- creating a demo run with mock data until real features are ready
```

### Modeling-owned folders/files:

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

---

## 5. What Partner Needs to Hand Over

The most important handoff is a clean model dataset.

Recommended file:

```text
data/processed/model_dataset.csv
```

This file should contain one row per pre-match snapshot.

### Required metadata columns

These columns are useful for tracking, filtering, debugging, and display.

They should **not** be used as model features.

```text
match_id
tournament
date
stage
team_A
team_B
team_A_id          optional but useful
team_B_id          optional but useful
is_knockout
matchday_in_group
```

### Required target columns

These are what the model learns to predict.

```text
goals_A
goals_B
```

### Required feature columns

The exact feature list can change, but all model feature columns should be numeric.

A good first version can include:

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

is_group_stage
is_knockout
matchday_in_group
rest_days_A
rest_days_B
rest_days_diff
```

The model can start with fewer columns. The important thing is that the agreed feature columns exist in the final dataset.

---

## 6. What Partner Needs to Implement

The exact internal implementation is flexible, but these functions are useful as clean connection points.

## Data loading functions

Located in:

```text
src/data/
```

Useful functions:

```python
load_historical_results(path) -> pd.DataFrame
load_elo_ratings(path) -> pd.DataFrame
load_fixtures(path) -> pd.DataFrame
validate_match_schema(matches_df) -> None
```

These functions should return clean DataFrames with consistent team names, dates, tournaments, and scores.

---

## Feature-building functions

Located in:

```text
src/features/
```

Important functions:

```python
build_pre_tournament_features(team, tournament_start_date, history_df, elo_df) -> dict
```

This should calculate features known before the tournament starts, for example:

```text
Elo before tournament
Elo change in last 365 days
last 10 matches goals for
last 10 matches goals against
recent win rate
average opponent Elo
```

```python
build_tournament_state_features(team, team_state) -> dict
```

This should calculate features from the current tournament state before the match, for example:

```text
tournament matches played
tournament points
tournament goals for
tournament goals against
tournament goal difference
```

```python
build_match_context_features(match, group_table=None) -> dict
```

This should calculate match-level features, for example:

```text
stage
is group stage
is knockout
matchday in group
rest days
```

```python
build_match_features(match, team_state, pre_tournament_data) -> pd.DataFrame
```

This is one of the most important functions.

It should create one feature row for one match using:

```text
match information
team A pre-tournament data
team B pre-tournament data
team A current tournament state
team B current tournament state
match context
```

```python
build_training_dataset(matches_df, history_df, elo_df) -> pd.DataFrame
```

This should loop through historical tournaments chronologically and create the final model dataset.

---

## State functions

Located in:

```text
src/state/
```

Useful functions:

```python
initialize_team_state(teams) -> dict
```

Creates empty tournament state before the tournament starts.

Example initial values:

```text
matches_played = 0
points = 0
goals_for = 0
goals_against = 0
goal_diff = 0
wins = 0
draws = 0
losses = 0
```

```python
update_state_after_match(team_state, match_result) -> dict
```

Updates the state after a real historical/live match result.

```python
initialize_group_table(groups) -> pd.DataFrame
update_group_table(group_table, match_result) -> pd.DataFrame
```

Used for group position and qualification context if needed.

---

## 7. How Modeling Side Uses Partner’s Output

The modeling side should not care how the features were created.

It only needs the final dataset:

```python
import pandas as pd

feature_df = pd.read_csv("data/processed/model_dataset.csv")
```

Then:

```python
FEATURE_COLUMNS = [
    "elo_A",
    "elo_B",
    "elo_diff",
    "recent_goals_for_A",
    "recent_goals_for_B",
    "recent_goals_for_diff",
    # etc.
]

TARGET_COLUMNS = ["goals_A", "goals_B"]

X = feature_df[FEATURE_COLUMNS]
y = feature_df[TARGET_COLUMNS]
```

Then train:

```python
model = train_model(X_train, y_train, model_type="tree")
```

Then predict:

```python
y_pred_expected = model.predict(X_test)
```

Then convert to exact score:

```python
y_pred_scores = convert_expected_goals_to_scores(y_pred_expected, method="poisson")
```

Then evaluate:

```python
summary = evaluate_predictions(y_test, y_pred_expected, y_pred_scores)
```

---

## 8. What Modeling Side Needs to Implement

## Model files

Located in:

```text
src/models/
```

Recommended model classes:

```python
ConstantScoreBaseline
AverageGoalsBaseline
EloHeuristicBaseline
TreeGoalModel
PoissonGoalModel
EnsembleGoalModel      optional later
```

Every model should support:

```python
fit(X, y)
predict(X)
```

`predict(X)` should return:

```text
expected_goals_A, expected_goals_B
```

Shape:

```text
(n_matches, 2)
```

Models should output floats, not rounded scores.

---

## Score conversion file

Located in:

```text
src/prediction/score_conversion.py
```

Important functions:

```python
round_expected_goals(lambda_a, lambda_b) -> tuple[int, int]
poisson_score_grid(lambda_a, lambda_b, max_goals=6) -> pd.DataFrame
most_likely_score(lambda_a, lambda_b, max_goals=6) -> tuple[int, int]
outcome_probabilities(lambda_a, lambda_b, max_goals=6) -> dict
convert_expected_goals_to_scores(y_pred_expected, method="poisson") -> np.ndarray
```

The model predicts expected goals. This file converts expected goals into user-facing scores.

---

## Evaluation files

Located in:

```text
src/evaluation/
```

Important functions:

```python
goal_mae_raw(y_true, y_pred_expected)
goal_rmse_raw(y_true, y_pred_expected)
rounded_score_mae(y_true, y_pred_scores)
exact_score_accuracy(y_true, y_pred_scores)
result_accuracy(y_true, y_pred_scores)
goal_difference_mae(y_true, y_pred_scores)
winner_aware_error(y_true, y_pred_scores, alpha=0.5)
evaluate_predictions(y_true, y_pred_expected, y_pred_scores)
compare_models(models, X_test, y_test, score_method="poisson")
```

The evaluation should report both:

```text
raw expected-goal quality
```

and:

```text
final exact-score / result quality
```

---

## 9. How Training Should Work

Training should be simple.

First, load the final dataset:

```python
feature_df = pd.read_csv("data/processed/model_dataset.csv")
```

Then choose train/test split strategy.

For development:

```text
Train on tournaments before World Cup 2022.
Test on World Cup 2022.
```

Example:

```python
train_df = feature_df[feature_df["tournament"] != "World Cup 2022"]
test_df = feature_df[feature_df["tournament"] == "World Cup 2022"]
```

Then:

```python
X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[["goals_A", "goals_B"]]

X_test = test_df[FEATURE_COLUMNS]
y_test = test_df[["goals_A", "goals_B"]]
```

Train models:

```python
models = {
    "average_baseline": AverageGoalsBaseline().fit(X_train, y_train),
    "elo_baseline": EloHeuristicBaseline().fit(X_train, y_train),
    "tree": TreeGoalModel().fit(X_train, y_train),
    "poisson": PoissonGoalModel().fit(X_train, y_train),
}
```

Evaluate:

```python
comparison = compare_models(models, X_test, y_test, score_method="poisson")
```

Save results:

```python
comparison.to_csv("outputs/evaluation/model_comparison.csv", index=False)
```

---

## 10. Why the Historical Dataset Must Already Contain Pre-Match State

For standard model training, the model should receive a normal feature table.

That means this is best:

```text
model_dataset.csv already contains tournament_points_A before the match
model_dataset.csv already contains tournament_goals_for_A before the match
model_dataset.csv already contains recent form before the match
```

This is better than trying to calculate rolling state during model training.

Why?

```text
- simpler model code
- easier debugging
- easier to inspect the dataset
- easier to avoid data leakage
- easier to compare models
```

So yes: for training, the partner should produce a predefined/precalculated state for each match.

The modeling side should trust that the feature rows are already correct, but should still run sanity checks.

---

## 11. Sanity Checks Modeling Side Should Run on Partner’s Dataset

Before training, check:

```text
- required columns exist
- target columns exist
- feature columns are numeric
- no missing values in required features
- no negative goals
- no duplicate match_id/team_A/team_B rows unless team-swapping is intentional
- World Cup 2022 rows exist for testing
- feature values look reasonable
```

Useful function:

```python
validate_model_dataset(feature_df, feature_columns, target_columns)
```

This can live in:

```text
src/evaluation/evaluate.py
```

or:

```text
src/data/validation.py
```

If placed in `src/data/validation.py`, coordinate with the partner first because that is partner-owned.

---

## 12. How Actual 2026 Prediction Should Work

For actual live use, the flow is slightly different from training.

There may not be a full `model_dataset.csv` row yet for a future match. Instead, we create one feature row on demand.

Expected flow:

```python
match = selected_match
team_state = current_2026_team_state
pre_tournament_data = current_2026_pre_tournament_data

feature_row = build_match_features(match, team_state, pre_tournament_data)

X_match = feature_row[FEATURE_COLUMNS]

expected_goals = model.predict(X_match)
predicted_score = convert_expected_goals_to_scores(expected_goals, method="poisson")
```

Then display:

```text
Team A expected goals
Team B expected goals
Predicted score
A win probability
Draw probability
B win probability
```

After the real match:

```python
team_state = update_state_after_match(team_state, actual_result)
```

This updated state is used for the next prediction.

---

## 13. Overlap Points Between Both Sides

These are the areas where both people need to agree.

## 13.1 Feature column names

The model needs exact column names.

If partner creates:

```text
elo_diff
```

but model expects:

```text
elo_difference
```

the pipeline breaks.

So agree on `FEATURE_COLUMNS` early.

---

## 13.2 Team A / Team B orientation

Both sides must agree what `team_A` and `team_B` mean.

Recommended:

```text
team_A = first listed team in the match row
team_B = second listed team in the match row
```

If team-swapping augmentation is used, then a second row is added:

```text
team_A = original team_B
team_B = original team_A
```

and the targets are swapped too.

---

## 13.3 Team IDs

Team IDs can exist in the dataset, but they are metadata only.

Good:

```text
team_A_id used for joins and debugging
```

Bad for MVP:

```text
team_A_id used as a numeric model feature
```

---

## 13.4 Missing values

Decide how to handle missing values.

Recommended for MVP:

```text
Partner tries to fill missing values during feature creation.
Modeling side validates and refuses to train if required features are missing.
```

Do not silently train on messy missing data.

---

## 13.5 Train/test split

Recommended development split:

```text
Train: tournaments before World Cup 2022
Test: World Cup 2022
```

Later final model:

```text
Train: all available historical tournaments including World Cup 2022
Predict: World Cup 2026
```

---

## 14. Practical Handoff Checklist

Before the modeling side starts using the real data, partner should provide:

```text
[ ] data/processed/model_dataset.csv
[ ] list of columns and meaning
[ ] explanation of which tournaments are included
[ ] confirmation that each row uses only pre-match information
[ ] confirmation of target definition: 90 min or after extra time
[ ] explanation of team_A/team_B ordering
[ ] whether team-swapping augmentation was applied
[ ] known missing-value issues, if any
```

Modeling side should provide partner:

```text
[ ] expected FEATURE_COLUMNS list
[ ] expected TARGET_COLUMNS list
[ ] metadata columns that are allowed but not modeled
[ ] metrics that will be reported
[ ] shape expected from model predictions: (n_samples, 2)
[ ] score conversion method: poisson or round
[ ] required format for a single future match feature row
```

---

## 15. Minimal Integration Example

This is the simplest version of the full connection.

```python
import pandas as pd

from src.models.train import train_model
from src.prediction.score_conversion import convert_expected_goals_to_scores
from src.evaluation.evaluate import evaluate_predictions

FEATURE_COLUMNS = [
    "elo_A",
    "elo_B",
    "elo_diff",
    "recent_goals_for_A",
    "recent_goals_for_B",
    "recent_goals_for_diff",
    "tournament_points_A",
    "tournament_points_B",
    "tournament_points_diff",
    "is_knockout",
    "matchday_in_group",
]

TARGET_COLUMNS = ["goals_A", "goals_B"]

feature_df = pd.read_csv("data/processed/model_dataset.csv")

train_df = feature_df[feature_df["tournament"] != "World Cup 2022"]
test_df = feature_df[feature_df["tournament"] == "World Cup 2022"]

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COLUMNS]

X_test = test_df[FEATURE_COLUMNS]
y_test = test_df[TARGET_COLUMNS]

model = train_model(X_train, y_train, model_type="tree")

y_pred_expected = model.predict(X_test)
y_pred_scores = convert_expected_goals_to_scores(y_pred_expected, method="poisson")

metrics = evaluate_predictions(y_test, y_pred_expected, y_pred_scores)

print(metrics)
```

---

## 16. Simple Summary

Partner produces:

```text
Clean historical pre-match feature dataset
State update functions
Feature-building functions for future matches
```

Modeling side produces:

```text
Models
Metrics
Score conversion
Model comparison
Evaluation reports
```

They connect through:

```text
data/processed/model_dataset.csv
```

for training, and:

```python
build_match_features(match, team_state, pre_tournament_data)
```

for live 2026 predictions.

If those two interfaces are clean, both people can work separately and integrate smoothly.

