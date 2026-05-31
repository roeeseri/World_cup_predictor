# Notebook: 05_build_model_dataset.ipynb

## Purpose
Builds the primary training dataset: `data/processed/model_dataset.csv` (21,539 matches × 43 columns).

## What's Done

### 1. Base matches
- Loads all Elo yearly CSVs (2001-2026)
- Filters to 2004+ (market value data available from this year)

### 2. Joins Elo/ranking features per match
- rating_a_before, rating_b_before (Elo before match)
- elo_diff = rating_a - rating_b
- rank_diff = rank_a - rank_b (FIFA ranking)

### 3. Joins Transfermarkt market value features (from processed file)
- 8 variants: log_market_value_a/b, log_market_value_year_centered_a/b, market_value_rel_mean_a/b, market_value_zscore_diff, avg_player_value_diff

### 4. Computes rolling form features (last 5 matches per team, pre-match)
- form_diff_last5, weighted_goals_for_diff_last5, weighted_goals_against_diff_last5, opponent_strength_diff_last5, rating_change_diff_last5

### 5. Computes match activity features
- team_a/b_matches_played_before, team_a/b_days_since_last_match, days_since_match_diff

### 6. Computes tournament state features (only from completed matches in same tournament)
- team_a/b_tournament_matches_played, tournament_points_diff, tournament_goal_diff_diff

### 7. Adds context features
- competition_weight: 0.5–5.0 based on competition type (World Cup=5.0, Friendly=0.5)
- is_home_adv: 1 if team_a has home advantage

### 8. Adds targets
- target_goals_a, target_goals_b, target_goal_diff, target_total_goals

## Output Schema
43 columns: 9 metadata + 30 numeric features + 4 target columns
All features computed using only pre-match information — no leakage.

## Critical Notes
- team_a = home/first team, team_b = away/second team (consistent orientation)
- Market value coverage ~85%; remaining rows have NaN market value features (models handle via imputation or NaN-safe algorithms)
- Tournament state features are 0 for each team's first match in that tournament

## Status
Complete. Output file at `data/processed/model_dataset.csv` is the primary training input for all models.
