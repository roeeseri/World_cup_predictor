# src/features/ — Feature Building for Live Tournament

## Status: ALL FILES ARE STUBS. Nothing is implemented.

## Context
During training, all features were pre-computed and stored in `model_dataset.csv`.
For live tournament use, features must be computed on demand before each match.
Output must exactly match the feat_a column schema the model was trained on.

---

## Files and What Each Must Implement

### pre_tournament_features.py
Called **once before the tournament starts**. Computes features that don't change during the tournament.

```
compute_elo_features(team_a, team_b, elo_ratings, rankings) → dict
  → rating_a_before, rating_b_before, elo_diff, rank_diff

compute_market_value_features(team_a, team_b, market_values, year) → dict
  → log_market_value_a, log_market_value_b, log_market_value_diff
  → log_market_value_year_centered_a, log_market_value_year_centered_b, log_market_value_year_centered_diff
  → market_value_rel_mean_a, market_value_rel_mean_b, market_value_rel_mean_diff
  → market_value_zscore_diff, avg_player_value_diff

compute_recent_form(team_a, team_b, historical_matches, cutoff_date) → dict
  → form_diff_last5 (weighted recency score over last 5 matches)
  → weighted_goals_for_diff_last5
  → weighted_goals_against_diff_last5
  → opponent_strength_diff_last5
  → rating_change_diff_last5

compute_match_activity(team_a, team_b, historical_matches, cutoff_date) → dict
  → team_a_matches_played_before, team_b_matches_played_before
  → team_a_days_since_last_match, team_b_days_since_last_match
  → days_since_match_diff
```

### tournament_state_features.py
Called **after each match** to update. Computes features that change as tournament progresses.

```
compute_tournament_state_features(team_a, team_b, team_states) → dict
  → team_a_tournament_matches_played
  → team_b_tournament_matches_played
  → tournament_points_diff (team_a_points - team_b_points from group table)
  → tournament_goal_diff_diff (team_a_gd - team_b_gd from group table)

Note: these are 0 for each team's first match (correct behavior — model was trained this way)
Note: for knockout matches, these still reflect group stage performance
```

### match_context_features.py
```
compute_match_context(match, tournament_info) → dict
  → competition_weight: 5.0 for all WC matches (group stage and knockout)
  → is_home_adv: 1 if team_a has geographic/designated home advantage, else 0
```

### build_features.py
Main orchestrator. Single function called before each match prediction.

```
build_pre_match_features(
    team_a: str,
    team_b: str,
    match_date: str,
    team_states: dict,
    historical_matches: pd.DataFrame,
    market_values: dict,
    elo_ratings: dict,
    rankings: dict,
    tournament_info: dict
) → pd.DataFrame  (1 row)

Output columns must match feat_a exactly:
  ['rating_a_before', 'rating_b_before', 'elo_diff', 'rank_diff',
   'log_market_value_a', 'log_market_value_b', 'log_market_value_diff',
   'log_market_value_year_centered_a', 'log_market_value_year_centered_b', 'log_market_value_year_centered_diff',
   'market_value_rel_mean_a', 'market_value_rel_mean_b', 'market_value_rel_mean_diff',
   'market_value_zscore_diff', 'avg_player_value_diff',
   'form_diff_last5', 'weighted_goals_for_diff_last5', 'weighted_goals_against_diff_last5',
   'opponent_strength_diff_last5', 'rating_change_diff_last5',
   'team_a_matches_played_before', 'team_b_matches_played_before',
   'team_a_days_since_last_match', 'team_b_days_since_last_match', 'days_since_match_diff',
   'competition_weight', 'is_home_adv']
```

---

## Critical Constraints
1. Column names must exactly match feat_a. Wrong name = silent wrong prediction.
2. No target columns (goals_a, goals_b, etc.) in output — model will error or use them as features.
3. Missing values: use NaN for unavailable market values (LightGBM handles NaN natively).
4. Verify output with `src/data/validation.py` before passing to model.
5. Get the definitive feat_a list by running `prepare_feature_sets(df)[0]` on any loaded df.
