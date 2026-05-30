# Feature Dictionary — World Cup Score Predictor

Official dataset:

`data/processed/model_dataset.csv`

Date range:

`2004+`

Targets:

- `target_goals_a`: goals scored by team A
- `target_goals_b`: goals scored by team B
- `target_goal_diff`: team A goals minus team B goals
- `target_total_goals`: total goals in the match

## Metadata Columns

- `date`: match date
- `team_a`: first team in the match row
- `team_b`: second team in the match row
- `competition`: competition name
- `location`: match location
- `season_id`: match year, used for Transfermarkt joins
- `tournament_year`: year of the competition
- `tournament_key`: competition + year identifier

## ELO / Ranking Features

- `rating_a_before`: team A ELO before kickoff
- `rating_b_before`: team B ELO before kickoff
- `elo_diff`: `rating_a_before - rating_b_before`
- `rank_diff`: team A rank before match minus team B rank before match

## Transfermarkt Market Value Features

- `log_market_value_a`: log-transformed squad market value for team A
- `log_market_value_b`: log-transformed squad market value for team B
- `log_market_value_diff`: team A minus team B log market value
- `log_market_value_year_centered_a`: team A log market value centered by yearly average
- `log_market_value_year_centered_b`: team B log market value centered by yearly average
- `log_market_value_year_centered_diff`: difference between yearly-centered log market values
- `market_value_rel_mean_a`: team A squad value divided by yearly average squad value
- `market_value_rel_mean_b`: team B squad value divided by yearly average squad value
- `market_value_rel_mean_diff`: difference between relative yearly market values
- `market_value_zscore_diff`: difference between yearly market value z-scores
- `avg_player_value_diff`: team A average player value minus team B average player value

## Recent Form Features

- `form_diff_last5`: difference in average points from the last 5 matches
- `weighted_goals_for_diff_last5`: recent goals scored, weighted by opponent strength
- `weighted_goals_against_diff_last5`: recent goals conceded, weighted by opponent strength
- `opponent_strength_diff_last5`: difference in average opponent ELO over last 5 matches
- `rating_change_diff_last5`: difference in average ELO change over last 5 matches

## Match Activity / Rest Features

- `team_a_matches_played_before`: number of historical matches team A played before this match
- `team_b_matches_played_before`: number of historical matches team B played before this match
- `team_a_days_since_last_match`: days since team A last played, clipped to 60
- `team_b_days_since_last_match`: days since team B last played, clipped to 60
- `days_since_match_diff`: team A rest days minus team B rest days

## Tournament State Features

All tournament state features are calculated before kickoff.

- `team_a_tournament_matches_played`: team A matches already played in current tournament
- `team_b_tournament_matches_played`: team B matches already played in current tournament
- `tournament_points_diff`: team A tournament points minus team B tournament points
- `tournament_goal_diff_diff`: team A tournament goal difference minus team B tournament goal difference

## Match Context Features

- `competition_weight`: manual competition importance score
  - World Cup: 5.0
  - Major continental tournaments: 4.0
  - Nations League: 2.5
  - Qualifiers: 2.0
  - Friendlies: 0.5
  - Other competitions: 1.5
- `is_home_adv`: 1 if team A is listed as playing at home/location match, otherwise 0

## Notes

- All feature rows are intended to represent information available before kickoff.
- Transfermarkt values are normalized by season to reduce market inflation effects.
- The modeling side should use only numeric feature columns for training.
- Metadata columns should be used for filtering, debugging, and display, not as direct model features.
