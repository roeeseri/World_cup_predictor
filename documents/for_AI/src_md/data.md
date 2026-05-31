# src/data/ — Data Loading for Live Tournament

## Status: ALL FILES ARE STUBS. Nothing is implemented.

---

## What Needs to Be Implemented

### load_results.py
Used to provide historical match context for feature computation.

```
load_historical_results() → pd.DataFrame
  Columns required: date, team_a, team_b, goals_a, goals_b, competition,
                    rating_a_before, rating_b_before, tournament_year
  Source: data/raw/elo_YEAR_results.csv files (already on disk)
  Returns all matches from 2004 onward in chronological order
```

### load_elo.py
Used to get current pre-match ratings for upcoming fixtures.

```
load_current_elo_ratings() → dict[str, float]
  Returns {team_name: elo_rating}
  Must reflect ratings as of the date just before the tournament starts
  (or updated after each match if Elo changes during tournament)

load_current_rankings() → dict[str, int]
  Returns {team_name: fifa_rank}
  Used for rank_diff feature
```

### load_fixtures.py
Used to know which matches are coming and when.

```
load_tournament_fixtures(path="data/raw/fifa_2026.txt") → pd.DataFrame
  Columns: date, team_a, team_b, stage, group (nullable for knockout)
  Must handle both group stage and knockout stage formats

get_upcoming_matches(fixtures: pd.DataFrame, completed_match_ids: list) → pd.DataFrame
  Filters to matches not yet played
  Returns next batch to predict
```

### validation.py
Run before any prediction to catch data problems early.

```
validate_feature_row(row: pd.DataFrame, expected_columns: list) → bool
  Checks: all expected columns present, no target columns in row,
          no all-NaN columns, values in expected ranges
  Raises informative ValueError if validation fails
```

---

## Interface Contract
All data loading must produce column names and dtypes matching `model_dataset.csv` schema.
Reference: `documents/Feature Dictionary.md` for full column list.
Reference: `src/models/world_cup_utils.py::prepare_feature_sets()` for exact feat_a columns.

---

## Data Already on Disk
- `data/raw/elo_YEAR_results.csv` (2001–2026): historical match results with Elo ratings
- `data/raw/fifa_2026.txt`: 2026 WC fixture schedule
- `data/processed/model_dataset.csv`: pre-built training dataset (do not re-build for live use)
- `data/processed/transfermarkt_market_values_clean.csv`: squad market values 2004–2026
