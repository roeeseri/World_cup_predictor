# Notebook: 01_1_elo_matches_results.ipynb

## Purpose
Loads and processes raw Elo yearly CSV files into a clean consolidated match results table.

## What's Done
- Loads all `data/raw/elo_YEAR_results.csv` files
- Standardizes column names across years
- Merges into single DataFrame
- Validates team orientation (team_a = home/first listed, team_b = away)
- Checks for duplicate matches and data integrity issues

## Output
Clean consolidated match results table with columns: date, team_a, team_b, goals_a, goals_b, competition, rating_a_before, rating_b_before.
Not saved to disk — feeds into notebook 05 (dataset building).

## Status
Complete. Owned by data/features partner.
