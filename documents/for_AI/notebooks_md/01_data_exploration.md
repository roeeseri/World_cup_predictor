# Notebook: 01_data_exploration.ipynb

## Purpose
Initial EDA of historical international football results from eloratings.net.

## Data Source
- 26 CSV files: `data/raw/elo_2001_results.csv` through `data/raw/elo_2026_results.csv`
- Each file: one year of international match results with pre-match Elo ratings for both teams

## What's Done
- Loads and merges yearly Elo CSVs into a single DataFrame
- Explores match count per year and competition type distribution
- Analyzes goal distributions (goals_a, goals_b, total, goal_diff)
- Checks Elo rating ranges and distributions
- Identifies data quality issues (missing values, duplicate matches)

## Key Findings
- Dataset spans 2001-2026, ~24k international matches
- Mean goals: ~1.3 (team_a) and ~1.0 (team_b)
- Most common scores: 1-0, 0-0, 1-1, 2-0
- Elo ratings range roughly 1000–2200

## Status
Complete. Exploratory only — no artifacts produced. Feeds understanding for notebook 05.
