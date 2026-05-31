# Notebook: 02_feature_engineering.ipynb

## Purpose
Cleans and normalizes Transfermarkt squad market value data so it can be joined with match data as features.

## Data Source
Raw Transfermarkt scrape produced by `06_scrape_transfermarkt_market_values.ipynb`.

## What's Done
- Loads raw scraped market value data (team × year → squad value in €M)
- Handles currency normalization
- Maps Transfermarkt team names to standardized names used in the Elo dataset
- Fills missing years via interpolation and forward-fill
- Creates 8 normalized market value variants per team:
  - `log_market_value`: log(squad value)
  - `log_market_value_year_centered`: value relative to same-year global mean
  - `market_value_rel_mean`: value as fraction of yearly mean
  - `market_value_zscore`: z-score within year
  - `avg_player_value`: squad value / squad size
- Saves to `data/processed/transfermarkt_market_values_clean.csv`

## Output
`data/processed/transfermarkt_market_values_clean.csv` — used in notebook 05 to join market value features onto match records.

## Coverage
~85% of matches from 2004-2026 have market value data. Matches before 2004 or with unmapped team names have NaN market value features.

## Status
Complete. Output file exists.
