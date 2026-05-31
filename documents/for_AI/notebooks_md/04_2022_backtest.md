# Notebook: 04_2022_backtest.ipynb

## Purpose
Evaluates model performance on the 2022 FIFA World Cup (Qatar, 64 matches). Tests generalization of historically trained models to a real WC.

## What's Done
- Trains model on all data before Nov 2022 (WC cutoff date)
- Runs chronological backtest through all 64 WC matches
- Predictions made before each match using only information available at that time
- Reports match-by-match predictions vs actuals
- Analyzes group stage vs knockout stage performance separately

## Key Findings
- Knockout matches harder to predict (high variance, often low-scoring draws)
- Group stage predictions better calibrated
- Exact score accuracy on 2022 WC: ~20-30% depending on model and training config
- Result accuracy: ~65-75%

## Status
Complete. Use as reference for 2022 WC performance. For WC-specific optimization experiments, see notebook 06.
