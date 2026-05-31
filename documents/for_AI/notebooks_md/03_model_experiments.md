# Notebook: 03_model_experiments.ipynb

## Purpose
First round of model experiments. Establishes baseline numbers before the full production pipeline.

## What's Done
- Loads model_dataset.csv
- Simple chronological train/test split
- Trains and compares: AverageGoalsBaseline, EloHeuristicBaseline, PoissonGoalModel, TreeGoalModel
- Reports: exact score accuracy, result accuracy, goal MAE

## Key Results
- Average baseline: ~15% exact score, ~60% result accuracy
- Poisson: ~20% exact score
- Tree model: outperforms Poisson modestly
- Establishes ~20% exact score as the floor to beat

## Status
Superseded by `03_model_experiments_v2.ipynb`. Historical reference only — do not use these results as current benchmarks.
