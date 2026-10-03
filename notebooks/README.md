# Research notebooks

Eight notebooks remain, each with a distinct role. This is a reading order, not a promise that Run All reproduces the saved models. Stored outputs were cleared so old results cannot be mistaken for a fresh run.

| Notebook | Purpose | Main output / caution |
| --- | --- | --- |
| [01_data_exploration](01_data_exploration.ipynb) | Understand match data and distributions | Descriptive analysis; not weighted training performance |
| [02_feature_engineering](02_feature_engineering.ipynb) | Clean squad market values | Clean market-value table; missing values need interpretation |
| [05_build_model_dataset](05_build_model_dataset.ipynb) | Assemble training features and targets | Writes model dataset; read provenance limitations first |
| [03_model_experiments](03_model_experiments.ipynb) | Compare baseline model families | Historical experiments; repeated selection on test data |
| [06_world_cup_models_optimization](06_world_cup_models_optimization.ipynb) | Explore tuned models and stacking | Expensive research, not an independent final benchmark |
| [08_feature_importance](08_feature_importance.ipynb) | Explain saved V6 feature usage | Gain/split importance, not causal attribution |
| [10_append_new_matches](10_append_new_matches.ipynb) | Rebuild an updated dataset | Rebuilds rather than simply appending; separate output |
| [13_calibration_evaluation](13_calibration_evaluation.ipynb) | Study adaptation during a tournament | Historical calibration evaluation; inspect fold assumptions |

Run from `notebooks/` using the project Python environment. Install `requirements-notebooks.txt`. Some training cells are expensive and write files. Work in a separate checkout before rerunning them.

Data collection lives in `data/scraping/scrape_elo.py` and `scripts/scraping/`. Peripheral API experiments, partial backtests, duplicated threshold comparisons, and superseded tuning notebooks were removed from the active repository after backup.
