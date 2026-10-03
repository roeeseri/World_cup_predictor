# Data and saved models

## Inputs retained

- `data/raw/elo_*_results.csv`: historical matches and Elo snapshots; loaded by the app and dataset notebooks.
- `data/raw/fixtures/`: fixture sources and normalized 2026 group-stage schedule.
- `data/raw/transfermarkt/`: squad and position-value source snapshots.
- `data/raw/world_cup_updates/`: recorded results, calibration observations, and saved model accuracy records. The snapshot contains 72 group-stage and 30 knockout results.
- `data/processed/model_dataset.csv` and `updated_model_dataset.csv`: separate historical training-table snapshots. They differ and must not be treated as interchangeable.
- Clean squad/position tables, fixture features, and third-place qualification combinations under `data/processed/`.

Raw source snapshots are retained for provenance even when a particular app page does not read them directly. Redundant timestamped backups were removed after a full external backup. No remaining data file was rewritten by cleanup.

## Artifacts

The application loads `production_model_v4.joblib`, `production_model_v5.joblib`, or `production_model_v6.joblib`. V4 is the default. The V6 drawband policy reads `production_config_v6.json`; V4 configuration remains as provenance. V5 uses its existing score-policy defaults.

Older V1–V3 models were not selected by the app and were removed. Saved model bytes and class import paths for V4–V6 are preserved. Version labels do not establish a ranking of accuracy.

## Reproduction limits

Dependency versions in requirements are captured from the verified local Python 3.11 environment. Source data snapshots are included, but there is no claim of bit-for-bit training reproducibility or chronological validity of historical scores. See `KNOWN_LIMITATIONS_HE.md` for data timing, ranking reconstruction, and model-selection issues.

Before a public release, review rights to redistribute third-party data and model artifacts. No repository-wide software license is asserted here: the project contains contributions and external data whose licensing must be established by the owners.
