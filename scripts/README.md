# Scripts

Run from the repository root with the project environment. These are explicit operations, not background services.

| Script | Responsibility | Writes |
| --- | --- | --- |
| `train_v5.py` | Train V5 and fit its calibration | Model, predictor and configuration artifacts |
| `train_v6.py` | Train V6 and fit drawband calibration | V6 model and configuration |
| `evaluate_v5.py` | Historical V5 evaluation protocol | Evaluation outputs |
| `rebuild_2026_features.py` | Rebuild fixture feature rows | Processed 2026 feature table |
| `scraping/scrape_transfermarkt_position_values_all_years.py` | Collect historical position values | Raw CSV |
| `scraping/scrape_transfermarkt_position_values_2026.py` | Collect 2026 position values | Raw CSV |

The Elo collector is `data/scraping/scrape_elo.py`. Collectors require network access and may depend on external page structure. They are not required to run the app.

Training is intentionally not part of application startup. See `docs/DATA.md` and known limitations before retraining; new artifacts may change predictions.
