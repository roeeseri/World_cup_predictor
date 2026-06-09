from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.tournament.simulate_world_cup import simulate_world_cup_2026


MODEL_PATH = "models/production_model_v4.joblib"
MODEL_DATASET_PATH = "data/processed/model_dataset.csv"
GROUP_FEATURES_PATH = "data/processed/world_cup_2026_group_stage_features.csv"

OUT_DIR = Path("outputs/evaluation/world_cup_2026_simulation")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)

    model_df = pd.read_csv(MODEL_DATASET_PATH)
    model_df["date"] = pd.to_datetime(model_df["date"])

    group_features = pd.read_csv(GROUP_FEATURES_PATH)

    results = simulate_world_cup_2026(
        model=model,
        model_df=model_df,
        group_features=group_features,
    )

    results["group_predictions"].to_csv(OUT_DIR / "group_predictions.csv", index=False)
    results["standings"].to_csv(OUT_DIR / "group_standings.csv", index=False)
    results["r32_fixtures"].to_csv(OUT_DIR / "round_of_32_fixtures.csv", index=False)
    results["knockout_results"].to_csv(OUT_DIR / "knockout_results.csv", index=False)

    print("=" * 80)
    print("WORLD CUP 2026 SIMULATION")
    print("=" * 80)
    print("Champion:", results["champion"])
    print("Runner-up:", results["runner_up"])
    print("Third place:", results["third_place"])

    print("\nFinal:")
    final = results["knockout_results"][results["knockout_results"]["round"] == "FINAL"]
    print(final[["team_a", "team_b", "pred_score", "winner"]].to_string(index=False))

    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
