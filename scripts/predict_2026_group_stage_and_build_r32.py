from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.features.feature_columns import FEATURE_COLS
from src.models.score_conversion import most_likely_score
from src.tournament.simulate_world_cup import build_knockout_from_group_predictions


MODEL_PATH = "models/production_model_v4.joblib"
GROUP_FEATURES_PATH = "data/processed/world_cup_2026_group_stage_features.csv"

OUT_GROUP_PREDICTIONS = "outputs/evaluation/world_cup_2026_group_predictions.csv"
OUT_GROUP_STANDINGS = "outputs/evaluation/world_cup_2026_group_standings.csv"
OUT_R32_FIXTURES = "outputs/evaluation/world_cup_2026_round_of_32_fixtures.csv"


def main():
    model = joblib.load(MODEL_PATH)

    fixtures = pd.read_csv(GROUP_FEATURES_PATH)

    for col in FEATURE_COLS:
        if col not in fixtures.columns:
            fixtures[col] = 0.0

    X_2026 = fixtures[FEATURE_COLS].fillna(0)

    preds = model.predict(X_2026)

    scores = [
        most_likely_score(
            float(pred[0]),
            float(pred[1]),
        )
        for pred in preds
    ]

    predictions = fixtures[
        ["match_id", "date", "team_a", "team_b", "group"]
    ].copy()

    predictions["pred_lambda_a"] = preds[:, 0]
    predictions["pred_lambda_b"] = preds[:, 1]
    predictions["pred_goals_a"] = [score[0] for score in scores]
    predictions["pred_goals_b"] = [score[1] for score in scores]
    predictions["pred_score"] = (
        predictions["pred_goals_a"].astype(str)
        + "-"
        + predictions["pred_goals_b"].astype(str)
    )

    standings, r32 = build_knockout_from_group_predictions(predictions)

    Path("outputs/evaluation").mkdir(parents=True, exist_ok=True)

    predictions.to_csv(OUT_GROUP_PREDICTIONS, index=False)
    standings.to_csv(OUT_GROUP_STANDINGS, index=False)
    r32.to_csv(OUT_R32_FIXTURES, index=False)

    print("Saved:")
    print(OUT_GROUP_PREDICTIONS)
    print(OUT_GROUP_STANDINGS)
    print(OUT_R32_FIXTURES)

    print("\nRound of 32:")
    print(r32)


if __name__ == "__main__":
    main()