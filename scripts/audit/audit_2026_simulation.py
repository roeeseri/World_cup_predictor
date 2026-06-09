from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import pandas as pd

from src.features.feature_columns import FEATURE_COLS
from src.tournament.simulate_world_cup import simulate_world_cup_2026


MODEL_PATH = "models/production_model_v4.joblib"
CONFIG_PATH = "models/production_config_v4.json"
MODEL_DATASET_PATH = "data/processed/model_dataset.csv"
GROUP_FEATURES_PATH = "data/processed/world_cup_2026_group_stage_features.csv"


def main():
    model = joblib.load(MODEL_PATH)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    model_df = pd.read_csv(MODEL_DATASET_PATH)
    model_df["date"] = pd.to_datetime(model_df["date"])

    fixtures = pd.read_csv(GROUP_FEATURES_PATH)

    print("=" * 80)
    print("FEATURE CHECK")
    print("=" * 80)
    print("config == FEATURE_COLS:", config["feature_cols"] == FEATURE_COLS)
    print("missing in fixtures:", [c for c in FEATURE_COLS if c not in fixtures.columns])
    print("missing in model_df:", [c for c in FEATURE_COLS if c not in model_df.columns])

    results = simulate_world_cup_2026(
        model=model,
        model_df=model_df,
        group_features=fixtures,
    )

    gp = results["group_predictions"]
    standings = results["standings"]
    ko = results["knockout_results"]

    print("\n" + "=" * 80)
    print("GROUP PREDICTION CHECK")
    print("=" * 80)
    print("rows:", len(gp))
    print("avg lambda a/b:", round(gp["lambda_a"].mean(), 3), round(gp["lambda_b"].mean(), 3))
    print("avg goals a/b:", round(gp["pred_goals_a"].mean(), 3), round(gp["pred_goals_b"].mean(), 3))
    print("\nscore distribution:")
    print(gp["pred_score"].value_counts().head(20))

    print("\nGroup standings:")
    print(
        standings[
            ["group", "position", "team", "points", "goal_diff", "goals_for"]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("KNOCKOUT CHECK")
    print("=" * 80)
    print(
        ko[
            [
                "round",
                "match_slot",
                "team_a",
                "team_b",
                "lambda_a",
                "lambda_b",
                "pred_score",
                "team_a_win_prob",
                "draw_prob",
                "team_b_win_prob",
                "winner",
            ]
        ].to_string(index=False)
    )

    print("\nKnockout score distribution:")
    print(ko["pred_score"].value_counts())

    print("\nChampion:", results["champion"])
    print("Runner-up:", results["runner_up"])
    print("Third:", results["third_place"])

    suspicious = ko[
        (
            (ko["winner"] == ko["team_a"])
            & (ko["team_a_win_prob"] < ko["team_b_win_prob"])
        )
        |
        (
            (ko["winner"] == ko["team_b"])
            & (ko["team_b_win_prob"] < ko["team_a_win_prob"])
        )
    ]

    print("\n" + "=" * 80)
    print("SUSPICIOUS MATCHES")
    print("=" * 80)

    if suspicious.empty:
        print("No winner/probability contradictions found.")
    else:
        print(
            suspicious[
                [
                    "round",
                    "team_a",
                    "team_b",
                    "pred_score",
                    "team_a_win_prob",
                    "draw_prob",
                    "team_b_win_prob",
                    "winner",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
