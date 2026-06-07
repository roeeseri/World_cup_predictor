from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.features.feature_columns import FEATURE_COLS
from src.features.position_value_features import (
    load_position_values,
    compute_position_value_features,
)


INPUT_PATH = "data/processed/world_cup_2026_group_stage_features.csv"
OUTPUT_PATH = "data/processed/world_cup_2026_group_stage_features.csv"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    pv = load_position_values()

    df["date"] = pd.to_datetime(df["date"])
    df["season_id"] = df["date"].dt.year

    position_rows = []

    for _, row in df.iterrows():
        position_rows.append(
            compute_position_value_features(
                team_a=row["team_a"],
                team_b=row["team_b"],
                year=int(row["season_id"]),
                position_values=pv,
            )
        )

    position_df = pd.DataFrame(position_rows)

    for col in position_df.columns:
        df[col] = position_df[col]

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    metadata_cols = [
        "match_id",
        "date",
        "team_a",
        "team_b",
        "group",
    ]

    output_cols = metadata_cols + FEATURE_COLS

    df = df[output_cols].copy()

    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Shape:", df.shape)
    print("Features:", len(FEATURE_COLS))
    print("Missing:", [c for c in FEATURE_COLS if c not in df.columns])
    print(df.head())


if __name__ == "__main__":
    main()
