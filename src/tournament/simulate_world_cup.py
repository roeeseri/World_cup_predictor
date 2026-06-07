"""World Cup tournament simulation utilities."""

from __future__ import annotations

import pandas as pd

from src.tournament.build_knockout import (
    build_round_of_32_fixtures,
    validate_round_of_32,
)
from src.tournament.group_standings import (
    build_group_standings,
    get_group_position_map,
)


def prepare_group_prediction_matches(group_predictions: pd.DataFrame) -> pd.DataFrame:
    """Normalize group-stage prediction output for standings calculation.

    Expected input columns:
    - group
    - team_a
    - team_b
    - pred_score OR pred_goals_a/pred_goals_b OR goals_a/goals_b

    Returns columns:
    - group
    - team_a
    - team_b
    - goals_a
    - goals_b
    """
    df = group_predictions.copy()

    if {"goals_a", "goals_b"}.issubset(df.columns):
        df["goals_a"] = df["goals_a"].astype(int)
        df["goals_b"] = df["goals_b"].astype(int)

    elif {"pred_goals_a", "pred_goals_b"}.issubset(df.columns):
        df["goals_a"] = df["pred_goals_a"].astype(int)
        df["goals_b"] = df["pred_goals_b"].astype(int)

    elif "pred_score" in df.columns:
        scores = df["pred_score"].astype(str).str.extract(r"(\d+)\s*-\s*(\d+)")
        if scores.isna().any().any():
            raise ValueError("Could not parse some pred_score values.")
        df["goals_a"] = scores[0].astype(int)
        df["goals_b"] = scores[1].astype(int)

    else:
        raise ValueError(
            "group_predictions must include either goals_a/goals_b, "
            "pred_goals_a/pred_goals_b, or pred_score."
        )

    required_cols = ["group", "team_a", "team_b", "goals_a", "goals_b"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[required_cols].copy()


def build_knockout_from_group_predictions(
    group_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build standings and Round of 32 fixtures from group predictions."""
    group_matches = prepare_group_prediction_matches(group_predictions)

    standings = build_group_standings(group_matches)
    position_map = get_group_position_map(standings)

    r32_fixtures = build_round_of_32_fixtures(
        standings=standings,
        position_map=position_map,
    )

    validate_round_of_32(r32_fixtures)

    return standings, r32_fixtures
