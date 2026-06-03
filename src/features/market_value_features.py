"""Transfermarkt market value feature helpers."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.team_names import normalize_team_name


def load_market_values(
    path: str | Path = "data/processed/transfermarkt_market_values_clean.csv",
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing market value file: {path}")

    df = pd.read_csv(path)

    required_cols = [
        "team_name_tm",
        "season_id",
        "log_market_value",
        "market_value_relative_to_year_mean",
        "avg_player_value_millions_eur",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing market value columns: {missing}")

    return df


def get_team_market_row(team: str, year: int, market_values: pd.DataFrame):
    team_tm = normalize_team_name(team)

    match = market_values[
        (market_values["team_name_tm"] == team_tm)
        & (market_values["season_id"] == int(year))
    ]

    if match.empty:
        return None

    return match.iloc[0]


def _safe_value(row, col: str) -> float:
    if row is None:
        return 0.0

    value = row.get(col, 0.0)

    if pd.isna(value) or value in [np.inf, -np.inf]:
        return 0.0

    return float(value)


def compute_market_value_features(
    team_a: str,
    team_b: str,
    year: int,
    market_values: pd.DataFrame,
) -> dict:
    row_a = get_team_market_row(team_a, year, market_values)
    row_b = get_team_market_row(team_b, year, market_values)

    return {
        "log_market_value_a": _safe_value(row_a, "log_market_value"),
        "market_value_rel_mean_diff": (
            _safe_value(row_a, "market_value_relative_to_year_mean")
            - _safe_value(row_b, "market_value_relative_to_year_mean")
        ),
        "avg_player_value_diff": (
            _safe_value(row_a, "avg_player_value_millions_eur")
            - _safe_value(row_b, "avg_player_value_millions_eur")
        ),
    }
