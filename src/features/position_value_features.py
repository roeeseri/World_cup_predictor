"""Transfermarkt position-level market value feature helpers."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.team_names import normalize_team_name


POSITION_VALUE_PATH = "data/processed/transfermarkt_position_values_2004_2026.csv"


def load_position_values(
    path: str | Path = POSITION_VALUE_PATH,
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing position value file: {path}")

    df = pd.read_csv(path)

    required_cols = [
        "team_name_tm",
        "season_id",
        "goalkeeper_market_value_millions_eur",
        "defender_market_value_millions_eur",
        "scraped_total_market_value_millions_eur",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing position value columns: {missing}")

    return df


def get_team_position_row(team: str, year: int, position_values: pd.DataFrame):
    team_tm = normalize_team_name(team)

    match = position_values[
        (position_values["team_name_tm"] == team_tm)
        & (position_values["season_id"] == int(year))
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


def _share(value: float, total: float) -> float:
    if total <= 0:
        return 0.0

    return value / total


def compute_position_value_features(
    team_a: str,
    team_b: str,
    year: int,
    position_values: pd.DataFrame,
) -> dict:
    row_a = get_team_position_row(team_a, year, position_values)
    row_b = get_team_position_row(team_b, year, position_values)

    total_a = _safe_value(row_a, "scraped_total_market_value_millions_eur")
    total_b = _safe_value(row_b, "scraped_total_market_value_millions_eur")

    gk_a = _safe_value(row_a, "goalkeeper_market_value_millions_eur")
    gk_b = _safe_value(row_b, "goalkeeper_market_value_millions_eur")

    def_a = _safe_value(row_a, "defender_market_value_millions_eur")
    def_b = _safe_value(row_b, "defender_market_value_millions_eur")

    return {
        "goalkeeper_share_diff": _share(gk_a, total_a) - _share(gk_b, total_b),
        "defender_share_diff": _share(def_a, total_a) - _share(def_b, total_b),
    }
