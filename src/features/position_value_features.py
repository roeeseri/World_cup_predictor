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
        "midfield_market_value_millions_eur",
        "attack_market_value_millions_eur",
        "goalkeeper_avg_age",
        "defender_avg_age",
        "midfield_avg_age",
        "attack_avg_age",
        "scraped_total_market_value_millions_eur",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing position value columns: {missing}")

    return df


def get_team_position_row(
    team: str,
    year: int,
    position_values: pd.DataFrame,
) -> pd.Series | None:
    team_tm = normalize_team_name(team)

    match = position_values[
        (position_values["team_name_tm"] == team_tm) &
        (position_values["season_id"] == int(year))
    ]

    if match.empty:
        return None

    return match.iloc[0]


def _safe_value(row: pd.Series | None, col: str) -> float:
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


def _age_diff(age_a: float, age_b: float) -> float:
    if age_a <= 0 or age_b <= 0:
        return 0.0

    return age_a - age_b


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0

    return numerator / denominator


def _mean_positive(values: list[float]) -> float:
    values = [v for v in values if v > 0]

    if not values:
        return 0.0

    return float(np.mean(values))

def compute_position_value_features(
    team_a: str,
    team_b: str,
    year: int,
    position_values: pd.DataFrame,
) -> dict:
    row_a = get_team_position_row(team_a, year, position_values)
    row_b = get_team_position_row(team_b, year, position_values)

    gk_a = _safe_value(row_a, "goalkeeper_market_value_millions_eur")
    gk_b = _safe_value(row_b, "goalkeeper_market_value_millions_eur")

    def_a = _safe_value(row_a, "defender_market_value_millions_eur")
    def_b = _safe_value(row_b, "defender_market_value_millions_eur")

    mid_a = _safe_value(row_a, "midfield_market_value_millions_eur")
    mid_b = _safe_value(row_b, "midfield_market_value_millions_eur")

    att_a = _safe_value(row_a, "attack_market_value_millions_eur")
    att_b = _safe_value(row_b, "attack_market_value_millions_eur")

    total_a = _safe_value(row_a, "scraped_total_market_value_millions_eur")
    total_b = _safe_value(row_b, "scraped_total_market_value_millions_eur")

    gk_age_a = _safe_value(row_a, "goalkeeper_avg_age")
    gk_age_b = _safe_value(row_b, "goalkeeper_avg_age")

    def_age_a = _safe_value(row_a, "defender_avg_age")
    def_age_b = _safe_value(row_b, "defender_avg_age")

    mid_age_a = _safe_value(row_a, "midfield_avg_age")
    mid_age_b = _safe_value(row_b, "midfield_avg_age")

    att_age_a = _safe_value(row_a, "attack_avg_age")
    att_age_b = _safe_value(row_b, "attack_avg_age")
    

    return {
        "goalkeeper_value_diff": gk_a - gk_b,
        "defender_value_diff": def_a - def_b,
        "midfield_value_diff": mid_a - mid_b,
        "attack_value_diff": att_a - att_b,

        "goalkeeper_share_diff": _share(gk_a, total_a) - _share(gk_b, total_b),
        "defender_share_diff": _share(def_a, total_a) - _share(def_b, total_b),
        "midfield_share_diff": _share(mid_a, total_a) - _share(mid_b, total_b),
        "attack_share_diff": _share(att_a, total_a) - _share(att_b, total_b),

        "goalkeeper_age_diff": _age_diff(gk_age_a, gk_age_b),
        "defender_age_diff": _age_diff(def_age_a, def_age_b),
        "midfield_age_diff": _age_diff(mid_age_a, mid_age_b),
        "attack_age_diff": _age_diff(att_age_a, att_age_b),

         "attack_defense_ratio_diff": (
            _safe_ratio(att_a, def_a) -
            _safe_ratio(att_b, def_b)
        ),
        "midfield_attack_ratio_diff": (
            _safe_ratio(mid_a, att_a) -
            _safe_ratio(mid_b, att_b)
        ),
        "total_age_diff": _age_diff(
            _mean_positive([gk_age_a, def_age_a, mid_age_a, att_age_a]),
            _mean_positive([gk_age_b, def_age_b, mid_age_b, att_age_b]),
        ),       
    }
