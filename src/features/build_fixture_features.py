"""Build feature rows for multiple fixtures."""

import pandas as pd

from src.features.build_features import build_pre_match_features
from src.features.feature_columns import FEATURE_COLS
from src.features.team_names import normalize_team_name


def build_features_for_fixtures(
    fixtures: pd.DataFrame,
    team_states: dict,
    historical_matches: pd.DataFrame,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    elo_ratings: dict[str, float],
    rankings: dict[str, int],
) -> pd.DataFrame:
    rows = []

    for _, match in fixtures.iterrows():
        team_a = normalize_team_name(match["team_a"])
        team_b = normalize_team_name(match["team_b"])

        feature_row = build_pre_match_features(
            team_a=team_a,
            team_b=team_b,
            match_date=match["date"],
            team_states=team_states,
            historical_matches=historical_matches,
            market_values=market_values,
            position_values=position_values,
            elo_ratings=elo_ratings,
            rankings=rankings,
        )

        feature_row.insert(0, "match_id", match["match_id"])
        feature_row.insert(1, "date", match["date"])
        feature_row.insert(2, "team_a", team_a)
        feature_row.insert(3, "team_b", team_b)
        feature_row.insert(4, "group", match["group"])

        rows.append(feature_row)

    result = pd.concat(rows, ignore_index=True)

    return result[
        ["match_id", "date", "team_a", "team_b", "group"]
        + FEATURE_COLS
    ]