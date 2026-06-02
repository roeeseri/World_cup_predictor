"""Pre-tournament feature helpers.

These features are known before a match starts and do not depend on live
tournament standings.
"""

import pandas as pd

from src.features.build_features import compute_elo_features
from src.features.market_value_features import compute_market_value_features
from src.features.position_value_features import compute_position_value_features
from src.features.recent_form_features import compute_recent_form_features


def build_pre_tournament_features(
    team_a: str,
    team_b: str,
    match_date,
    historical_matches: pd.DataFrame,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    elo_ratings: dict[str, float],
    rankings: dict[str, int],
) -> dict:
    """Build all non-tournament-state features for one upcoming match."""
    match_date = pd.to_datetime(match_date)
    year = int(match_date.year)

    features = {}

    features.update(
        compute_elo_features(
            team_a=team_a,
            team_b=team_b,
            elo_ratings=elo_ratings,
            rankings=rankings,
        )
    )

    features.update(
        compute_market_value_features(
            team_a=team_a,
            team_b=team_b,
            year=year,
            market_values=market_values,
        )
    )

    features.update(
        compute_position_value_features(
            team_a=team_a,
            team_b=team_b,
            year=year,
            position_values=position_values,
        )
    )

    features.update(
        compute_recent_form_features(
            team_a=team_a,
            team_b=team_b,
            historical_matches=historical_matches,
            cutoff_date=match_date,
        )
    )

    return features
