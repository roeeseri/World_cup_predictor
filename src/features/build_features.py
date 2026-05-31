"""Build one pre-match feature row for prediction."""

import pandas as pd

from src.data.validation import validate_feature_columns, validate_no_target_columns
from src.features.feature_columns import FEATURE_COLS
from src.features.market_value_features import compute_market_value_features
from src.features.match_context_features import compute_match_context
from src.features.recent_form_features import compute_recent_form_features
from src.features.tournament_state_features import compute_tournament_state_features


def compute_elo_features(
    team_a: str,
    team_b: str,
    elo_ratings: dict[str, float],
    rankings: dict[str, int],
) -> dict:
    """Compute rating/ranking features for a match."""
    if team_a not in elo_ratings:
        raise ValueError(f"Missing ELO rating for team_a: {team_a}")

    if team_b not in elo_ratings:
        raise ValueError(f"Missing ELO rating for team_b: {team_b}")

    if team_a not in rankings:
        raise ValueError(f"Missing ranking for team_a: {team_a}")

    if team_b not in rankings:
        raise ValueError(f"Missing ranking for team_b: {team_b}")

    rating_a_before = float(elo_ratings[team_a])
    rating_b_before = float(elo_ratings[team_b])

    rank_a_before = int(rankings[team_a])
    rank_b_before = int(rankings[team_b])

    return {
        "rating_a_before": rating_a_before,
        "rating_b_before": rating_b_before,
        "elo_diff": rating_a_before - rating_b_before,
        "rank_diff": rank_a_before - rank_b_before,
    }


def build_pre_match_features(
    team_a: str,
    team_b: str,
    match_date,
    team_states: dict,
    historical_matches: pd.DataFrame,
    market_values: pd.DataFrame,
    elo_ratings: dict[str, float],
    rankings: dict[str, int],
    competition: str = "World Cup",
    is_home_adv: int = 0,
) -> pd.DataFrame:
    """Build a single-row DataFrame with exactly the production feature columns."""
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
        compute_recent_form_features(
            team_a=team_a,
            team_b=team_b,
            historical_matches=historical_matches,
            cutoff_date=match_date,
        )
    )

    features.update(
        compute_tournament_state_features(
            team_a=team_a,
            team_b=team_b,
            team_states=team_states,
        )
    )

    features.update(
        compute_match_context(
            competition=competition,
            is_home_adv=is_home_adv,
        )
    )

    feature_row = pd.DataFrame([features])

    feature_row = feature_row[FEATURE_COLS].copy()

    validate_no_target_columns(feature_row)
    validate_feature_columns(feature_row, FEATURE_COLS)

    return feature_row
