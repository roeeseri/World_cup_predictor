"""Production prediction interface for the WC score predictor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import build_pre_match_features
from src.features.feature_columns import FEATURE_COLS
from src.models.score_conversion import most_likely_score, top_scores, win_draw_loss_probs
from src.models.train import load_model

DEFAULT_MODEL_PATH = "models/production_model.joblib"


def load_production_model(path: str = DEFAULT_MODEL_PATH):
    """Load a saved production model from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"No model found at {path}. "
            "Run `python -m src.models.train_production` to train one."
        )
    return load_model(p)


def predict_match_from_features(model, feature_row: pd.DataFrame) -> dict:
    """
    Run inference on a pre-built feature row.

    Args:
        model: Fitted goal model (LGBMGoalModel, XGBGoalModel, etc.)
        feature_row: Single-row DataFrame with exactly FEATURE_COLS columns

    Returns:
        dict with keys: lambda_a, lambda_b, win_prob, draw_prob, loss_prob,
                        most_likely_score (tuple), top_scores (list of tuples)
    """
    X = feature_row[FEATURE_COLS].fillna(0)
    pred = np.clip(model.predict(X), 0, None)
    lambda_a, lambda_b = float(pred[0, 0]), float(pred[0, 1])

    win_prob, draw_prob, loss_prob = win_draw_loss_probs(lambda_a, lambda_b)
    best_score = most_likely_score(lambda_a, lambda_b)
    best_scores = top_scores(lambda_a, lambda_b, n=5)

    return {
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "win_prob": win_prob,
        "draw_prob": draw_prob,
        "loss_prob": loss_prob,
        "most_likely_score": best_score,
        "top_scores": best_scores,
    }


def predict_fixture(
    model,
    team_a: str,
    team_b: str,
    match_date,
    team_states: dict,
    historical_matches: pd.DataFrame,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    elo_ratings: dict[str, float],
    rankings: dict[str, int],
) -> dict:
    """
    End-to-end prediction: build features from live state then predict.

    Args:
        model: Fitted production model
        team_a, team_b: Team names (normalized)
        match_date: Date of the match (str or datetime)
        team_states: Tournament state dict from initialize_team_states()
        historical_matches: Full pre-tournament historical matches DataFrame
        market_values: Transfermarkt market values DataFrame
        position_values: Transfermarkt position values DataFrame
        elo_ratings: Dict of {team_name: elo_rating}
        rankings: Dict of {team_name: fifa_rank}

    Returns:
        dict with prediction results plus built feature_row
    """
    feature_row = build_pre_match_features(
        team_a=team_a,
        team_b=team_b,
        match_date=match_date,
        team_states=team_states,
        historical_matches=historical_matches,
        market_values=market_values,
        position_values=position_values,
        elo_ratings=elo_ratings,
        rankings=rankings,
    )

    result = predict_match_from_features(model, feature_row)
    result["team_a"] = team_a
    result["team_b"] = team_b
    result["match_date"] = str(match_date)
    result["feature_row"] = feature_row
    return result
