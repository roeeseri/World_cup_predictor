from __future__ import annotations

import pandas as pd

from src.features.tournament_state_features import (
    compute_tournament_state_features,
    update_state_after_match,
)
from src.models.predict import predict_match_from_features
from src.features.feature_columns import FEATURE_COLS


def run_chronological_backtest(
    matches_df: pd.DataFrame,
    model,
    initial_state: dict,
    pre_tournament_data=None,
) -> pd.DataFrame:
    """
    Simulate a tournament match-by-match in chronological order.

    For each match:
    1. Build a feature row from the current tournament state (pre-match snapshot)
    2. Predict expected goals
    3. Record the prediction
    4. Update tournament state with the actual result

    Args:
        matches_df: DataFrame with columns: date, team_a, team_b,
                    all FEATURE_COLS features, goals_a, goals_b
        model: Fitted production model
        initial_state: Starting tournament state dict (usually empty)
        pre_tournament_data: Unused; kept for API compatibility

    Returns:
        DataFrame with one row per match containing predictions and actuals
    """
    if "date" in matches_df.columns:
        matches_df = matches_df.sort_values("date").reset_index(drop=True)

    team_state = dict(initial_state) if initial_state else {}
    predictions = []

    for _, row in matches_df.iterrows():
        team_a = row["team_a"]
        team_b = row["team_b"]

        # Inject live tournament state into the feature row
        state_features = compute_tournament_state_features(team_a, team_b, team_state)
        feature_row = row[FEATURE_COLS].copy()
        for col, val in state_features.items():
            if col in feature_row.index:
                feature_row[col] = val

        feature_df = pd.DataFrame([feature_row])
        prediction = predict_match_from_features(model, feature_df)

        actual_goals_a = int(row.get("goals_a", row.get("goals_A", 0)))
        actual_goals_b = int(row.get("goals_b", row.get("goals_B", 0)))

        predictions.append({
            "team_a": team_a,
            "team_b": team_b,
            "date": row.get("date"),
            "goals_a": actual_goals_a,
            "goals_b": actual_goals_b,
            "pred_lambda_a": prediction["lambda_a"],
            "pred_lambda_b": prediction["lambda_b"],
            "pred_score_a": prediction["most_likely_score"][0],
            "pred_score_b": prediction["most_likely_score"][1],
            "win_prob": prediction["win_prob"],
            "draw_prob": prediction["draw_prob"],
            "loss_prob": prediction["loss_prob"],
        })

        team_state = update_state_after_match(
            team_state, team_a, team_b, actual_goals_a, actual_goals_b
        )

    return pd.DataFrame(predictions)
