from __future__ import annotations

import pandas as pd

from ..prediction.predict_match import predict_match
from ..state.update_state import update_state_after_match


def run_chronological_backtest(
    matches_df: pd.DataFrame,
    model,
    initial_state: dict,
    pre_tournament_data,
) -> pd.DataFrame:
    if "date" in matches_df.columns:
        matches_df = matches_df.sort_values("date")

    team_state = initial_state or {}
    predictions = []

    for _, row in matches_df.iterrows():
        match = row.to_dict()
        prediction = predict_match(match, model, team_state, pre_tournament_data)

        prediction["home_goals"] = int(row["home_goals"])
        prediction["away_goals"] = int(row["away_goals"])
        prediction["pred_home_goals"] = prediction["expected_home_goals"]
        prediction["pred_away_goals"] = prediction["expected_away_goals"]
        predictions.append(prediction)

        team_state = update_state_after_match(team_state, match)

    return pd.DataFrame(predictions)
