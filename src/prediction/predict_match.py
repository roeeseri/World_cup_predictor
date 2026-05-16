from __future__ import annotations

import pandas as pd

from .score_conversion import most_likely_score, outcome_probabilities


def predict_match(match: dict, model, team_state: dict, pre_tournament_data) -> dict:
    home_team = match["home_team"]
    away_team = match["away_team"]

    features = _build_features(match, team_state, pre_tournament_data)
    preds = model.predict(features)
    lambda_home, lambda_away = float(preds[0][0]), float(preds[0][1])

    predicted_score = most_likely_score(lambda_home, lambda_away)
    outcome_probs = outcome_probabilities(lambda_home, lambda_away)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "expected_home_goals": lambda_home,
        "expected_away_goals": lambda_away,
        "predicted_score": predicted_score,
        "outcome_probabilities": outcome_probs,
    }


def _build_features(match: dict, team_state: dict, pre_tournament_data) -> pd.DataFrame:
    features: dict[str, float] = {}

    for key, value in match.items():
        if key in {"home_team", "away_team"}:
            continue
        if isinstance(value, (int, float)):
            features[key] = float(value)

    if "features" in match and isinstance(match["features"], dict):
        for key, value in match["features"].items():
            if isinstance(value, (int, float)):
                features[key] = float(value)

    features.update(_team_features(match["home_team"], "home_", team_state, pre_tournament_data))
    features.update(_team_features(match["away_team"], "away_", team_state, pre_tournament_data))

    return pd.DataFrame([features])


def _team_features(team: str, prefix: str, team_state: dict, pre_tournament_data) -> dict:
    features: dict[str, float] = {}
    state = team_state.get(team, {}) if team_state else {}
    for key in [
        "points",
        "goals_for",
        "goals_against",
        "goal_diff",
        "wins",
        "draws",
        "losses",
        "played",
        "elo",
    ]:
        if key in state:
            features[f"{prefix}{key}"] = float(state[key])

    if isinstance(pre_tournament_data, pd.DataFrame):
        team_col = None
        if "team" in pre_tournament_data.columns:
            team_col = "team"
        elif "Team" in pre_tournament_data.columns:
            team_col = "Team"

        if team_col:
            row = pre_tournament_data.loc[pre_tournament_data[team_col] == team]
            if not row.empty:
                row = row.iloc[0]
                for col in pre_tournament_data.columns:
                    if col == team_col:
                        continue
                    if pd.api.types.is_numeric_dtype(pre_tournament_data[col]):
                        features[f"{prefix}{col}"] = float(row[col])

    return features
