from __future__ import annotations

import pandas as pd

from src.features.feature_columns import FEATURE_COLS
from src.models.score_conversion import most_likely_score, win_draw_loss_probs


DIFF_COLS = [
    "rank_diff",
    "elo_diff",
    "avg_player_value_diff",
    "opponent_strength_diff_last5",
    "weighted_goals_for_diff_last5",
    "weighted_goals_against_diff_last5",
    "market_value_rel_mean_diff",
    "rating_change_diff_last5",
    "defender_share_diff",
    "goalkeeper_share_diff",
    "tournament_goal_diff_diff",
    "tournament_points_diff",
]

SWAP_PAIRS = [
    ("rating_a_before", "rating_b_before"),
    ("team_a_matches_played_before", "team_b_matches_played_before"),
    ("team_a_days_since_last_match", "team_b_days_since_last_match"),
    ("team_a_tournament_matches_played", "team_b_tournament_matches_played"),
]


def reverse_feature_row(row: pd.Series) -> pd.Series:
    features = row[FEATURE_COLS].copy()

    for col in DIFF_COLS:
        if col in features.index:
            features[col] = -features[col]

    for col_a, col_b in SWAP_PAIRS:
        if col_a in features.index and col_b in features.index:
            old_a = features[col_a]
            features[col_a] = features[col_b]
            features[col_b] = old_a

    return features


def build_match_feature_row(
    model_df: pd.DataFrame,
    team_a: str,
    team_b: str,
) -> pd.DataFrame:
    direct = model_df[
        (model_df["team_a"] == team_a)
        & (model_df["team_b"] == team_b)
    ].sort_values("date")

    if not direct.empty:
        return pd.DataFrame([direct.iloc[-1][FEATURE_COLS]])

    reverse = model_df[
        (model_df["team_a"] == team_b)
        & (model_df["team_b"] == team_a)
    ].sort_values("date")

    if not reverse.empty:
        return pd.DataFrame([reverse_feature_row(reverse.iloc[-1])])

    fallback = model_df[FEATURE_COLS].median(numeric_only=True)

    latest_a = model_df[
        (model_df["team_a"] == team_a)
        | (model_df["team_b"] == team_a)
    ].sort_values("date")

    latest_b = model_df[
        (model_df["team_a"] == team_b)
        | (model_df["team_b"] == team_b)
    ].sort_values("date")

    if not latest_a.empty:
        row_a = latest_a.iloc[-1]
        if row_a["team_a"] == team_a:
            fallback["rating_a_before"] = row_a["rating_a_before"]
            fallback["team_a_matches_played_before"] = row_a["team_a_matches_played_before"]
            fallback["team_a_days_since_last_match"] = row_a["team_a_days_since_last_match"]
        else:
            fallback["rating_a_before"] = row_a["rating_b_before"]
            fallback["team_a_matches_played_before"] = row_a["team_b_matches_played_before"]
            fallback["team_a_days_since_last_match"] = row_a["team_b_days_since_last_match"]

    if not latest_b.empty:
        row_b = latest_b.iloc[-1]
        if row_b["team_b"] == team_b:
            fallback["rating_b_before"] = row_b["rating_b_before"]
            fallback["team_b_matches_played_before"] = row_b["team_b_matches_played_before"]
            fallback["team_b_days_since_last_match"] = row_b["team_b_days_since_last_match"]
        else:
            fallback["rating_b_before"] = row_b["rating_a_before"]
            fallback["team_b_matches_played_before"] = row_b["team_a_matches_played_before"]
            fallback["team_b_days_since_last_match"] = row_b["team_a_days_since_last_match"]

    fallback["elo_diff"] = fallback["rating_a_before"] - fallback["rating_b_before"]

    return pd.DataFrame([fallback[FEATURE_COLS]])


def simulate_match(
    model,
    model_df: pd.DataFrame,
    team_a: str,
    team_b: str,
    knockout: bool = False,
) -> dict:
    X = build_match_feature_row(model_df, team_a, team_b)
    X = X[FEATURE_COLS].fillna(0)

    pred = model.predict(X)
    lambda_a = float(pred[0, 0])
    lambda_b = float(pred[0, 1])

    goals_a, goals_b = most_likely_score(lambda_a, lambda_b)
    win_a, draw, win_b = win_draw_loss_probs(lambda_a, lambda_b)

    if goals_a > goals_b:
        winner = team_a
        loser = team_b
    elif goals_b > goals_a:
        winner = team_b
        loser = team_a
    else:
        if knockout:
            winner = team_a if win_a >= win_b else team_b
            loser = team_b if winner == team_a else team_a
        else:
            winner = "Draw"
            loser = None

    return {
        "team_a": team_a,
        "team_b": team_b,
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "goals_a": goals_a,
        "goals_b": goals_b,
        "pred_score": f"{goals_a}-{goals_b}",
        "team_a_win_prob": win_a,
        "draw_prob": draw,
        "team_b_win_prob": win_b,
        "winner": winner,
        "loser": loser,
    }
