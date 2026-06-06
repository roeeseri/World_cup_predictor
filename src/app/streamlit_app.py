from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.feature_columns import FEATURE_COLS
from src.models.lgbm_model import LGBMGoalModel
from src.models.weighting import apply_competition_weights
from src.prediction.score_conversion import (
    convert_expected_goals_to_scores,
    outcome_probabilities,
    poisson_score_grid,
)


DATA_PATH = ROOT / "data" / "processed" / "model_dataset.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_resource
def train_model(df: pd.DataFrame) -> LGBMGoalModel:
    train_df = df[df["date"] < "2026-01-01"].copy()

    X = train_df[FEATURE_COLS].fillna(0)
    y = train_df[["target_goals_a", "target_goals_b"]]
    weights = apply_competition_weights(train_df)

    model = LGBMGoalModel()
    model.fit(X, y, sample_weight=weights)

    return model


def reverse_feature_row(row: pd.Series) -> pd.Series:
    features = row[FEATURE_COLS].copy()

    diff_cols = [
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

    swap_pairs = [
        ("rating_a_before", "rating_b_before"),
        ("team_a_matches_played_before", "team_b_matches_played_before"),
        ("team_a_days_since_last_match", "team_b_days_since_last_match"),
        ("team_a_tournament_matches_played", "team_b_tournament_matches_played"),
    ]

    for col in diff_cols:
        if col in features.index:
            features[col] = -features[col]

    for col_a, col_b in swap_pairs:
        if col_a in features.index and col_b in features.index:
            old_a = features[col_a]
            features[col_a] = features[col_b]
            features[col_b] = old_a

    return features


def find_latest_feature_row(
    df: pd.DataFrame,
    team_a: str,
    team_b: str,
) -> pd.DataFrame:
    direct = df[
        (df["team_a"] == team_a)
        & (df["team_b"] == team_b)
    ].sort_values("date")

    if not direct.empty:
        return pd.DataFrame([direct.iloc[-1][FEATURE_COLS]])

    reverse = df[
        (df["team_a"] == team_b)
        & (df["team_b"] == team_a)
    ].sort_values("date")

    if not reverse.empty:
        reversed_features = reverse_feature_row(reverse.iloc[-1])
        return pd.DataFrame([reversed_features])

    # Fallback for teams that never played each other:
    # Use the median feature row, then replace absolute rating features with team latest values.
    fallback = df[FEATURE_COLS].median(numeric_only=True)

    latest_a = df[(df["team_a"] == team_a) | (df["team_b"] == team_a)].sort_values("date")
    latest_b = df[(df["team_a"] == team_b) | (df["team_b"] == team_b)].sort_values("date")

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


def top_score_options(lambda_a: float, lambda_b: float, max_goals: int = 6) -> pd.DataFrame:
    grid = poisson_score_grid(lambda_a, lambda_b, max_goals=max_goals)

    options = (
        grid
        .stack()
        .reset_index()
    )

    options.columns = ["team_a_goals", "team_b_goals", "probability"]

    options = (
        options
        .sort_values("probability", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    options["probability_%"] = (options["probability"] * 100).round(2)

    return options


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    st.set_page_config(
        page_title="World Cup Score Predictor",
        page_icon="⚽",
        layout="wide",
    )

    st.title("⚽ World Cup Score Predictor")
    st.caption(
        "LightGBM model trained with the final 21 production features from src."
    )

    df = load_data()
    model = train_model(df)

    teams = sorted(
        set(df["team_a"].dropna().unique())
        | set(df["team_b"].dropna().unique())
    )

    st.sidebar.header("Match Setup")

    team_a = st.sidebar.selectbox(
        "Team A",
        teams,
        index=teams.index("Argentina") if "Argentina" in teams else 0,
    )

    team_b = st.sidebar.selectbox(
        "Team B",
        teams,
        index=teams.index("France") if "France" in teams else 1,
    )

    predict_clicked = st.sidebar.button("Predict Match", type="primary")

    if team_a == team_b:
        st.warning("Choose two different teams.")
        return

    if not predict_clicked:
        st.info("Choose two teams and click Predict Match.")
        return

    X_match = find_latest_feature_row(df, team_a, team_b)
    X_match = X_match[FEATURE_COLS].fillna(0)

    preds = model.predict(X_match)

    lambda_a = float(preds[0, 0])
    lambda_b = float(preds[0, 1])

    score_array = convert_expected_goals_to_scores([[lambda_a, lambda_b]])
    pred_score = tuple(score_array[0])

    probs = outcome_probabilities(lambda_a, lambda_b)
    score_grid = poisson_score_grid(lambda_a, lambda_b)
    score_prob = float(score_grid.loc[pred_score[0], pred_score[1]])

    if pred_score[0] > pred_score[1]:
        winner = team_a
    elif pred_score[0] < pred_score[1]:
        winner = team_b
    else:
        winner = "Draw"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Score",
            f"{team_a} {pred_score[0]} - {pred_score[1]} {team_b}",
        )

    with col2:
        st.metric(
            "Expected Goals",
            f"{lambda_a:.2f} - {lambda_b:.2f}",
        )

    with col3:
        st.metric(
            "Most Likely Result",
            winner,
        )

    st.divider()

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.metric(f"{team_a} Win", format_pct(probs["home_win"]))

    with p2:
        st.metric("Draw", format_pct(probs["draw"]))

    with p3:
        st.metric(f"{team_b} Win", format_pct(probs["away_win"]))

    with p4:
        st.metric("Predicted Score Probability", format_pct(score_prob))

    st.subheader("Top Score Options")
    st.dataframe(
        top_score_options(lambda_a, lambda_b),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Feature Row Used by the Model")
    st.dataframe(
        X_match.T.rename(columns={X_match.index[0]: "value"}),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()