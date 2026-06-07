from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.feature_columns import FEATURE_COLS
from src.models.score_conversion import most_likely_score, top_scores, win_draw_loss_probs
from src.tournament.simulate_world_cup import simulate_world_cup_2026


MODEL_PATH = ROOT / "models" / "production_model_v2.joblib"
MODEL_DATASET_PATH = ROOT / "data" / "processed" / "model_dataset.csv"
GROUP_FEATURES_PATH = ROOT / "data" / "processed" / "world_cup_2026_group_stage_features.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_model_dataset():
    df = pd.read_csv(MODEL_DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_2026_group_features():
    return pd.read_csv(GROUP_FEATURES_PATH)


@st.cache_data
def run_cached_simulation(_model, model_df, group_features):
    return simulate_world_cup_2026(
        model=_model,
        model_df=model_df,
        group_features=group_features,
    )


def build_direct_match_row(model_df, team_a, team_b):
    direct = model_df[
        (model_df["team_a"] == team_a)
        & (model_df["team_b"] == team_b)
    ].sort_values("date")

    if not direct.empty:
        return pd.DataFrame([direct.iloc[-1][FEATURE_COLS]])

    fallback = model_df[FEATURE_COLS].median(numeric_only=True)
    return pd.DataFrame([fallback[FEATURE_COLS]])


def show_match_predictor(model, model_df):
    st.header("⚽ Single Match Predictor")

    teams = sorted(set(model_df["team_a"]) | set(model_df["team_b"]))

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team A", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
    with col2:
        team_b = st.selectbox("Team B", teams, index=teams.index("France") if "France" in teams else 1)

    if team_a == team_b:
        st.warning("Choose two different teams.")
        return

    if st.button("Predict Match", type="primary"):
        X = build_direct_match_row(model_df, team_a, team_b).fillna(0)
        pred = model.predict(X)

        lambda_a = float(pred[0, 0])
        lambda_b = float(pred[0, 1])

        score_a, score_b = most_likely_score(lambda_a, lambda_b)
        win_a, draw, win_b = win_draw_loss_probs(lambda_a, lambda_b)

        if score_a > score_b:
            winner = team_a
        elif score_b > score_a:
            winner = team_b
        else:
            winner = "Draw"

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Score", f"{team_a} {score_a} - {score_b} {team_b}")
        c2.metric("Expected Goals", f"{lambda_a:.2f} - {lambda_b:.2f}")
        c3.metric("Most Likely Result", winner)

        p1, p2, p3 = st.columns(3)
        p1.metric(f"{team_a} Win", f"{win_a * 100:.1f}%")
        p2.metric("Draw", f"{draw * 100:.1f}%")
        p3.metric(f"{team_b} Win", f"{win_b * 100:.1f}%")

        score_options = pd.DataFrame(
            [
                {
                    "score": f"{a}-{b}",
                    "team_a_goals": a,
                    "team_b_goals": b,
                    "probability_%": round(prob * 100, 2),
                }
                for a, b, prob in top_scores(lambda_a, lambda_b, n=10)
            ]
        )

        st.subheader("Top Score Options")
        st.dataframe(score_options, use_container_width=True, hide_index=True)

        st.subheader("Feature Row Used")
        st.dataframe(X.T.rename(columns={X.index[0]: "value"}), use_container_width=True)


def show_world_cup_dashboard(model, model_df, group_features):
    st.header("🏆 World Cup 2026 Full Simulation")

    results = run_cached_simulation(model, model_df, group_features)

    champion = results["champion"]
    runner_up = results["runner_up"]
    third_place = results["third_place"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Champion", champion)
    c2.metric("Runner-up", runner_up)
    c3.metric("Third Place", third_place)

    st.info(
        "This is a deterministic single-path simulation: each match uses the model's most likely score. "
        "Later we can add Monte Carlo simulations for title probabilities."
    )

    tabs = st.tabs([
        "Group Predictions",
        "Group Standings",
        "Round of 32",
        "Knockout Bracket",
        "Final Summary",
        "Model Features",
    ])

    with tabs[0]:
        st.subheader("Group Stage Predictions")
        df = results["group_predictions"].copy()
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("Group Standings")
        standings = results["standings"].copy()

        for group in sorted(standings["group"].unique()):
            st.markdown(f"### {group}")
            gdf = standings[standings["group"] == group].copy()
            st.dataframe(
                gdf[
                    [
                        "position",
                        "team",
                        "played",
                        "wins",
                        "draws",
                        "losses",
                        "goals_for",
                        "goals_against",
                        "goal_diff",
                        "points",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    with tabs[2]:
        st.subheader("Round of 32 Fixtures")
        st.dataframe(results["r32_fixtures"], use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("Knockout Results")
        knockout = results["knockout_results"].copy()

        round_order = ["R32", "R16", "QF", "SF", "THIRD_PLACE", "FINAL"]

        for round_name in round_order:
            rdf = knockout[knockout["round"] == round_name].copy()
            if rdf.empty:
                continue

            title = {
                "R32": "Round of 32",
                "R16": "Round of 16",
                "QF": "Quarter Finals",
                "SF": "Semi Finals",
                "THIRD_PLACE": "Third Place Match",
                "FINAL": "Final",
            }.get(round_name, round_name)

            st.markdown(f"### {title}")
            st.dataframe(
                rdf[
                    [
                        "match_slot",
                        "team_a",
                        "team_b",
                        "pred_score",
                        "lambda_a",
                        "lambda_b",
                        "team_a_win_prob",
                        "draw_prob",
                        "team_b_win_prob",
                        "winner",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    with tabs[4]:
        st.subheader("Tournament Final")
        final = results["knockout_results"][results["knockout_results"]["round"] == "FINAL"].iloc[0]
        third = results["knockout_results"][results["knockout_results"]["round"] == "THIRD_PLACE"].iloc[0]

        st.markdown(f"## 🏆 {champion}")
        st.write(f"Final: **{final['team_a']} {final['pred_score']} {final['team_b']}**")
        st.write(f"Winner: **{final['winner']}**")

        st.markdown("### Third Place")
        st.write(f"Third-place match: **{third['team_a']} {third['pred_score']} {third['team_b']}**")
        st.write(f"Third place: **{third['winner']}**")

    with tabs[5]:
        st.subheader("Production Feature Columns")
        st.write(f"Number of features: **{len(FEATURE_COLS)}**")
        st.dataframe(pd.DataFrame({"feature": FEATURE_COLS}), use_container_width=True, hide_index=True)

        st.subheader("2026 Feature Coverage")
        missing = [c for c in FEATURE_COLS if c not in group_features.columns]
        if missing:
            st.error(f"Missing 2026 features: {missing}")
        else:
            st.success("All production features exist in the 2026 fixture dataset.")

        st.dataframe(group_features.head(20), use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="World Cup Score Predictor",
        page_icon="⚽",
        layout="wide",
    )

    st.title("⚽ World Cup Score Predictor")
    st.caption("Production model + final 21 features + 2026 tournament simulator")

    model = load_model()
    model_df = load_model_dataset()
    group_features = load_2026_group_features()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose page",
        [
            "Single Match Predictor",
            "World Cup 2026 Simulation",
        ],
    )

    st.sidebar.divider()
    st.sidebar.write("Model:")
    st.sidebar.code(type(model).__name__)
    st.sidebar.write("Features:")
    st.sidebar.code(str(len(FEATURE_COLS)))

    if page == "Single Match Predictor":
        show_match_predictor(model, model_df)
    else:
        show_world_cup_dashboard(model, model_df, group_features)


if __name__ == "__main__":
    main()
