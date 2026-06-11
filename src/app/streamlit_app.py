from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.feature_columns import FEATURE_COLS, FEATURE_COLS_V5_PROD
from src.features.build_features import build_pre_match_features, build_pre_match_features_v5
from src.models.score_conversion import (
    most_likely_score,
    most_likely_score_v5,
    most_likely_score_v6,
    top_scores,
    win_draw_loss_probs,
)
from src.app.live_tournament_page import show_live_tournament
from src.app.simulation_page import show_wc_simulation
from src.state.live_state import derive_rankings_from_elo


MODEL_PATH_V4 = ROOT / "models" / "production_model_v4.joblib"
MODEL_PATH_V5 = ROOT / "models" / "production_model_v5.joblib"
MODEL_PATH_V6 = ROOT / "models" / "production_model_v6.joblib"
CONFIG_PATH_V6 = ROOT / "models" / "production_config_v6.json"
MODEL_DATASET_PATH = ROOT / "data" / "processed" / "model_dataset.csv"
GROUP_FEATURES_PATH = ROOT / "data" / "processed" / "world_cup_2026_group_stage_features.csv"
MARKET_VALUES_PATH = ROOT / "data" / "processed" / "transfermarkt_market_values_clean.csv"
POSITION_VALUES_PATH = ROOT / "data" / "processed" / "transfermarkt_position_values_2004_2026.csv"
FIXTURES_PATH = ROOT / "data" / "raw" / "fixtures" / "world_cup_2026_group_stage.csv"
RAW_DATA_DIR = ROOT / "data" / "raw"


@st.cache_resource
def load_model_v4():
    return joblib.load(MODEL_PATH_V4)


@st.cache_resource
def load_model_v5():
    return joblib.load(MODEL_PATH_V5)


@st.cache_resource
def load_model_v6():
    return joblib.load(MODEL_PATH_V6)


@st.cache_resource
def make_v6_score_fn():
    """Drawband score_fn with calibration params from the V6 config file."""
    import json
    from functools import partial

    params = {}
    if CONFIG_PATH_V6.exists():
        with open(CONFIG_PATH_V6) as f:
            db = json.load(f).get("drawband", {})
        params = {
            "draw_threshold": db.get("draw_threshold", 0.33),
            "threshold_b": db.get("threshold_b", 0.5),
            "scale_c": db.get("scale_c", 0.9992),
            "rho": db.get("rho", -0.3294),
        }
    return partial(most_likely_score_v6, **params)


@st.cache_data
def load_model_dataset():
    df = pd.read_csv(MODEL_DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_2026_group_features():
    return pd.read_csv(GROUP_FEATURES_PATH)


@st.cache_data
def load_market_values():
    return pd.read_csv(MARKET_VALUES_PATH)


@st.cache_data
def load_position_values():
    return pd.read_csv(POSITION_VALUES_PATH)


@st.cache_data
def load_fixtures():
    from src.data.load_fixtures import load_tournament_fixtures
    return load_tournament_fixtures(FIXTURES_PATH)


@st.cache_data
def load_raw_historical():
    from src.data.load_results import load_historical_results
    return load_historical_results(RAW_DATA_DIR)


@st.cache_data
def run_cached_simulation(_model, model_df, group_features):
    return simulate_world_cup_2026(
        model=_model,
        model_df=model_df,
        group_features=group_features,
    )


@st.cache_data
def _extract_elo_ratings(historical_matches: pd.DataFrame) -> dict[str, float]:
    """Get the most recent post-match ELO for every team from raw historical data."""
    a = historical_matches[["team_a", "rating_a", "date"]].rename(columns={"team_a": "team", "rating_a": "rating"})
    b = historical_matches[["team_b", "rating_b", "date"]].rename(columns={"team_b": "team", "rating_b": "rating"})
    latest = (
        pd.concat([a, b])
        .sort_values("date")
        .groupby("team")["rating"]
        .last()
    )
    return latest.to_dict()


def show_match_predictor(model, raw_historical, market_values, position_values, feature_fn=None, score_fn=None):
    st.header("⚽ Single Match Predictor")

    if feature_fn is None:
        feature_fn = build_pre_match_features
    if score_fn is None:
        score_fn = most_likely_score

    elo_ratings = _extract_elo_ratings(raw_historical)
    rankings = derive_rankings_from_elo(elo_ratings)

    teams = sorted(elo_ratings.keys())

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team A", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
    with col2:
        team_b = st.selectbox("Team B", teams, index=teams.index("France") if "France" in teams else 1)

    if team_a == team_b:
        st.warning("Choose two different teams.")
        return

    if st.button("Predict Match", type="primary"):
        try:
            X = feature_fn(
                team_a=team_a,
                team_b=team_b,
                match_date=pd.Timestamp.now(),
                team_states={},
                historical_matches=raw_historical,
                market_values=market_values,
                position_values=position_values,
                elo_ratings=elo_ratings,
                rankings=rankings,
            ).fillna(0)
        except Exception as e:
            st.error(f"Could not build features: {e}")
            return
        pred = model.predict(X)

        lambda_a = float(pred[0, 0])
        lambda_b = float(pred[0, 1])

        score_a, score_b = score_fn(lambda_a, lambda_b)
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
    st.caption("Production model + 2026 tournament simulator")

    model_df = load_model_dataset()
    group_features = load_2026_group_features()
    market_values = load_market_values()
    position_values = load_position_values()
    fixtures = load_fixtures()
    raw_historical = load_raw_historical()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose page",
        [
            "Single Match Predictor",
            "World Cup 2026 Simulation",
            "Live Tournament",
        ],
    )

    st.sidebar.divider()

    # Model version selector
    model_options = ["V4 (production)"]
    if MODEL_PATH_V5.exists():
        model_options.append("V5 (conditional floor)")
    if MODEL_PATH_V6.exists():
        model_options.append("V6 (drawband)")
    model_choice = st.sidebar.radio("Model version", model_options, index=0)

    if model_choice.startswith("V6"):
        model = load_model_v6()
        score_fn = make_v6_score_fn()
        feature_fn = build_pre_match_features_v5  # V6 uses the same 20 features as V5
        feature_cols = FEATURE_COLS_V5_PROD
    elif model_choice.startswith("V5"):
        model = load_model_v5()
        score_fn = most_likely_score_v5
        feature_fn = build_pre_match_features_v5
        feature_cols = FEATURE_COLS_V5_PROD
    else:
        model = load_model_v4()
        score_fn = most_likely_score
        feature_fn = build_pre_match_features
        feature_cols = FEATURE_COLS

    st.sidebar.write("Features:")
    st.sidebar.code(str(len(feature_cols)))

    if page == "Single Match Predictor":
        show_match_predictor(model, raw_historical, market_values, position_values, feature_fn=feature_fn, score_fn=score_fn)
    elif page == "World Cup 2026 Simulation":
        show_wc_simulation(model, raw_historical, fixtures, market_values, position_values, score_fn=score_fn, feature_fn=feature_fn)
    elif page == "Live Tournament":
        show_live_tournament(
            model=model,
            fixtures=fixtures,
            historical_matches=raw_historical,
            market_values=market_values,
            position_values=position_values,
            score_fn=score_fn,
            feature_fn=feature_fn,
        )


if __name__ == "__main__":
    main()
