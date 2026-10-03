"""Streamlit entry point. main() selects V4 by default and wires model, feature builder
and score policy to each page. See docs/ARCHITECTURE.md."""

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


def main():
    st.set_page_config(
        page_title="World Cup Score Predictor",
        page_icon="⚽",
        layout="wide",
    )

    st.title("⚽ World Cup Score Predictor")
    st.caption("Production model + 2026 tournament simulator")

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
        model_name = "V6"
    elif model_choice.startswith("V5"):
        model = load_model_v5()
        score_fn = most_likely_score_v5
        feature_fn = build_pre_match_features_v5
        feature_cols = FEATURE_COLS_V5_PROD
        model_name = "V5"
    else:
        model = load_model_v4()
        score_fn = most_likely_score
        feature_fn = build_pre_match_features
        feature_cols = FEATURE_COLS
        model_name = "V4"

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
            model_name=model_name,
        )


if __name__ == "__main__":
    main()
