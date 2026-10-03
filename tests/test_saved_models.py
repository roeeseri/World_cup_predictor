"""Regression checks for real saved models and their pre-match feature builders."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.load_results import load_historical_results
from src.features.build_features import build_pre_match_features, build_pre_match_features_v5
from src.state.live_state import derive_rankings_from_elo

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prediction_inputs():
    history = load_historical_results(ROOT / "data/raw")
    ratings = pd.concat([
        history[[f"team_{side}", f"rating_{side}", "date"]].rename(
            columns={f"team_{side}": "team", f"rating_{side}": "rating"}
        )
        for side in ("a", "b")
    ]).sort_values("date").groupby("team")["rating"].last().to_dict()
    return dict(
        match_date=pd.Timestamp("2026-10-03"),
        team_states={},
        historical_matches=history,
        market_values=pd.read_csv(ROOT / "data/processed/transfermarkt_market_values_clean.csv"),
        position_values=pd.read_csv(ROOT / "data/processed/transfermarkt_position_values_2004_2026.csv"),
        elo_ratings=ratings,
        rankings=derive_rankings_from_elo(ratings),
    )


@pytest.mark.parametrize("version", [4, 5, 6])
def test_saved_model_predictions_survive_cleanup(version, prediction_inputs):
    """Protect serialization paths, feature ordering, and existing predictions."""
    import json

    expected = json.loads((ROOT / "tests/fixtures/saved_predictions.json").read_text())
    model = joblib.load(ROOT / f"models/production_model_v{version}.joblib")
    build = build_pre_match_features if version == 4 else build_pre_match_features_v5
    for team_a, team_b in [("Argentina", "France"), ("France", "Argentina"), ("Brazil", "Japan")]:
        features = build(team_a=team_a, team_b=team_b, **prediction_inputs).fillna(0)
        np.testing.assert_allclose(
            model.predict(features), expected[f"v{version}:{team_a}:{team_b}"],
            rtol=1e-6, atol=1e-8,
        )
