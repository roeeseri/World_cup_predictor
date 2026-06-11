"""Verify team-swap consistency of V4 and V5 production models."""
import joblib
import numpy as np
import pandas as pd

from src.features.feature_columns import FEATURE_COLS, FEATURE_COLS_V5_PROD

rng = np.random.default_rng(7)


def make_row(cols):
    row = {}
    for c in cols:
        if c == "rank_diff":
            row[c] = -25.0
        elif c == "elo_diff":
            row[c] = -120.0
        elif c == "rating_a_before":
            row[c] = 1750.0
        elif c == "rating_b_before":
            row[c] = 1870.0
        elif c == "competition_importance":
            row[c] = 4.0
        elif c == "team_a_days_since_last_match":
            row[c] = 4.0
        elif c == "team_b_days_since_last_match":
            row[c] = 9.0
        elif c == "rest_diff":
            row[c] = -5.0
        elif c == "team_a_matches_played_before":
            row[c] = 320.0
        elif c == "team_b_matches_played_before":
            row[c] = 410.0
        elif c == "team_a_tournament_matches_played":
            row[c] = 2.0
        elif c == "team_b_tournament_matches_played":
            row[c] = 2.0
        elif "diff" in c:
            row[c] = float(rng.normal(0, 1))
        else:
            row[c] = float(rng.normal(0, 1))
    return pd.DataFrame([row])[cols]


def swap_row(df, cols):
    """Manually construct the true team-swapped row."""
    r = df.iloc[0].to_dict()
    out = dict(r)
    for c in cols:
        if c.endswith("_diff") or c.startswith(("rank_diff", "elo_diff")) or "_diff_" in c:
            out[c] = -r[c]
    # exact paired swaps
    pairs = [
        ("rating_a_before", "rating_b_before"),
        ("team_a_matches_played_before", "team_b_matches_played_before"),
        ("team_a_days_since_last_match", "team_b_days_since_last_match"),
        ("team_a_tournament_matches_played", "team_b_tournament_matches_played"),
    ]
    for a, b in pairs:
        if a in cols and b in cols:
            out[a], out[b] = r[b], r[a]
    if "competition_importance" in cols:
        out["competition_importance"] = r["competition_importance"]
    return pd.DataFrame([out])[cols]


for name, path, cols in [
    ("V4", "models/production_model_v4.joblib", FEATURE_COLS),
    ("V5", "models/production_model_v5.joblib", FEATURE_COLS_V5_PROD),
]:
    model = joblib.load(path)
    X = make_row(cols)
    X_sw = swap_row(X, cols)
    p1 = model.predict(X)        # (la, lb) for A vs B
    p2 = model.predict(X_sw)     # (la', lb') for B vs A
    # consistency: la should equal lb', lb should equal la'
    d1 = abs(p1[0, 0] - p2[0, 1])
    d2 = abs(p1[0, 1] - p2[0, 0])
    print(f"{name}: AvsB=({p1[0,0]:.4f},{p1[0,1]:.4f})  BvsA=({p2[0,0]:.4f},{p2[0,1]:.4f})  "
          f"asym=({d1:.4f},{d2:.4f})")
