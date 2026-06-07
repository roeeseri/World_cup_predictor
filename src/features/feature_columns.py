"""Final production feature columns and symmetry utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

# log_market_value_a was dropped: it has no _b counterpart, causing positional
# bias when teams are swapped. Market value info is retained via the diff features.
FEATURE_COLS = [
    "rank_diff",
    "elo_diff",
    "rating_a_before",
    "rating_b_before",
    "avg_player_value_diff",
    "opponent_strength_diff_last5",
    "weighted_goals_for_diff_last5",
    #"log_market_value_a",
    "weighted_goals_against_diff_last5",
    "team_b_matches_played_before",
    "team_a_matches_played_before",
    "market_value_rel_mean_diff",
    "rating_change_diff_last5",
    "defender_share_diff",
    "goalkeeper_share_diff",
    "team_b_days_since_last_match",
    "team_a_days_since_last_match",
    "tournament_goal_diff_diff",
    "tournament_points_diff",
    "team_a_tournament_matches_played",
    "team_b_tournament_matches_played",
]

# Difference features: sign flips when teams are swapped
_DIFF_COLS = [
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

# Paired absolute features: (team_a col, team_b col) swap when teams are swapped
_PAIRED_COLS = [
    ("rating_a_before",               "rating_b_before"),
    ("team_a_matches_played_before",  "team_b_matches_played_before"),
    ("team_a_days_since_last_match",  "team_b_days_since_last_match"),
    ("team_a_tournament_matches_played", "team_b_tournament_matches_played"),
]

# Pre-compute indices for fast numpy mirroring
_DIFF_INDICES  = [FEATURE_COLS.index(c) for c in _DIFF_COLS  if c in FEATURE_COLS]
_PAIR_INDICES  = [
    (FEATURE_COLS.index(a), FEATURE_COLS.index(b))
    for a, b in _PAIRED_COLS
    if a in FEATURE_COLS and b in FEATURE_COLS
]


def mirror_features(X):
    """
    Swap team_a and team_b positions in the feature matrix.

    Difference features are negated; paired absolute features are exchanged.
    Works with both pd.DataFrame (column names) and np.ndarray (positional indices).
    """
    if isinstance(X, pd.DataFrame):
        X_m = X.copy()
        for col in _DIFF_COLS:
            if col in X_m.columns:
                X_m[col] = -X_m[col]
        for col_a, col_b in _PAIRED_COLS:
            if col_a in X_m.columns and col_b in X_m.columns:
                X_m[col_a], X_m[col_b] = X_m[col_b].copy(), X_m[col_a].copy()
        return X_m
    else:
        X_m = np.array(X, dtype=float)
        for i in _DIFF_INDICES:
            X_m[:, i] = -X_m[:, i]
        for ia, ib in _PAIR_INDICES:
            X_m[:, ia], X_m[:, ib] = X_m[:, ib].copy(), X_m[:, ia].copy()
        return X_m
