"""Production feature columns (v4 and v5) and symmetry utilities."""

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


# ── V5 feature set ─────────────────────────────────────────────────────────────
# Changes from v4:
#   DROPPED: team_a/b_matches_played_before (career count noise ~300-400, per design doc)
#   ADDED:   competition_importance (numeric from COMPETITION_WEIGHTS; 4.0 at WC inference)
#            tournament_key_numeric (ordinal 1=WC…6=Friendly)
#            rest_diff (team_a_days - team_b_days; signed diff replaces raw pair for model)
#            tournament_goals_for_per_match_diff (per-tournament attack rate diff)
#            tournament_goals_against_per_match_diff (per-tournament defense rate diff)
FEATURE_COLS_V5 = [
    "rank_diff",
    "elo_diff",
    "rating_a_before",
    "rating_b_before",
    "avg_player_value_diff",
    "opponent_strength_diff_last5",
    "weighted_goals_for_diff_last5",
    "weighted_goals_against_diff_last5",
    "market_value_rel_mean_diff",
    "rating_change_diff_last5",
    "defender_share_diff",
    "goalkeeper_share_diff",
    "team_a_days_since_last_match",
    "team_b_days_since_last_match",
    "rest_diff",
    "competition_importance",
    "tournament_key_numeric",
    "tournament_goal_diff_diff",
    "tournament_points_diff",
    "team_a_tournament_matches_played",
    "team_b_tournament_matches_played",
    "tournament_goals_for_per_match_diff",
    "tournament_goals_against_per_match_diff",
]

# Production V5 feature set (20 features): V4 minus matches_played + rest_diff + competition_importance
# This is what is trained and deployed; FEATURE_COLS_V5 is the full research set (23 features).
FEATURE_COLS_V5_PROD = [
    "rank_diff",
    "elo_diff",
    "rating_a_before",
    "rating_b_before",
    "avg_player_value_diff",
    "opponent_strength_diff_last5",
    "weighted_goals_for_diff_last5",
    "weighted_goals_against_diff_last5",
    "market_value_rel_mean_diff",
    "rating_change_diff_last5",
    "defender_share_diff",
    "goalkeeper_share_diff",
    "team_b_days_since_last_match",
    "team_a_days_since_last_match",
    "rest_diff",
    "competition_importance",
    "tournament_goal_diff_diff",
    "tournament_points_diff",
    "team_a_tournament_matches_played",
    "team_b_tournament_matches_played",
]

_DIFF_COLS_V5 = [
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
    "rest_diff",
    "tournament_goal_diff_diff",
    "tournament_points_diff",
    "tournament_goals_for_per_match_diff",
    "tournament_goals_against_per_match_diff",
]

_PAIRED_COLS_V5 = [
    ("rating_a_before",                     "rating_b_before"),
    ("team_a_days_since_last_match",        "team_b_days_since_last_match"),
    ("team_a_tournament_matches_played",    "team_b_tournament_matches_played"),
]

_DIFF_INDICES_V5 = [FEATURE_COLS_V5.index(c) for c in _DIFF_COLS_V5 if c in FEATURE_COLS_V5]
_PAIR_INDICES_V5 = [
    (FEATURE_COLS_V5.index(a), FEATURE_COLS_V5.index(b))
    for a, b in _PAIRED_COLS_V5
    if a in FEATURE_COLS_V5 and b in FEATURE_COLS_V5
]


def mirror_features_v5(X):
    """mirror_features equivalent for FEATURE_COLS_V5."""
    if isinstance(X, pd.DataFrame):
        X_m = X.copy()
        for col in _DIFF_COLS_V5:
            if col in X_m.columns:
                X_m[col] = -X_m[col]
        for col_a, col_b in _PAIRED_COLS_V5:
            if col_a in X_m.columns and col_b in X_m.columns:
                X_m[col_a], X_m[col_b] = X_m[col_b].copy(), X_m[col_a].copy()
        return X_m
    else:
        X_m = np.array(X, dtype=float)
        for i in _DIFF_INDICES_V5:
            X_m[:, i] = -X_m[:, i]
        for ia, ib in _PAIR_INDICES_V5:
            X_m[:, ia], X_m[:, ib] = X_m[:, ib].copy(), X_m[:, ia].copy()
        return X_m


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
