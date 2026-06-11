"""
V6 experimental features: leakage-safe in-tournament form + reduced-Elo transforms.

Everything here is additive — no V4/V5 file is touched. All in-tournament features
are computed sequentially within each (competition, tournament_year) group so that
a row only ever sees matches played strictly before it (same convention as
protocol._add_tournament_rate_features).

Feature design notes
--------------------
* All new in-tournament features are signed A-minus-B diffs → mirroring = negation.
  This keeps team-swap consistency trivially correct (no paired columns to swap).
* Rate features are shrunk by m/(m+SHRINK_TAU) so 1-match form moves the value less
  than 3-match form ("sample-size weighted form").
* attack/defense Elo-like updates: after each match a team's in-tournament attack
  rating moves by K * (actual_goals - elo_expected_goals). The expectation comes
  from the pre-match elo_diff of that row, so beating a strong opponent 3-0 moves
  the rating much more than beating a minnow 3-0 ("opponent-strength-adjusted").
* form_prior_disagreement: in-tournament adjusted goal-diff rate minus the
  Elo-implied expected goal-diff. Positive = team is outperforming its ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.protocol import MAJOR_TOURNAMENTS

# Shrinkage time-constant for rate features: rate_shrunk = rate * m / (m + TAU)
SHRINK_TAU = 1.5

# Learning rate of the in-tournament attack/defense Elo-style update
ATTACK_K = 0.4

# Elo→goals linearization: expected goal advantage ≈ elo_diff / ELO_PER_GOAL
ELO_PER_GOAL = 350.0
BASE_GOALS = 1.25  # global mean goals per team per match

# Elo cap for the reduced-dependence variant
ELO_CAP = 250.0


# ── New column groups ──────────────────────────────────────────────────────────
V6_FORM_COLS = [
    "t_gf_pm_diff",          # goals for per match (shrunk)
    "t_ga_pm_diff",          # goals against per match (shrunk)
    "t_win_rate_diff",       # win rate (shrunk)
    "t_cs_rate_diff",        # clean sheet rate (shrunk)
    "t_fts_rate_diff",       # failed-to-score rate (shrunk)
    "t_margin_pm_diff",      # avg goal margin (shrunk)
    "t_adj_gd_pm_diff",      # opponent-elo-weighted goal diff per match (shrunk)
]

V6_ELOUPD_COLS = [
    "t_attack_elo_diff",         # in-tournament attack rating diff (Elo-like update)
    "t_defense_elo_diff",        # in-tournament defense rating diff (lower = better defense)
    "form_prior_disagreement",   # form gd rate minus elo-implied gd
]

V6_CAP_COLS = ["elo_diff_capped", "rank_diff_log"]

ALL_V6_COLS = V6_FORM_COLS + V6_ELOUPD_COLS + V6_CAP_COLS


def _empty_state() -> dict:
    return {
        "m": 0, "gf": 0, "ga": 0, "wins": 0, "draws": 0, "cs": 0, "fts": 0,
        "margin_sum": 0.0, "adj_gd_sum": 0.0, "attack": 0.0, "defense": 0.0,
    }


def _shrink(rate: float, m: int, tau: float = SHRINK_TAU) -> float:
    return rate * (m / (m + tau)) if m > 0 else 0.0


def add_v6_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all V6 experimental columns to a prepared dataset
    (output of protocol.load_and_prepare_dataset).

    In-tournament features are 0 outside MAJOR_TOURNAMENTS rows, matching the
    convention of the existing tournament-state features.
    """
    df = df.copy()
    for col in ALL_V6_COLS:
        df[col] = 0.0

    team_a_col = "team_A" if "team_A" in df.columns else "team_a"
    team_b_col = "team_B" if "team_B" in df.columns else "team_b"
    goal_a_col = "goals_A" if "goals_A" in df.columns else "target_goals_a"
    goal_b_col = "goals_B" if "goals_B" in df.columns else "target_goals_b"

    # Static Elo transforms (every row)
    df["elo_diff_capped"] = df["elo_diff"].clip(-ELO_CAP, ELO_CAP)
    df["rank_diff_log"] = np.sign(df["rank_diff"]) * np.log1p(np.abs(df["rank_diff"]))

    df_sorted = df.sort_values("date").reset_index(drop=False)

    for (comp, year), group in df_sorted.groupby(["competition", "tournament_year"], sort=False):
        if comp not in MAJOR_TOURNAMENTS:
            continue

        states: dict[str, dict] = {}

        for _, row in group.sort_values("date").iterrows():
            orig_idx = row["index"]
            ta, tb = row[team_a_col], row[team_b_col]
            elo_diff = float(row["elo_diff"])

            sa = states.get(ta, _empty_state())
            sb = states.get(tb, _empty_state())
            ma, mb = sa["m"], sb["m"]

            # ── pre-match snapshot features ──────────────────────────────────
            def rates(s: dict) -> dict:
                m = s["m"]
                if m == 0:
                    return {k: 0.0 for k in
                            ("gf_pm", "ga_pm", "win", "cs", "fts", "margin", "adj_gd")}
                return {
                    "gf_pm": _shrink(s["gf"] / m, m),
                    "ga_pm": _shrink(s["ga"] / m, m),
                    "win": _shrink(s["wins"] / m, m),
                    "cs": _shrink(s["cs"] / m, m),
                    "fts": _shrink(s["fts"] / m, m),
                    "margin": _shrink(s["margin_sum"] / m, m),
                    "adj_gd": _shrink(s["adj_gd_sum"] / m, m),
                }

            ra, rb = rates(sa), rates(sb)
            df.loc[orig_idx, "t_gf_pm_diff"] = ra["gf_pm"] - rb["gf_pm"]
            df.loc[orig_idx, "t_ga_pm_diff"] = ra["ga_pm"] - rb["ga_pm"]
            df.loc[orig_idx, "t_win_rate_diff"] = ra["win"] - rb["win"]
            df.loc[orig_idx, "t_cs_rate_diff"] = ra["cs"] - rb["cs"]
            df.loc[orig_idx, "t_fts_rate_diff"] = ra["fts"] - rb["fts"]
            df.loc[orig_idx, "t_margin_pm_diff"] = ra["margin"] - rb["margin"]
            df.loc[orig_idx, "t_adj_gd_pm_diff"] = ra["adj_gd"] - rb["adj_gd"]
            df.loc[orig_idx, "t_attack_elo_diff"] = sa["attack"] - sb["attack"]
            df.loc[orig_idx, "t_defense_elo_diff"] = sa["defense"] - sb["defense"]

            elo_implied_gd = elo_diff / ELO_PER_GOAL
            form_gd = (ra["adj_gd"] - rb["adj_gd"])
            # only meaningful once both teams have some tournament history
            if ma > 0 or mb > 0:
                df.loc[orig_idx, "form_prior_disagreement"] = form_gd - elo_implied_gd

            # ── post-match state update ──────────────────────────────────────
            g_a = int(row[goal_a_col])
            g_b = int(row[goal_b_col])

            # Elo-implied expected goals for each side of THIS match
            exp_a = float(np.clip(BASE_GOALS + 0.5 * elo_diff / ELO_PER_GOAL * 2, 0.3, 3.0))
            exp_b = float(np.clip(BASE_GOALS - 0.5 * elo_diff / ELO_PER_GOAL * 2, 0.3, 3.0))

            # Opponent strength factor for adjusted goal diff:
            # beating a 1900-Elo side counts ~1.36x a 1500-Elo side
            rat_a = float(row.get("rating_a_before", 1600.0) or 1600.0)
            rat_b = float(row.get("rating_b_before", 1600.0) or 1600.0)
            w_opp_for_a = rat_b / 1700.0
            w_opp_for_b = rat_a / 1700.0

            for team, s, gf, ga, exp_gf, exp_ga, w_opp in (
                (ta, sa, g_a, g_b, exp_a, exp_b, w_opp_for_a),
                (tb, sb, g_b, g_a, exp_b, exp_a, w_opp_for_b),
            ):
                s["m"] += 1
                s["gf"] += gf
                s["ga"] += ga
                s["wins"] += int(gf > ga)
                s["draws"] += int(gf == ga)
                s["cs"] += int(ga == 0)
                s["fts"] += int(gf == 0)
                s["margin_sum"] += (gf - ga)
                s["adj_gd_sum"] += (gf - ga) * w_opp
                s["attack"] += ATTACK_K * (gf - exp_gf)
                s["defense"] += ATTACK_K * (ga - exp_ga)
                states[team] = s

    return df


def add_raw_tournament_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add raw per-team in-tournament counts (NOT model features — used by the
    lambda-level Poisson-Gamma posterior blend):
        t_m_a, t_gf_a, t_ga_a, t_m_b, t_gf_b, t_ga_b
    Leakage-safe: counts only include matches strictly before the row.
    """
    df = df.copy()
    for col in ("t_m_a", "t_gf_a", "t_ga_a", "t_m_b", "t_gf_b", "t_ga_b"):
        df[col] = 0.0

    team_a_col = "team_A" if "team_A" in df.columns else "team_a"
    team_b_col = "team_B" if "team_B" in df.columns else "team_b"
    goal_a_col = "goals_A" if "goals_A" in df.columns else "target_goals_a"
    goal_b_col = "goals_B" if "goals_B" in df.columns else "target_goals_b"

    df_sorted = df.sort_values("date").reset_index(drop=False)
    for (comp, year), group in df_sorted.groupby(["competition", "tournament_year"], sort=False):
        if comp not in MAJOR_TOURNAMENTS:
            continue
        st: dict[str, list] = {}
        for _, row in group.sort_values("date").iterrows():
            idx = row["index"]
            ta, tb = row[team_a_col], row[team_b_col]
            sa = st.get(ta, [0, 0, 0])
            sb = st.get(tb, [0, 0, 0])
            df.loc[idx, ["t_m_a", "t_gf_a", "t_ga_a"]] = sa
            df.loc[idx, ["t_m_b", "t_gf_b", "t_ga_b"]] = sb
            g_a, g_b = int(row[goal_a_col]), int(row[goal_b_col])
            st[ta] = [sa[0] + 1, sa[1] + g_a, sa[2] + g_b]
            st[tb] = [sb[0] + 1, sb[1] + g_b, sb[2] + g_a]
    return df


def posterior_blend_lambda(lam: float, gf: float, m: float, k: float) -> float:
    """
    Poisson-Gamma posterior: treat λ_model as a Gamma prior worth k pseudo-matches,
    update with observed tournament goals. λ_post = (gf + k·λ) / (m + k).
    m=0 → λ unchanged. Larger k → trust the model more, form less.
    """
    if m <= 0 or k <= 0:
        return lam
    return (gf + k * lam) / (m + k)


# ── V6 feature sets ─────────────────────────────────────────────────────────────
def get_v6_feature_sets() -> dict[str, list[str]]:
    """Named feature sets for the V6 experiments. F_V5 reproduces V5-prod (control)."""
    from src.features.feature_columns import FEATURE_COLS_V5_PROD

    f_v5 = list(FEATURE_COLS_V5_PROD)
    f_form = f_v5 + V6_FORM_COLS
    f_eloupd = f_form + V6_ELOUPD_COLS
    f_cap = [
        ("elo_diff_capped" if c == "elo_diff" else
         "rank_diff_log" if c == "rank_diff" else c)
        for c in f_eloupd
    ]
    f_norank = [c for c in f_eloupd if c != "rank_diff"]

    return {
        "F_V5": f_v5,            # control — V5-prod features, correct mirror
        "F_FORM": f_form,        # + 7 leakage-safe tournament form diffs
        "F_ELOUPD": f_eloupd,    # + attack/defense Elo updates + disagreement
        "F_CAP": f_cap,          # F_ELOUPD with capped elo / log rank
        "F_NORANK": f_norank,    # F_ELOUPD without rank_diff
    }


# ── V6 mirror (correct for all V6 columns) ─────────────────────────────────────
# Diff columns (negate on mirror) — superset; intersected with actual columns.
V6_DIFF_COLS = [
    "rank_diff", "elo_diff", "elo_diff_capped", "rank_diff_log",
    "avg_player_value_diff", "opponent_strength_diff_last5",
    "weighted_goals_for_diff_last5", "weighted_goals_against_diff_last5",
    "market_value_rel_mean_diff", "rating_change_diff_last5",
    "defender_share_diff", "goalkeeper_share_diff",
    "rest_diff",  # ← missing from V4's _DIFF_COLS; the V5-prod asymmetry bug
    "tournament_goal_diff_diff", "tournament_points_diff",
    "tournament_goals_for_per_match_diff", "tournament_goals_against_per_match_diff",
] + V6_FORM_COLS + V6_ELOUPD_COLS

# Paired columns (swap on mirror)
V6_PAIRED_COLS = [
    ("rating_a_before", "rating_b_before"),
    ("team_a_matches_played_before", "team_b_matches_played_before"),
    ("team_a_days_since_last_match", "team_b_days_since_last_match"),
    ("team_a_tournament_matches_played", "team_b_tournament_matches_played"),
]


def mirror_features_v6(X: pd.DataFrame) -> pd.DataFrame:
    """Team-swap mirror that correctly handles every V6 (and V5-prod) column."""
    X_m = X.copy()
    for col in V6_DIFF_COLS:
        if col in X_m.columns:
            X_m[col] = -X_m[col]
    for col_a, col_b in V6_PAIRED_COLS:
        if col_a in X_m.columns and col_b in X_m.columns:
            X_m[col_a], X_m[col_b] = X_m[col_b].copy(), X_m[col_a].copy()
    return X_m
