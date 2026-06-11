"""
Form-vs-rank conflict scenario test + team-swap symmetry check for V6.

Scenario: Team A is higher-ranked (elo +120, better rank) but has a weak
tournament (0-0 draw, 0-1 loss). Team B is lower-ranked but won 3-0 and 3-0.
A good model should now see B as at least equal. V4/V5 historically still
leaned ~1.17-0.79 toward Team A.

Usage:
    python scripts/v6_scenario_test.py <fset>      (e.g. F_ELOUPD)

Trains the V6 ensemble on all-except-WC2022 (fresh; prod artifacts untouched)
and contrasts its lambdas with the V5 production model on the same scenario.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.protocol import create_holdout_split
from src.experiments.v6_eval import TARGET_COLS
from src.experiments.v6_features import SHRINK_TAU, ATTACK_K, ELO_PER_GOAL, BASE_GOALS, get_v6_feature_sets, mirror_features_v6
from src.experiments.v6_models import build_v6_ensemble
from src.features.feature_columns import FEATURE_COLS_V5_PROD


def shrink(rate: float, m: int) -> float:
    return rate * m / (m + SHRINK_TAU)


def build_scenario_row(cols: list[str]) -> pd.DataFrame:
    """Team A: elo 1820 / rank 8, drew 0-0 + lost 0-1.  Team B: elo 1700 / rank 28, won 3-0 twice."""
    elo_a, elo_b = 1820.0, 1700.0
    elo_diff = elo_a - elo_b           # +120 → A favored pre-tournament
    rank_diff = 8 - 28                 # -20 → A better ranked

    m = 2
    # Team A tournament: 0-0, 0-1 → gf 0, ga 1, 0 wins, 1 cs, 2 fts, margin -1
    a = dict(gf_pm=0.0, ga_pm=0.5, win=0.0, cs=0.5, fts=1.0, margin=-0.5)
    # Team B tournament: 3-0, 3-0 → gf 6, ga 0, 2 wins, 2 cs, 0 fts, margin +3
    b = dict(gf_pm=3.0, ga_pm=0.0, win=1.0, cs=1.0, fts=0.0, margin=3.0)

    # opponent-strength weight ~ mid-field opponents (1650 elo)
    w_opp = 1650.0 / 1700.0
    adj_a = shrink(a["margin"] * w_opp, m)
    adj_b = shrink(b["margin"] * w_opp, m)

    # attack/defense Elo updates (both teams faced ~equal-elo opponents)
    exp_gf_a = np.clip(BASE_GOALS + (elo_a - 1650) / ELO_PER_GOAL, 0.3, 3.0)
    exp_gf_b = np.clip(BASE_GOALS + (elo_b - 1650) / ELO_PER_GOAL, 0.3, 3.0)
    attack_a = ATTACK_K * ((0 - exp_gf_a) + (0 - exp_gf_a))
    attack_b = ATTACK_K * ((3 - exp_gf_b) + (3 - exp_gf_b))
    defense_a = ATTACK_K * ((0 - BASE_GOALS) + (1 - BASE_GOALS))
    defense_b = ATTACK_K * ((0 - BASE_GOALS) + (0 - BASE_GOALS))

    row = {
        "rank_diff": rank_diff,
        "elo_diff": elo_diff,
        "elo_diff_capped": float(np.clip(elo_diff, -250, 250)),
        "rank_diff_log": float(np.sign(rank_diff) * np.log1p(abs(rank_diff))),
        "rating_a_before": elo_a,
        "rating_b_before": elo_b,
        "avg_player_value_diff": 5.0,
        "opponent_strength_diff_last5": 50.0,
        "weighted_goals_for_diff_last5": 0.2,
        "weighted_goals_against_diff_last5": -0.1,
        "market_value_rel_mean_diff": 0.3,
        "rating_change_diff_last5": -15.0,
        "defender_share_diff": 0.0,
        "goalkeeper_share_diff": 0.0,
        "team_a_days_since_last_match": 4.0,
        "team_b_days_since_last_match": 4.0,
        "rest_diff": 0.0,
        "competition_importance": 4.0,
        "tournament_goal_diff_diff": -1 - 6,
        "tournament_points_diff": 1 - 6,
        "team_a_tournament_matches_played": m,
        "team_b_tournament_matches_played": m,
        "tournament_goals_for_per_match_diff": a["gf_pm"] - b["gf_pm"],
        "tournament_goals_against_per_match_diff": a["ga_pm"] - b["ga_pm"],
        # V6 form diffs (A minus B, shrunk)
        "t_gf_pm_diff": shrink(a["gf_pm"], m) - shrink(b["gf_pm"], m),
        "t_ga_pm_diff": shrink(a["ga_pm"], m) - shrink(b["ga_pm"], m),
        "t_win_rate_diff": shrink(a["win"], m) - shrink(b["win"], m),
        "t_cs_rate_diff": shrink(a["cs"], m) - shrink(b["cs"], m),
        "t_fts_rate_diff": shrink(a["fts"], m) - shrink(b["fts"], m),
        "t_margin_pm_diff": shrink(a["margin"], m) - shrink(b["margin"], m),
        "t_adj_gd_pm_diff": adj_a - adj_b,
        "t_attack_elo_diff": attack_a - attack_b,
        "t_defense_elo_diff": defense_a - defense_b,
        "form_prior_disagreement": (adj_a - adj_b) - elo_diff / ELO_PER_GOAL,
        "team_a_matches_played_before": 350.0,
        "team_b_matches_played_before": 350.0,
    }
    return pd.DataFrame([row])[[c for c in cols]]


def main() -> None:
    fset = sys.argv[1] if len(sys.argv) > 1 else "F_ELOUPD"
    fsets = get_v6_feature_sets()
    cols = fsets[fset]

    # ── V5 production model on the same scenario ──────────────────────────────
    v5 = joblib.load(ROOT / "models" / "production_model_v5.joblib")
    row_v5 = build_scenario_row(FEATURE_COLS_V5_PROD)
    p5 = v5.predict(row_v5)
    print(f"V5 prod        : lambda  A={p5[0,0]:.3f}  B={p5[0,1]:.3f}   "
          f"(A favored: {p5[0,0] > p5[0,1]})")

    # ── V6 fresh-trained on all-except-2022 ───────────────────────────────────
    sys.path.insert(0, str(ROOT / "scripts"))
    from v6_experiments import load_v6_dataset, weights_fn

    df = load_v6_dataset()
    train_df, _ = create_holdout_split(df)
    model = build_v6_ensemble()
    model.fit(train_df[cols].fillna(0), train_df[TARGET_COLS].values,
              sample_weight=weights_fn(train_df))

    row_v6 = build_scenario_row(cols)
    p6 = model.predict(row_v6)
    print(f"V6 {fset:11s}: lambda  A={p6[0,0]:.3f}  B={p6[0,1]:.3f}   "
          f"(A favored: {p6[0,0] > p6[0,1]})")

    # ── No-form control: same teams, matchday 1 (no tournament history) ──────
    row_ctrl = build_scenario_row(cols).copy()
    for c in row_ctrl.columns:
        if c.startswith("t_") or c.startswith("tournament_") or c == "form_prior_disagreement":
            row_ctrl[c] = 0.0
        if "tournament_matches_played" in c:
            row_ctrl[c] = 0.0
    p6c = model.predict(row_ctrl)
    print(f"V6 no-form ctrl: lambda  A={p6c[0,0]:.3f}  B={p6c[0,1]:.3f}   "
          f"(form swing on A: {p6[0,0]-p6c[0,0]:+.3f}, on B: {p6[0,1]-p6c[0,1]:+.3f})")

    # ── Team-swap symmetry check ──────────────────────────────────────────────
    row_sw = mirror_features_v6(row_v6)
    p_sw = model.predict(row_sw)
    print(f"swap-symmetry  : |dA|={abs(p6[0,0]-p_sw[0,1]):.6f}  |dB|={abs(p6[0,1]-p_sw[0,0]):.6f}")


if __name__ == "__main__":
    main()
