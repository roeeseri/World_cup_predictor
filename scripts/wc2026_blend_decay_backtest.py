#!/usr/bin/env python3
"""
Backtest: V4+V6 blend model across multiple calibration configurations on the first 24 WC 2026 games.

Processes games chronologically. State (ELO, form, tournament table) is always updated with
actual results. Each calibration config runs in parallel with its own calibration state.

Configs tested:
    std-cal             standard (equal-weight) Bayesian calibration
    decay(5)            exponential half-life 5 games
    decay(5)+prior(16)  decay-5 with tighter prior (16 effective games instead of 48)
    decay(10)           exponential half-life 10 games
    decay(10)+prior(16) decay-10 with tighter prior
    decay(1)+prior(16)  very aggressive decay with tighter prior

Usage:
    python scripts/wc2026_blend_decay_backtest.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.feature_columns import FEATURE_COLS, FEATURE_COLS_V5_PROD
from src.features.build_features import build_pre_match_features, build_pre_match_features_v5
from src.features.team_names import normalize_team_name
from src.models.ensemble import EnsembleGoalModel
from src.models.lgbm_model import LGBMGoalModel
from src.models.xgb_model import XGBGoalModel
from src.models.goal_models_v6 import V6LGBMGoalModel, V6XGBGoalModel
from src.models.base import load_model_dataset
from src.models.score_conversion import most_likely_score_v6, win_draw_loss_probs
from src.models.weighting import apply_competition_weights, get_competition_weight
from src.state.live_state import initialize_live_state, record_match_result
from src.state.tournament_calibration import (
    load_prior, initialize_calibration, add_game, get_factors as get_factors_std,
)


# ── Calibration configurations ─────────────────────────────────────────────────

CONFIGS = [
    {"label": "std-cal",             "halflife": None, "prior_n": 36},
    {"label": "decay(5)",            "halflife": 5.0,  "prior_n": 48},
    {"label": "decay(5)+prior(16)",  "halflife": 5.0,  "prior_n": 16},
    {"label": "decay(10)",           "halflife": 10.0, "prior_n": 48},
    {"label": "decay(10)+prior(16)", "halflife": 10.0, "prior_n": 16},
    {"label": "decay(1)+prior(16)",  "halflife": 1.0,  "prior_n": 16},
]

# ── Other constants ─────────────────────────────────────────────────────────────

BLEND_WEIGHT_V4 = 0.5
COMPETITION_IMPORTANCE_WC = 4.0

V6_PARAMS = {
    "draw_threshold": 0.33,
    "threshold_b": 0.5,
    "scale_c": 0.9992361938714952,
    "rho": -0.3293651939032305,
}

BLEND_FEATURE_COLS = list(dict.fromkeys(FEATURE_COLS + FEATURE_COLS_V5_PROD))


# ── Model ──────────────────────────────────────────────────────────────────────

class BlendedGoalModel:
    """50/50 λ blend of V4 and V6 ensembles; scored by V6 drawband."""

    def __init__(self, model_v4, model_v6, weight_v4: float = BLEND_WEIGHT_V4):
        self.model_v4 = model_v4
        self.model_v6 = model_v6
        self.weight_v4 = weight_v4

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pa = self.model_v4.predict(X[FEATURE_COLS].fillna(0))
        pb = self.model_v6.predict(X[FEATURE_COLS_V5_PROD].fillna(0))
        return self.weight_v4 * pa + (1.0 - self.weight_v4) * pb


# ── Calibration helpers ────────────────────────────────────────────────────────

def _make_prior_with_n(base_prior: dict, prior_n: int) -> dict:
    """Return a copy of base_prior with goal_prior_n and draw Beta params rescaled."""
    dr = float(base_prior["draw_rate"])
    return {
        "goals_per_game": base_prior["goals_per_game"],
        "draw_rate": dr,
        "goal_prior_n": prior_n,
        "draw_alpha": dr * prior_n,
        "draw_beta": (1.0 - dr) * prior_n,
        "n_source_games": base_prior["n_source_games"],
    }


def _get_factors_decayed(calibration: dict, halflife: float) -> dict:
    """Calibration factors with exponential decay (most recent game = weight 1.0)."""
    prior = calibration["prior"]
    games = calibration["games"]
    n = len(games)

    if n == 0:
        return {"goal_scale": 1.0, "draw_adj": 1.0}

    weights = np.array([2.0 ** (-(n - 1 - i) / halflife) for i in range(n)])
    w_sum = float(weights.sum())

    goal_prior_n = float(prior["goal_prior_n"])
    pred_totals = np.array([g["pred_total"] for g in games])
    actual_totals = np.array([g["actual_total"] for g in games])
    obs_ratio = float((weights * actual_totals).sum()) / max(float((weights * pred_totals).sum()), 1e-6)
    goal_scale = (goal_prior_n + w_sum * obs_ratio) / (goal_prior_n + w_sum)
    goal_scale = float(np.clip(goal_scale, 0.5, 2.0))

    alpha_d = float(prior["draw_alpha"])
    beta_d = float(prior["draw_beta"])
    is_draws = np.array([g["is_draw"] for g in games])
    pred_draw_probs = np.array([g["pred_draw_prob"] for g in games])
    weighted_draws = float((weights * is_draws).sum())
    posterior_draw_rate = (alpha_d + weighted_draws) / (alpha_d + beta_d + w_sum)
    model_draw_rate = float((weights * pred_draw_probs).sum()) / w_sum
    draw_adj = posterior_draw_rate / max(model_draw_rate, 1e-6)
    draw_adj = float(np.clip(draw_adj, 0.3, 3.0))

    return {"goal_scale": goal_scale, "draw_adj": draw_adj}


def _get_factors(calibration: dict, halflife) -> dict:
    """Dispatch to decayed or standard calibration."""
    if halflife is None:
        return get_factors_std(calibration)
    return _get_factors_decayed(calibration, halflife)


def _apply_calibration(la_raw: float, lb_raw: float, calibration: dict, halflife):
    """Return (la_cal, lb_cal, win_a, draw, win_b) after calibration."""
    fac = _get_factors(calibration, halflife)
    la_cal = float(np.clip(la_raw * fac["goal_scale"], 0.1, 20))
    lb_cal = float(np.clip(lb_raw * fac["goal_scale"], 0.1, 20))
    win_a, draw, win_b = win_draw_loss_probs(la_cal, lb_cal)

    if len(calibration["games"]) > 0:
        draw_adj = fac["draw_adj"]
        raw_total = win_a + draw + win_b
        draw_cal = min(draw * draw_adj, raw_total * 0.95)
        excess = draw_cal - draw
        total_wins = win_a + win_b
        if total_wins > 1e-9:
            win_a = max(win_a - excess * win_a / total_wins, 0.0)
            win_b = max(win_b - excess * win_b / total_wins, 0.0)
        total = win_a + draw_cal + win_b
        if total > 1e-9:
            win_a, draw_cal, win_b = win_a / total, draw_cal / total, win_b / total
        draw = draw_cal

    return la_cal, lb_cal, win_a, draw, win_b


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_historical() -> pd.DataFrame:
    frames = []
    for year in range(2001, 2027):
        p = ROOT / "data" / "raw" / f"elo_{year}_results.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["_year"] = year
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.sort_values("date").reset_index(drop=True)
    # ELO CSVs store post-match ratings; recent_form_features expects pre-match columns
    if "rating_a_before" not in combined.columns:
        combined["rating_a_before"] = combined["rating_a"] - combined.get("rating_change_a", 0)
    if "rating_b_before" not in combined.columns:
        combined["rating_b_before"] = combined["rating_b"] - combined.get("rating_change_b", 0)
    return combined


def _load_fixtures() -> pd.DataFrame:
    fixtures = pd.read_csv(ROOT / "data" / "processed" / "world_cup_2026_group_stage_features.csv")
    fixtures["date"] = pd.to_datetime(fixtures["date"])
    fixtures["stage"] = "GROUPS"
    fixtures["location"] = "neutral"
    fixtures["is_completed"] = False
    fixtures["goals_a"] = float("nan")
    fixtures["goals_b"] = float("nan")
    return fixtures


def _prepare_training_data() -> pd.DataFrame:
    df = load_model_dataset(ROOT / "data" / "processed" / "updated_model_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["rest_diff"] = df["team_a_days_since_last_match"] - df["team_b_days_since_last_match"]
    df["competition_importance"] = df["competition"].apply(get_competition_weight)
    return df


# ── Training ───────────────────────────────────────────────────────────────────

def train_blend_model(df: pd.DataFrame) -> BlendedGoalModel:
    weights = apply_competition_weights(df)
    y = df[["goals_A", "goals_B"]].values

    print("  Training V4 ensemble (LGBM 0.9 + XGB 0.1)...")
    model_v4 = EnsembleGoalModel([LGBMGoalModel(), XGBGoalModel()], weights=[0.9, 0.1])
    model_v4.fit(df[FEATURE_COLS].fillna(0), y, sample_weight=weights)

    print("  Training V6 ensemble (LGBM 0.9 + XGB 0.1, corrected mirror)...")
    model_v6 = EnsembleGoalModel([V6LGBMGoalModel(), V6XGBGoalModel()], weights=[0.9, 0.1])
    model_v6.fit(df[FEATURE_COLS_V5_PROD].fillna(0), y, sample_weight=weights)

    return BlendedGoalModel(model_v4, model_v6, BLEND_WEIGHT_V4)


# ── Feature building ───────────────────────────────────────────────────────────

def _build_features(
    team_a: str,
    team_b: str,
    match_date,
    state: dict,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
) -> pd.DataFrame:
    kwargs = dict(
        team_a=team_a, team_b=team_b, match_date=match_date,
        team_states=state["team_states"],
        historical_matches=state["historical_matches"],
        market_values=market_values,
        position_values=position_values,
        elo_ratings=state["elo_ratings"],
        rankings=state["rankings"],
    )
    X_v4 = build_pre_match_features(**kwargs)
    X_v6 = build_pre_match_features_v5(**kwargs, competition_importance=COMPETITION_IMPORTANCE_WC)
    X = X_v4.copy()
    X["rest_diff"] = float(X_v6["rest_diff"].iloc[0])
    X["competition_importance"] = float(X_v6["competition_importance"].iloc[0])
    return X[BLEND_FEATURE_COLS]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("WC 2026 Backtest: V4+V6 Blend | Multiple Calibration Configs")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    historical = _load_historical()
    fixtures = _load_fixtures()
    market_values = pd.read_csv(ROOT / "data" / "processed" / "transfermarkt_market_values_clean.csv")
    position_values = pd.read_csv(ROOT / "data" / "processed" / "transfermarkt_position_values_2004_2026.csv")
    updates = pd.read_csv(ROOT / "data" / "raw" / "world_cup_updates" / "all_world_cup_2026_updates.csv")
    updates["date"] = pd.to_datetime(updates["date"])
    updates = updates.dropna(subset=["goals_a", "goals_b"]).sort_values("date").reset_index(drop=True)
    print(f"  Historical matches: {len(historical):,}  |  WC 2026 games: {len(updates)}")

    # Initialize live state
    print("\n[2/4] Initializing live state...")
    state = initialize_live_state(historical, fixtures)

    # Train
    print("\n[3/4] Training blend model on updated_model_dataset.csv...")
    df_train = _prepare_training_data()
    print(f"  Dataset: {len(df_train):,} matches")
    blend_model = train_blend_model(df_train)

    # Initialize one calibration state per config
    base_prior = load_prior(ROOT / "data" / "raw")
    print(f"\n[4/4] Prior: {base_prior['goals_per_game']:.3f} goals/g, "
          f"{base_prior['draw_rate']:.3f} draw rate (n={base_prior['n_source_games']})")

    calibrations = {}
    for cfg in CONFIGS:
        prior = _make_prior_with_n(base_prior, cfg["prior_n"])
        calibrations[cfg["label"]] = initialize_calibration(prior)

    # Records: per-config list of per-game dicts
    all_records = {cfg["label"]: [] for cfg in CONFIGS}

    print(f"\nProcessing {len(updates)} games across {len(CONFIGS)} configs...\n")

    for row in updates.itertuples():
        team_a = normalize_team_name(row.team_a)
        team_b = normalize_team_name(row.team_b)
        match_date = row.date
        actual_a = int(row.goals_a)
        actual_b = int(row.goals_b)

        # Find fixture
        fix = fixtures[
            ((fixtures["team_a"] == team_a) & (fixtures["team_b"] == team_b)) |
            ((fixtures["team_a"] == team_b) & (fixtures["team_b"] == team_a))
        ]
        if fix.empty:
            print(f"  WARNING: no fixture for {team_a} vs {team_b}")
            continue

        fix_row = fix.iloc[0]
        match_id = int(fix_row["match_id"])
        fixture_a = fix_row["team_a"]
        fixture_b = fix_row["team_b"]
        reversed_order = fixture_a == team_b
        actual_fa = actual_b if reversed_order else actual_a
        actual_fb = actual_a if reversed_order else actual_b

        # Build features and get raw λ (same for all configs)
        try:
            X = _build_features(fixture_a, fixture_b, match_date, state, market_values, position_values)
        except Exception as exc:
            print(f"  WARNING: feature build failed for {fixture_a} vs {fixture_b}: {exc}")
            continue

        pred = blend_model.predict(X)
        la_raw, lb_raw = float(pred[0, 0]), float(pred[0, 1])

        # Apply each calibration config
        for cfg in CONFIGS:
            label = cfg["label"]
            halflife = cfg["halflife"]
            cal = calibrations[label]

            la_cal, lb_cal, win_a, draw, win_b = _apply_calibration(la_raw, lb_raw, cal, halflife)
            pred_fa, pred_fb = most_likely_score_v6(la_cal, lb_cal, **V6_PARAMS)

            exact = pred_fa == actual_fa and pred_fb == actual_fb
            pred_res = "W" if pred_fa > pred_fb else ("D" if pred_fa == pred_fb else "L")
            true_res = "W" if actual_fa > actual_fb else ("D" if actual_fa == actual_fb else "L")

            fac = _get_factors(cal, halflife)
            all_records[label].append({
                "match": f"{fixture_a} v {fixture_b}",
                "pred": f"{pred_fa}-{pred_fb}",
                "actual": f"{actual_fa}-{actual_fb}",
                "la": round(la_cal, 2),
                "lb": round(lb_cal, 2),
                "Gscale": round(fac["goal_scale"], 3),
                "W%": round(win_a * 100),
                "D%": round(draw * 100),
                "L%": round(win_b * 100),
                "exact": exact,
                "correct_result": pred_res == true_res,
            })

            # Update this config's calibration
            calibrations[label] = add_game(
                cal, match_id, la_raw, lb_raw, actual_fa, actual_fb, pred_fa, pred_fb
            )

        # Update live state (same for all configs — actual results only)
        state = record_match_result(state, match_id, actual_fa, actual_fb)

    # ── Summary comparison table ──
    n = len(updates)
    summary_rows = []
    for cfg in CONFIGS:
        label = cfg["label"]
        recs = all_records[label]
        n_exact = sum(1 for r in recs if r["exact"])
        n_result = sum(1 for r in recs if r["correct_result"])
        fac = _get_factors(calibrations[label], cfg["halflife"])
        summary_rows.append({
            "Config": label,
            "Exact": n_exact,
            "Exact%": f"{100*n_exact/n:.1f}%",
            "Result": n_result,
            "Result%": f"{100*n_result/n:.1f}%",
            "Gscale": f"{fac['goal_scale']:.4f}",
            "DrawAdj": f"{fac['draw_adj']:.4f}",
        })

    print("\n" + "=" * 65)
    print(f"SUMMARY  ({n} games)")
    print("=" * 65)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # ── Best config by correct result % ──
    best_label = max(CONFIGS, key=lambda c: sum(1 for r in all_records[c["label"]] if r["correct_result"]))["label"]
    best_exact = sum(1 for r in all_records[best_label] if r["exact"])
    best_result = sum(1 for r in all_records[best_label] if r["correct_result"])

    print(f"\n{'=' * 65}")
    print(f"BEST CONFIG: {best_label}  (Exact {best_exact}/{n}, Result {best_result}/{n})")
    print(f"{'=' * 65}")

    recs = all_records[best_label]
    table = pd.DataFrame([{
        "#": i + 1,
        "Match": r["match"][:28],
        "Pred": r["pred"],
        "Actual": r["actual"],
        "la": r["la"],
        "lb": r["lb"],
        "Gscale": r["Gscale"],
        "W%": r["W%"],
        "D%": r["D%"],
        "L%": r["L%"],
        "Exact": "Y" if r["exact"] else "-",
        "Result": "Y" if r["correct_result"] else "-",
    } for i, r in enumerate(recs)])

    pd.set_option("display.width", 150)
    pd.set_option("display.max_colwidth", 30)
    print(table.to_string(index=False))

    fac = _get_factors(calibrations[best_label], next(c["halflife"] for c in CONFIGS if c["label"] == best_label))
    print(f"\nFinal calibration state: goal_scale={fac['goal_scale']:.4f}  draw_adj={fac['draw_adj']:.4f}")


if __name__ == "__main__":
    main()
