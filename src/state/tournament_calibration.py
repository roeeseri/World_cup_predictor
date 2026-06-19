"""
Bayesian Poisson calibration for WC 2026 live tournament.

How it works
------------
Two independent Bayesian updates run in parallel as real results come in:

1. GOAL RATE CALIBRATION  (prior: model is unbiased, ratio actual/predicted = 1.0)
   ─────────────────────
   The model outputs expected goals λ_a and λ_b.  If this WC is systematically
   higher- or lower-scoring than the model expects, we detect it here.

   After n completed games where we stored the model's pre-match λ predictions:
     obs_ratio  = Σ actual_total_goals / Σ predicted_total_goals

   Bayesian weighted average with prior weight = PRIOR_N_EFFECTIVE:
     goal_scale = (PRIOR_N × 1.0  +  n × obs_ratio) / (PRIOR_N + n)

   λ_a_cal = λ_a × goal_scale,  λ_b_cal = λ_b × goal_scale.

   With PRIOR_N = 48 (≈ one WC group stage): after 10 games the prior dominates,
   after 48 games they're equal weight, after 96 games the tournament data has 2×
   the weight of the prior.

2. DRAW RATE CALIBRATION  (Beta-Bernoulli conjugate)
   ──────────────────────
   Draws from an independent-Poisson model are systematically underpredicted in
   practice.  We track the tournament draw rate and update a Beta prior.

   Prior: Beta(α_d, β_d) where α_d/( α_d+β_d ) = historical draw rate,
          effective strength = PRIOR_N games.

   After n games with D observed draws:
     posterior_draw_rate = (α_d + D) / (α_d + β_d + n)

   The Poisson model's average predicted draw probability over the n games:
     model_mean_draw_rate = Σ P(draw | λ_a, λ_b) / n

   draw_adj = posterior_draw_rate / model_mean_draw_rate

   Applied to the output win/draw/loss probabilities: draw × draw_adj, then
   the excess is taken proportionally from win_a and win_b before renormalising.

PRIOR SOURCE
────────────
FIFA World Cup 2006-2022 group-stage + Euro 2024 + Copa America 2024.
Computed from the elo_YEAR_results.csv files at runtime (falls back to hardcoded
defaults if files are missing).

PERSISTENCE
───────────
Each time a real result is submitted through the UI, the model's pre-match
λ predictions are saved alongside the actual score in
  data/raw/world_cup_updates/calibration_predictions.csv

This file is loaded on app startup so calibration survives session restarts.
Clearing results via the UI also clears this file.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRIOR_N_EFFECTIVE = 48          # prior "worth" this many games
_DEFAULT_GOALS_PER_GAME = 2.50
_DEFAULT_DRAW_RATE = 0.225
_CALIBRATION_COLS = ["match_id", "pred_la", "pred_lb", "pred_goals_a", "pred_goals_b", "actual_a", "actual_b"]


# ---------------------------------------------------------------------------
# Prior loading
# ---------------------------------------------------------------------------

def _is_target_competition(comp: str, year: int) -> bool:
    """True for WC 2006-2022 (non-qualifier) and Euro/Copa 2024."""
    c = comp.lower()
    if "world cup" in c and "qualifier" not in c and "qualification" not in c:
        return 2006 <= year <= 2022
    if "euro" in c and "qualifier" not in c and year == 2024:
        return True
    if "copa america" in c and year == 2024:
        return True
    return False


def load_prior(raw_dir: str | Path = "data/raw") -> dict:
    """
    Fit prior from WC 2006-2022 and Euro/Copa 2024 matches.
    Falls back to hardcoded defaults if data files are unavailable.
    """
    raw_dir = Path(raw_dir)
    frames = []
    for year in range(2006, 2026):
        p = raw_dir / f"elo_{year}_results.csv"
        if p.exists():
            try:
                df = pd.read_csv(p)
                df["_year"] = year
                frames.append(df)
            except Exception:
                pass

    if not frames:
        return _make_prior(_DEFAULT_GOALS_PER_GAME, _DEFAULT_DRAW_RATE, 0)

    combined = pd.concat(frames, ignore_index=True)
    comp_col = combined.get("competition", pd.Series([""] * len(combined), dtype=str)).fillna("").astype(str)
    year_col = combined["_year"]

    mask = [_is_target_competition(c, int(y)) for c, y in zip(comp_col, year_col)]
    major = combined[mask].copy()

    if len(major) < 10:
        return _make_prior(_DEFAULT_GOALS_PER_GAME, _DEFAULT_DRAW_RATE, 0)

    goals_a = pd.to_numeric(major.get("goals_a", 0), errors="coerce").fillna(0)
    goals_b = pd.to_numeric(major.get("goals_b", 0), errors="coerce").fillna(0)
    total_goals = (goals_a + goals_b).values
    draws = (goals_a == goals_b).astype(int).values

    return _make_prior(
        float(np.mean(total_goals)),
        float(np.mean(draws)),
        n_source_games=int(len(major)),
    )


def _make_prior(goals_per_game: float, draw_rate: float, n_source_games: int = 0) -> dict:
    n = PRIOR_N_EFFECTIVE
    return {
        "goals_per_game": goals_per_game,
        "draw_rate": draw_rate,
        "goal_prior_n": n,
        "draw_alpha": draw_rate * n,
        "draw_beta": (1.0 - draw_rate) * n,
        "n_source_games": n_source_games,
    }


# ---------------------------------------------------------------------------
# Calibration state helpers
# ---------------------------------------------------------------------------

def initialize_calibration(prior: dict) -> dict:
    """Create a fresh calibration state from a prior."""
    return {"prior": prior, "games": []}


def _poisson_draw_prob(lambda_a: float, lambda_b: float, max_k: int = 12) -> float:
    """P(draw) under independent Poisson(λ_a), Poisson(λ_b)."""
    from scipy.stats import poisson as _poisson
    ks = np.arange(max_k + 1)
    return float(np.sum(
        _poisson.pmf(ks, max(lambda_a, 1e-6)) *
        _poisson.pmf(ks, max(lambda_b, 1e-6))
    ))


def add_game(
    calibration: dict,
    match_id: int,
    pred_lambda_a: float,
    pred_lambda_b: float,
    actual_goals_a: int,
    actual_goals_b: int,
    pred_goals_a: int | None = None,
    pred_goals_b: int | None = None,
) -> dict:
    """
    Record one completed game.  Idempotent on match_id.
    Returns a new calibration dict (original is not mutated).
    pred_goals_a/b are the model's integer score prediction (optional but needed
    for displaying predicted vs actual scores in the UI).
    """
    cal = copy.deepcopy(calibration)

    if any(g["match_id"] == int(match_id) for g in cal["games"]):
        return cal  # already recorded

    pred_draw_prob = _poisson_draw_prob(pred_lambda_a, pred_lambda_b)

    cal["games"].append({
        "match_id": int(match_id),
        "pred_la": float(pred_lambda_a),
        "pred_lb": float(pred_lambda_b),
        "pred_total": float(pred_lambda_a + pred_lambda_b),
        "pred_draw_prob": pred_draw_prob,
        "pred_goals_a": int(pred_goals_a) if pred_goals_a is not None else None,
        "pred_goals_b": int(pred_goals_b) if pred_goals_b is not None else None,
        "actual_a": int(actual_goals_a),
        "actual_b": int(actual_goals_b),
        "actual_total": int(actual_goals_a + actual_goals_b),
        "is_draw": int(actual_goals_a == actual_goals_b),
    })

    return cal


# ---------------------------------------------------------------------------
# Factor computation
# ---------------------------------------------------------------------------

def get_factors(calibration: dict) -> dict:
    """
    Compute both calibration factors from current state.

    Returns
    -------
    goal_scale         : multiply every λ by this before prediction
    draw_adj           : multiply the Poisson draw probability by this (then renorm)
    n_games            : WC 2026 games with stored predictions
    obs_goals_per_game : actual average total goals (None if no games yet)
    pred_goals_per_game: model-predicted average total goals (None if no games yet)
    obs_draw_rate      : actual draw fraction (None if no games yet)
    model_draw_rate    : model's mean predicted draw probability
    posterior_draw_rate: Beta posterior draw rate
    prior_goals        : historical prior mean goals/game
    prior_draw_rate    : historical prior draw rate
    prior_n            : effective prior weight in games
    """
    prior = calibration["prior"]
    games = calibration["games"]
    n = len(games)

    # ── Goal rate ──
    goal_prior_n = float(prior["goal_prior_n"])

    if n == 0:
        goal_scale = 1.0
        obs_goals_per_game = None
        pred_goals_per_game = None
    else:
        total_actual    = sum(g["actual_total"] for g in games)
        total_predicted = sum(g["pred_total"]   for g in games)
        obs_goals_per_game  = total_actual    / n
        pred_goals_per_game = total_predicted / n

        obs_ratio  = total_actual / max(total_predicted, 1e-6)
        # Bayesian blend: prior ratio = 1.0 (model unbiased), weight = goal_prior_n
        goal_scale = (goal_prior_n * 1.0 + n * obs_ratio) / (goal_prior_n + n)

    goal_scale = float(np.clip(goal_scale, 0.5, 2.0))

    # ── Draw rate ──
    alpha_d = float(prior["draw_alpha"])
    beta_d  = float(prior["draw_beta"])
    n_draws = sum(g["is_draw"] for g in games)

    posterior_draw_rate = (alpha_d + n_draws) / (alpha_d + beta_d + n)

    if n == 0:
        model_draw_rate = float(prior["draw_rate"])
        obs_draw_rate   = None
    else:
        model_draw_rate = sum(g["pred_draw_prob"] for g in games) / n
        obs_draw_rate   = n_draws / n

    draw_adj = posterior_draw_rate / max(model_draw_rate, 1e-6)
    draw_adj = float(np.clip(draw_adj, 0.3, 3.0))

    return {
        "goal_scale":          goal_scale,
        "draw_adj":            draw_adj,
        "n_games":             n,
        "obs_goals_per_game":  obs_goals_per_game,
        "pred_goals_per_game": pred_goals_per_game,
        "obs_draw_rate":       obs_draw_rate,
        "model_draw_rate":     model_draw_rate,
        "posterior_draw_rate": posterior_draw_rate,
        "prior_goals":         float(prior["goals_per_game"]),
        "prior_draw_rate":     float(prior["draw_rate"]),
        "prior_n":             int(prior["goal_prior_n"]),
    }


# ---------------------------------------------------------------------------
# Applying calibration
# ---------------------------------------------------------------------------

def calibrate_lambdas(
    lambda_a: float,
    lambda_b: float,
    calibration: dict,
) -> tuple[float, float]:
    """Scale model λ predictions by the goal rate calibration factor."""
    scale = get_factors(calibration)["goal_scale"]
    return float(lambda_a * scale), float(lambda_b * scale)


def calibrate_win_draw_loss(
    win_a: float,
    draw: float,
    win_b: float,
    calibration: dict,
) -> tuple[float, float, float]:
    """
    Adjust (win_a, draw, win_b) probabilities by the draw calibration.
    Excess draw probability is taken proportionally from win_a and win_b.
    """
    draw_adj = get_factors(calibration)["draw_adj"]

    raw_total  = win_a + draw + win_b
    draw_cal   = min(draw * draw_adj, raw_total * 0.95)
    excess     = draw_cal - draw
    total_wins = win_a + win_b

    if total_wins > 1e-9:
        win_a_cal = win_a - excess * win_a / total_wins
        win_b_cal = win_b - excess * win_b / total_wins
    else:
        win_a_cal = win_a
        win_b_cal = win_b

    win_a_cal = max(win_a_cal, 0.0)
    win_b_cal = max(win_b_cal, 0.0)

    total = win_a_cal + draw_cal + win_b_cal
    if total < 1e-9:
        return win_a, draw, win_b

    return win_a_cal / total, draw_cal / total, win_b_cal / total


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------

def load_calibration_from_csv(calibration: dict, path: str | Path) -> dict:
    """Merge stored game records from CSV into the calibration state."""
    path = Path(path)
    if not path.exists():
        return calibration
    try:
        df = pd.read_csv(path)
        if df.empty:
            return calibration
        for _, row in df.iterrows():
            pa = int(row["pred_goals_a"]) if "pred_goals_a" in row and pd.notna(row["pred_goals_a"]) else None
            pb = int(row["pred_goals_b"]) if "pred_goals_b" in row and pd.notna(row["pred_goals_b"]) else None
            calibration = add_game(
                calibration,
                int(row["match_id"]),
                float(row["pred_la"]),
                float(row["pred_lb"]),
                int(row["actual_a"]),
                int(row["actual_b"]),
                pred_goals_a=pa,
                pred_goals_b=pb,
            )
    except Exception:
        pass
    return calibration


def save_calibration_to_csv(calibration: dict, path: str | Path) -> None:
    """Write all game records to CSV (full overwrite)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "match_id":    g["match_id"],
            "pred_la":     g["pred_la"],
            "pred_lb":     g["pred_lb"],
            "pred_goals_a": g.get("pred_goals_a"),
            "pred_goals_b": g.get("pred_goals_b"),
            "actual_a":    g["actual_a"],
            "actual_b":    g["actual_b"],
        }
        for g in calibration["games"]
    ]
    pd.DataFrame(rows, columns=_CALIBRATION_COLS).to_csv(path, index=False)


def clear_calibration_csv(path: str | Path) -> None:
    """Write an empty calibration CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=_CALIBRATION_COLS).to_csv(path, index=False)
