"""
V6 evaluation: full metric suite + decision rules applied to cached lambdas.

Decision rules all map (lambda_a, lambda_b [, is_knockout]) -> (goals_a, goals_b).
Calibration params for grid-based rules are cross-fitted on OOF folds
(fold i scored with params fit on folds j != i), matching PLAN_V5 discipline.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.models.score_grid import (
    dixon_coles_grid,
    knockout_grid,
    fit_lambda_scale,
    fit_rho,
    apply_lambda_scale,
)

TARGET_COLS = ["goals_A", "goals_B"]


# ── Metric suite ───────────────────────────────────────────────────────────────
def full_metrics(y_true: np.ndarray, y_pred: np.ndarray, ko_mask: np.ndarray | None = None) -> dict:
    """All metrics requested for the V6 comparison."""
    yt = y_true.astype(int)
    yp = y_pred.astype(int)

    exact = np.all(yt == yp, axis=1)
    true_sign = np.sign(yt[:, 0] - yt[:, 1])
    pred_sign = np.sign(yp[:, 0] - yp[:, 1])
    outcome = true_sign == pred_sign
    gd_ok = (yt[:, 0] - yt[:, 1]) == (yp[:, 0] - yp[:, 1])

    btts_true = (yt[:, 0] > 0) & (yt[:, 1] > 0)
    btts_pred = (yp[:, 0] > 0) & (yp[:, 1] > 0)

    m = {
        "n": len(yt),
        "exact_%": 100 * exact.mean(),
        "outcome_%": 100 * outcome.mean(),
        "gd_acc_%": 100 * gd_ok.mean(),
        "team_goals_mae": float(np.abs(yt - yp).mean()),
        "total_goals_mae": float(np.abs(yt.sum(1) - yp.sum(1)).mean()),
        "pred_draw_%": 100 * float((yp[:, 0] == yp[:, 1]).mean()),
        "actual_draw_%": 100 * float((yt[:, 0] == yt[:, 1]).mean()),
        "pred_btts_%": 100 * float(btts_pred.mean()),
        "actual_btts_%": 100 * float(btts_true.mean()),
    }

    # Top predicted scorelines
    counts: dict[str, int] = {}
    for a, b in yp:
        k = f"{a}-{b}"
        counts[k] = counts.get(k, 0) + 1
    m["top_pred_scores"] = dict(sorted(counts.items(), key=lambda x: -x[1])[:5])

    if ko_mask is not None and ko_mask.any():
        g = ~ko_mask
        m["group_exact_%"] = 100 * exact[g].mean() if g.any() else np.nan
        m["ko_exact_%"] = 100 * exact[ko_mask].mean()
        m["group_outcome_%"] = 100 * outcome[g].mean() if g.any() else np.nan
        m["ko_outcome_%"] = 100 * outcome[ko_mask].mean()
        m["ko_pred_draw_%"] = 100 * float((yp[ko_mask, 0] == yp[ko_mask, 1]).mean())
        m["ko_actual_draw_%"] = 100 * float((yt[ko_mask, 0] == yt[ko_mask, 1]).mean())

    return m


def knockout_mask(df: pd.DataFrame) -> np.ndarray:
    """KO rows: either team already played >= 3 tournament matches (16 per WC)."""
    return (
        (df["team_a_tournament_matches_played"] >= 3)
        | (df["team_b_tournament_matches_played"] >= 3)
    ).values


# ── Simple decision rules (no grid) ────────────────────────────────────────────
def rule_floor01(la: float, lb: float) -> tuple[int, int]:
    """V4 production rule."""
    return int(la + 0.1), int(lb + 0.1)


def rule_cond_floor(la: float, lb: float, tb: float = 0.5) -> tuple[int, int]:
    """V5 production rule: raise B's floor only when A is predicted to score 2+."""
    ga = int(la + 0.1)
    gb = int(lb + tb) if ga >= 2 else int(lb + 0.1)
    return ga, gb


def rule_draw_band(
    la: float,
    lb: float,
    rho: float,
    scale_c: float = 1.0,
    draw_threshold: float = 0.30,
    tb: float = 0.5,
) -> tuple[int, int]:
    """
    Draw-calibration hybrid: use the V5 conditional floor by default, but when the
    calibrated DC grid puts >= draw_threshold mass on a draw, predict the most
    likely draw scoreline instead. Targets the draw under-prediction (9% vs 25%)
    without giving up the floor rule's exact-score edge on decided games.
    """
    la_c, lb_c = apply_lambda_scale(np.array([la]), np.array([lb]), scale_c)
    grid = dixon_coles_grid(float(la_c[0]), float(lb_c[0]), rho=rho)
    draw_prob = float(np.trace(grid))
    if draw_prob >= draw_threshold:
        k = int(np.argmax(np.diag(grid)))
        return k, k
    return rule_cond_floor(la, lb, tb=tb)


# ── Grid-based rules ───────────────────────────────────────────────────────────
def grid_pick(
    la: float,
    lb: float,
    rho: float,
    scale_c: float = 1.0,
    draw_beta: float = 0.0,
    prior: np.ndarray | None = None,
    prior_w: float = 0.0,
    is_ko: bool = False,
    et_scale: float = 30.0 / 90.0,
    max_goals: int = 8,
) -> tuple[int, int]:
    """
    Calibrated DC grid argmax with optional draw bonus and empirical prior blending.

    draw_beta:  added to diagonal cell probabilities before argmax (draw calibration).
    prior:      (max_goals+1)^2 empirical WC scoreline distribution; blended as
                P' = (1-w)*P + w*prior  (linear, keeps mass normalized).
    """
    la_c, lb_c = apply_lambda_scale(np.array([la]), np.array([lb]), scale_c)
    la_c, lb_c = float(la_c[0]), float(lb_c[0])
    if is_ko:
        grid = knockout_grid(la_c, lb_c, rho=rho, et_scale=et_scale, max_goals=max_goals)
    else:
        grid = dixon_coles_grid(la_c, lb_c, rho=rho, max_goals=max_goals)

    if prior is not None and prior_w > 0:
        grid = (1 - prior_w) * grid + prior_w * prior

    if draw_beta != 0.0:
        grid = grid + draw_beta * np.eye(grid.shape[0])

    idx = np.unravel_index(np.argmax(grid), grid.shape)
    return int(idx[0]), int(idx[1])


def empirical_score_prior(goals_a: np.ndarray, goals_b: np.ndarray, max_goals: int = 8,
                          smooth: float = 0.5) -> np.ndarray:
    """Smoothed empirical scoreline distribution (symmetrized so it's swap-invariant)."""
    grid = np.full((max_goals + 1, max_goals + 1), smooth)
    for a, b in zip(goals_a.astype(int), goals_b.astype(int)):
        grid[min(a, max_goals), min(b, max_goals)] += 1
    grid = (grid + grid.T) / 2  # symmetry: prior must not favor team_a
    return grid / grid.sum()


# ── Cross-fitted calibration on OOF folds ─────────────────────────────────────
def crossfit_grid_params(oof_df: pd.DataFrame) -> dict[str, dict]:
    """
    For each fold, fit scale_c + rho on the OTHER folds' OOF lambdas.
    Returns {fold_name: {"scale_c": ..., "rho": ...}}.
    """
    params = {}
    folds = oof_df.groupby(["fold_competition", "fold_year"]).groups
    for key in folds:
        other = oof_df[
            ~((oof_df["fold_competition"] == key[0]) & (oof_df["fold_year"] == key[1]))
        ]
        la = other["pred_lambda_a"].values
        lb = other["pred_lambda_b"].values
        ga = other["goals_A"].values
        gb = other["goals_B"].values
        scale_c = fit_lambda_scale(la, lb, ga, gb)
        la_s, lb_s = apply_lambda_scale(la, lb, scale_c)
        rho = fit_rho(la_s, lb_s, ga.astype(int), gb.astype(int))
        params[f"{key[0]} {key[1]}"] = {"scale_c": float(scale_c), "rho": float(rho)}
    return params


def fit_grid_params(oof_df: pd.DataFrame) -> dict:
    """Fit scale_c + rho on ALL OOF folds (used for the final holdout pass)."""
    la = oof_df["pred_lambda_a"].values
    lb = oof_df["pred_lambda_b"].values
    ga = oof_df["goals_A"].values
    gb = oof_df["goals_B"].values
    scale_c = fit_lambda_scale(la, lb, ga, gb)
    la_s, lb_s = apply_lambda_scale(la, lb, scale_c)
    rho = fit_rho(la_s, lb_s, ga.astype(int), gb.astype(int))
    return {"scale_c": float(scale_c), "rho": float(rho)}


# ── Apply a rule over a lambda frame ──────────────────────────────────────────
def score_lambda_frame(
    df: pd.DataFrame,
    rule: Callable[..., tuple[int, int]],
    per_fold_params: dict[str, dict] | None = None,
    fixed_params: dict | None = None,
    use_ko: bool = False,
    **rule_kwargs,
) -> np.ndarray:
    """
    Vector-apply a decision rule. If per_fold_params given, each row uses its
    fold's cross-fitted params; else fixed_params (or none).
    """
    ko = knockout_mask(df) if use_ko else np.zeros(len(df), dtype=bool)
    preds = np.zeros((len(df), 2), dtype=int)

    for i, (_, row) in enumerate(df.iterrows()):
        kwargs = dict(rule_kwargs)
        if per_fold_params is not None:
            fold_name = f"{row['fold_competition']} {row['fold_year']}"
            kwargs.update(per_fold_params[fold_name])
        elif fixed_params is not None:
            kwargs.update(fixed_params)
        if use_ko:
            kwargs["is_ko"] = bool(ko[i])
        preds[i] = rule(row["pred_lambda_a"], row["pred_lambda_b"], **kwargs)

    return preds
