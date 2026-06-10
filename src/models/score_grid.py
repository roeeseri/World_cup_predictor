"""
Dixon-Coles joint score grid, lambda calibration, decision rules, and knockout ET mixture.

All functions operate purely on lambdas and goal counts — no model artifacts touched.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson


# ── Dixon-Coles correction ─────────────────────────────────────────────────────
def _dc_tau(i: int, j: int, lambda_a: float, lambda_b: float, rho: float) -> float:
    """
    Dixon-Coles τ correction factor for low-score cells.

    τ > 1 inflates (i,j), τ < 1 deflates it.
    With rho < 0: τ(0,0) > 1 and τ(1,1) > 1 → more draws.
    """
    if i == 0 and j == 0:
        return 1.0 - lambda_a * lambda_b * rho
    if i == 0 and j == 1:
        return 1.0 + lambda_a * rho
    if i == 1 and j == 0:
        return 1.0 + lambda_b * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def dixon_coles_grid(
    lambda_a: float,
    lambda_b: float,
    rho: float = 0.0,
    max_goals: int = 8,
) -> np.ndarray:
    """
    Build a (max_goals+1, max_goals+1) joint probability matrix with Dixon-Coles correction.

    P[i, j] ∝ Poisson(i; λa) * Poisson(j; λb) * τ(i, j, λa, λb, ρ)

    With ρ = 0 this reduces to independent Poisson.
    ρ < 0 inflates 0-0 and 1-1 (draws), fixing under-prediction of draws.
    """
    lambda_a = max(lambda_a, 1e-6)
    lambda_b = max(lambda_b, 1e-6)
    goals = np.arange(max_goals + 1)
    pa = poisson.pmf(goals, lambda_a)
    pb = poisson.pmf(goals, lambda_b)
    grid = np.outer(pa, pb)  # independent Poisson baseline

    # Apply τ correction to low-score cells
    for i in range(min(2, max_goals + 1)):
        for j in range(min(2, max_goals + 1)):
            grid[i, j] *= _dc_tau(i, j, lambda_a, lambda_b, rho)

    grid = np.maximum(grid, 0.0)
    total = grid.sum()
    if total > 0:
        grid /= total
    return grid


# ── W/D/L from grid ────────────────────────────────────────────────────────────
def win_draw_loss_from_grid(grid: np.ndarray) -> tuple[float, float, float]:
    """Extract (win, draw, loss) probabilities from a score grid."""
    win = float(np.tril(grid, -1).sum())
    draw = float(np.trace(grid))
    loss = float(np.triu(grid, 1).sum())
    return win, draw, loss


# ── Decision rule ──────────────────────────────────────────────────────────────
def pick_score(
    grid: np.ndarray,
    alpha: float = 0.0,
) -> tuple[int, int]:
    """
    Select the predicted scoreline from a probability grid.

    alpha = 0: pure argmax of P(exact score)  → optimal for the exact-score objective
    alpha > 0: argmax of P(i,j) + alpha * P(outcome of (i,j))
               gives secondary value to getting W/D/L right

    alpha should be tuned on folds; 0 is the right default for the scoring objective.
    """
    if alpha == 0.0:
        idx = np.unravel_index(np.argmax(grid), grid.shape)
        return int(idx[0]), int(idx[1])

    n = grid.shape[0]
    win_mass = float(np.tril(grid, -1).sum())
    draw_mass = float(np.trace(grid))
    loss_mass = float(np.triu(grid, 1).sum())

    score_with_bonus = grid.copy()
    for i in range(n):
        for j in range(n):
            if i > j:
                outcome_p = win_mass
            elif i == j:
                outcome_p = draw_mass
            else:
                outcome_p = loss_mass
            score_with_bonus[i, j] = grid[i, j] + alpha * outcome_p

    idx = np.unravel_index(np.argmax(score_with_bonus), score_with_bonus.shape)
    return int(idx[0]), int(idx[1])


# ── Lambda calibration ─────────────────────────────────────────────────────────
def fit_lambda_scale(
    lambdas_a: np.ndarray,
    lambdas_b: np.ndarray,
    goals_a: np.ndarray,
    goals_b: np.ndarray,
) -> float:
    """
    Fit a multiplicative scale c so that λ' = c * λ maximises Poisson log-likelihood.

    Closed-form solution: c = Σ(goals_a + goals_b) / Σ(λ_a + λ_b)
    """
    total_goals = float(np.sum(goals_a) + np.sum(goals_b))
    total_lambda = float(np.sum(lambdas_a) + np.sum(lambdas_b))
    if total_lambda <= 0:
        return 1.0
    return total_goals / total_lambda


def fit_lambda_affine(
    lambdas_a: np.ndarray,
    lambdas_b: np.ndarray,
    goals_a: np.ndarray,
    goals_b: np.ndarray,
) -> tuple[float, float]:
    """
    Fit log-affine calibration: log λ' = a + b * log λ  (i.e. λ' = e^a * λ^b).

    b < 1 compresses the spread — helps when ELO dominates and λa≈λb in knockouts.
    Returns (a, b) where b is the exponent.
    """
    la = np.asarray(lambdas_a, dtype=float)
    lb = np.asarray(lambdas_b, dtype=float)
    ga = np.asarray(goals_a, dtype=float)
    gb = np.asarray(goals_b, dtype=float)

    la_safe = np.maximum(la, 1e-6)
    lb_safe = np.maximum(lb, 1e-6)
    log_la = np.log(la_safe)
    log_lb = np.log(lb_safe)

    def neg_log_lik(params: np.ndarray) -> float:
        a, b = params
        lp_a = np.clip(a + b * log_la, -10, 3)
        lp_b = np.clip(a + b * log_lb, -10, 3)
        lambda_a_cal = np.exp(lp_a)
        lambda_b_cal = np.exp(lp_b)
        ll = (
            ga * lp_a - lambda_a_cal
            + gb * lp_b - lambda_b_cal
        )
        return -ll.sum()

    # Initial guess: a=0 (scale=1), b=1 (no compression)
    result = minimize(neg_log_lik, x0=[0.0, 1.0], method="Nelder-Mead",
                      options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
    a, b = result.x
    return float(a), float(b)


def apply_lambda_scale(lambdas_a: np.ndarray, lambdas_b: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply multiplicative scale calibration."""
    return np.maximum(lambdas_a * c, 1e-6), np.maximum(lambdas_b * c, 1e-6)


def apply_lambda_affine(
    lambdas_a: np.ndarray,
    lambdas_b: np.ndarray,
    a: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply log-affine calibration: λ' = e^a * λ^b."""
    la = np.maximum(lambdas_a, 1e-6)
    lb = np.maximum(lambdas_b, 1e-6)
    cal_a = np.maximum(np.exp(a) * la ** b, 1e-6)
    cal_b = np.maximum(np.exp(a) * lb ** b, 1e-6)
    return cal_a, cal_b


# ── Dixon-Coles rho fitting ────────────────────────────────────────────────────
def fit_rho(
    lambdas_a: np.ndarray,
    lambdas_b: np.ndarray,
    goals_a: np.ndarray,
    goals_b: np.ndarray,
    rho_bounds: tuple[float, float] = (-0.9, 0.3),
) -> float:
    """
    Fit the Dixon-Coles correlation parameter ρ by MLE on OOF lambdas.

    Only the τ terms contribute to the ρ gradient (independent Poisson terms are fixed).
    Negative ρ inflates 0-0 and 1-1 → more predicted draws.
    """
    la = np.asarray(lambdas_a, dtype=float)
    lb = np.asarray(lambdas_b, dtype=float)
    ga = np.asarray(goals_a, dtype=int)
    gb = np.asarray(goals_b, dtype=int)

    # Pre-compute τ only for low-scoring rows (others contribute log(1)=0)
    low_mask = (ga <= 1) & (gb <= 1)
    la_low = la[low_mask]
    lb_low = lb[low_mask]
    ga_low = ga[low_mask]
    gb_low = gb[low_mask]

    def neg_log_lik(rho: float) -> float:
        tau_vals = np.array([
            _dc_tau(int(a), int(b), float(la_), float(lb_), rho)
            for a, b, la_, lb_ in zip(ga_low, gb_low, la_low, lb_low)
        ])
        # If any tau <= 0, rho is invalid
        if np.any(tau_vals <= 0):
            return 1e10
        return -np.sum(np.log(tau_vals))

    result = minimize_scalar(neg_log_lik, bounds=rho_bounds, method="bounded")
    return float(result.x)


# ── Knockout ET mixture ────────────────────────────────────────────────────────
def knockout_grid(
    lambda_a: float,
    lambda_b: float,
    rho: float = 0.0,
    et_scale: float = 30.0 / 90.0,
    max_goals: int = 8,
) -> np.ndarray:
    """
    Build a knockout-aware score grid with conditional extra-time mixture.

    Targets are 120-min scores (dataset already records KO scores after 120 min,
    penalties excluded). The ET extension is applied *only* to the draw mass at
    90 minutes — non-draw outcomes are unchanged.

    G_final[i,j] = G90[i,j]  for i≠j
    G_final[k,k] → redistributed via G_ET centered on (k,k) for each draw cell k

    G_ET uses λ_ET = λ_90 * et_scale (default 30/90 ≈ 0.333 for 30 ET minutes).

    Mass is conserved: the draw cells at 90' get spread over ET scorelines, with
    any residual draw mass in ET representing a genuine pens-bound game.
    """
    lambda_a = max(lambda_a, 1e-6)
    lambda_b = max(lambda_b, 1e-6)

    # 90-minute grid
    g90 = dixon_coles_grid(lambda_a, lambda_b, rho, max_goals)

    # Extra-time grid (scaled lambdas)
    la_et = max(lambda_a * et_scale, 1e-6)
    lb_et = max(lambda_b * et_scale, 1e-6)
    g_et = dixon_coles_grid(la_et, lb_et, rho=0.0, max_goals=max_goals)

    # Start from non-draw cells unchanged
    g_final = g90.copy()
    g_final -= np.diag(np.diag(g90))  # zero out diagonal (draws at 90')

    # Redistribute each draw cell k,k using ET grid shifted by k goals per team
    for k in range(max_goals + 1):
        draw_mass = g90[k, k]
        if draw_mass < 1e-12:
            continue
        # Shift g_et so that k+i, k+j indexing lines up; clip to max_goals
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                new_i = min(k + i, max_goals)
                new_j = min(k + j, max_goals)
                g_final[new_i, new_j] += draw_mass * g_et[i, j]

    g_final = np.maximum(g_final, 0.0)
    total = g_final.sum()
    if total > 0:
        g_final /= total
    return g_final


# ── Calibration fitting on OOF lambdas ────────────────────────────────────────
def fit_calibration_params(
    oof_lambda_a: np.ndarray,
    oof_lambda_b: np.ndarray,
    oof_goals_a: np.ndarray,
    oof_goals_b: np.ndarray,
    fit_affine: bool = True,
) -> dict:
    """
    Fit all calibration parameters on OOF lambdas and return a config dict.

    Returns:
        {
          "scale_c":  float,
          "affine_a": float,  (if fit_affine)
          "affine_b": float,  (if fit_affine)
          "rho":      float,
        }
    """
    la = np.asarray(oof_lambda_a, dtype=float)
    lb = np.asarray(oof_lambda_b, dtype=float)
    ga = np.asarray(oof_goals_a, dtype=float)
    gb = np.asarray(oof_goals_b, dtype=float)

    params: dict = {}

    params["scale_c"] = fit_lambda_scale(la, lb, ga, gb)

    if fit_affine:
        a, b = fit_lambda_affine(la, lb, ga, gb)
        params["affine_a"] = a
        params["affine_b"] = b

    # Fit rho on scale-calibrated lambdas for cleaner signal
    la_scaled, lb_scaled = apply_lambda_scale(la, lb, params["scale_c"])
    params["rho"] = fit_rho(la_scaled, lb_scaled, ga.astype(int), gb.astype(int))

    return params


def make_score_fn(
    rho: float = 0.0,
    scale_c: float = 1.0,
    affine_a: float | None = None,
    affine_b: float | None = None,
    alpha: float = 0.0,
    knockout: bool = False,
    et_scale: float = 30.0 / 90.0,
) -> tuple[callable, callable]:
    """
    Build (score_fn, probs_fn) callables from calibration parameters.

    Returns:
        score_fn: (lambda_a, lambda_b) -> (int, int) predicted score
        probs_fn: (lambda_a, lambda_b) -> (win, draw, loss)
    """
    def _calibrate(la: float, lb: float) -> tuple[float, float]:
        if affine_a is not None and affine_b is not None:
            la_cal, lb_cal = apply_lambda_affine(
                np.array([la]), np.array([lb]), affine_a, affine_b
            )
            return float(la_cal[0]), float(lb_cal[0])
        else:
            la_cal, lb_cal = apply_lambda_scale(np.array([la]), np.array([lb]), scale_c)
            return float(la_cal[0]), float(lb_cal[0])

    def score_fn(lambda_a: float, lambda_b: float) -> tuple[int, int]:
        la, lb = _calibrate(lambda_a, lambda_b)
        if knockout:
            grid = knockout_grid(la, lb, rho=rho, et_scale=et_scale)
        else:
            grid = dixon_coles_grid(la, lb, rho=rho)
        return pick_score(grid, alpha=alpha)

    def probs_fn(lambda_a: float, lambda_b: float) -> tuple[float, float, float]:
        la, lb = _calibrate(lambda_a, lambda_b)
        if knockout:
            grid = knockout_grid(la, lb, rho=rho, et_scale=et_scale)
        else:
            grid = dixon_coles_grid(la, lb, rho=rho)
        return win_draw_loss_from_grid(grid)

    return score_fn, probs_fn
