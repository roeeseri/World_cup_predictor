"""Poisson-based score probability conversion utilities."""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

# Production score threshold: switch 0→1 at λ=0.9, 1→2 at λ=1.9, etc.
# Set to 1.0 to revert to the pure Poisson mode (grid argmax).
SCORE_THRESHOLD: float = 0.9


def poisson_score_grid(lambda_a: float, lambda_b: float, max_goals: int = 8) -> np.ndarray:
    """
    Build a (max_goals+1, max_goals+1) probability matrix for all scorelines.

    Entry [i, j] = P(team_a scores i goals) * P(team_b scores j goals),
    assuming independent Poisson processes.
    """
    goals = np.arange(max_goals + 1)
    prob_a = poisson.pmf(goals, max(lambda_a, 1e-6))
    prob_b = poisson.pmf(goals, max(lambda_b, 1e-6))
    return np.outer(prob_a, prob_b)


def win_draw_loss_probs(lambda_a: float, lambda_b: float, max_goals: int = 8) -> tuple[float, float, float]:
    """
    Return (win_prob, draw_prob, loss_prob) for team_a given expected goals.
    """
    grid = poisson_score_grid(lambda_a, lambda_b, max_goals)
    win = float(np.tril(grid, -1).sum())   # i > j (team_a scores more)
    draw = float(np.trace(grid))           # i == j
    loss = float(np.triu(grid, 1).sum())   # i < j
    return win, draw, loss


def top_scores(lambda_a: float, lambda_b: float, n: int = 5, max_goals: int = 8) -> list[tuple[int, int, float]]:
    """
    Return the n most likely (score_a, score_b, probability) tuples, sorted by probability.
    """
    grid = poisson_score_grid(lambda_a, lambda_b, max_goals)
    indices = np.dstack(np.unravel_index(np.argsort(grid, axis=None)[::-1], grid.shape))[0]
    return [(int(i), int(j), float(grid[i, j])) for i, j in indices[:n]]


def most_likely_score_v5(
    lambda_a: float,
    lambda_b: float,
    threshold_b: float = 0.5,
) -> tuple[int, int]:
    """V5 conditional floor rule: raise team B threshold only when team A scores 2+.

    When goals_a >= 2, uses floor(lambda_b + threshold_b) instead of the V4 0.1 bias.
    This targets 2-0→2-1 conversions without touching 1-0 predictions (no outcome% loss).
    """
    goals_a = int(lambda_a + 0.1)
    goals_b = int(lambda_b + threshold_b) if goals_a >= 2 else int(lambda_b + 0.1)
    return goals_a, goals_b


def most_likely_score(
    lambda_a: float,
    lambda_b: float,
    max_goals: int = 8,
    threshold: float = SCORE_THRESHOLD,
) -> tuple[int, int]:
    """Return the predicted integer scoreline.

    When threshold < 1.0 uses floor(λ + (1 - threshold)), so e.g. threshold=0.9
    switches 0→1 at λ=0.9, 1→2 at λ=1.9, etc.
    When threshold == 1.0 falls back to the Poisson grid argmax (true mode).
    Pass threshold=1.0 explicitly to get the pure Poisson mode behaviour.
    """
    if threshold < 1.0:
        shift = 1.0 - threshold
        return int(lambda_a + shift), int(lambda_b + shift)
    grid = poisson_score_grid(lambda_a, lambda_b, max_goals)
    idx = np.unravel_index(np.argmax(grid), grid.shape)
    return int(idx[0]), int(idx[1])
