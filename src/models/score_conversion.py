"""Poisson-based score probability conversion utilities."""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson


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


def most_likely_score(lambda_a: float, lambda_b: float, max_goals: int = 8) -> tuple[int, int]:
    """Return the single most probable integer scoreline."""
    grid = poisson_score_grid(lambda_a, lambda_b, max_goals)
    idx = np.unravel_index(np.argmax(grid), grid.shape)
    return int(idx[0]), int(idx[1])
