from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson


def round_expected_goals(lambda_a: float, lambda_b: float) -> tuple[int, int]:
    return int(np.rint(lambda_a)), int(np.rint(lambda_b))


def poisson_score_grid(lambda_a: float, lambda_b: float, max_goals: int = 6) -> pd.DataFrame:
    goals = np.arange(0, max_goals + 1)
    home_probs = poisson.pmf(goals, lambda_a)
    away_probs = poisson.pmf(goals, lambda_b)
    grid = np.outer(home_probs, away_probs)
    return pd.DataFrame(grid, index=goals, columns=goals)


def most_likely_score(lambda_a: float, lambda_b: float, max_goals: int = 6) -> tuple[int, int]:
    grid = poisson_score_grid(lambda_a, lambda_b, max_goals=max_goals)
    max_idx = grid.stack().idxmax()
    return int(max_idx[0]), int(max_idx[1])


def outcome_probabilities(lambda_a: float, lambda_b: float, max_goals: int = 6) -> dict:
    grid = poisson_score_grid(lambda_a, lambda_b, max_goals=max_goals).to_numpy()
    home_win = np.tril(grid, -1).sum()
    draw = np.trace(grid)
    away_win = np.triu(grid, 1).sum()
    return {"home_win": float(home_win), "draw": float(draw), "away_win": float(away_win)}
