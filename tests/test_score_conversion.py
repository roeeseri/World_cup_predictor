"""Contracts for the current score-conversion API and offline batch adapter."""
import numpy as np
import pytest
from src.models.score_conversion import (
    convert_expected_goals_to_scores, most_likely_score,
    poisson_score_grid, win_draw_loss_probs,
)

def test_grid_and_outcome_orientation():
    grid = poisson_score_grid(2.4, 0.6, max_goals=12)
    assert grid.shape == (13, 13)
    assert np.all(grid >= 0)
    win, draw, loss = win_draw_loss_probs(2.4, 0.6, max_goals=12)
    assert win > loss
    assert win + draw + loss == pytest.approx(grid.sum())
    assert grid.sum() == pytest.approx(1, abs=1e-5)

def test_production_threshold_is_distinct_from_poisson_mode():
    assert most_likely_score(0.95, 0.4) == (1, 0)
    assert most_likely_score(0.95, 0.4, threshold=1.0) == (0, 0)
    assert convert_expected_goals_to_scores([[0.95, 0.4]]).tolist() == [[0, 0]]

def test_batch_rounding_and_empty_input():
    assert convert_expected_goals_to_scores([[1.2, 2.7]], method="round").tolist() == [[1, 3]]
    assert convert_expected_goals_to_scores(np.empty((0, 2))).shape == (0, 2)

@pytest.mark.parametrize("values", [[[np.nan, 1]], [[np.inf, 1]], [[-1, 1]], [1, 2], [[1, 2, 3]]])
def test_invalid_goals_are_rejected(values):
    with pytest.raises(ValueError):
        convert_expected_goals_to_scores(values)

def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        convert_expected_goals_to_scores([[1, 1]], method="typo")
