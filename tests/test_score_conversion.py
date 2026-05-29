import numpy as np

from src.prediction.score_conversion import (
    convert_expected_goals_to_scores,
    most_likely_score,
    outcome_probabilities,
    poisson_score_grid,
    round_expected_goals,
)


def test_round_expected_goals():
    assert round_expected_goals(1.2, 2.7) == (1, 3)


def test_poisson_grid_properties():
    grid = poisson_score_grid(1.4, 1.1, max_goals=6)
    assert grid.shape == (7, 7)
    assert grid.to_numpy().min() >= 0
    assert grid.to_numpy().sum() <= 1.0


def test_most_likely_score_and_outcomes():
    score = most_likely_score(1.2, 1.0, max_goals=6)
    assert isinstance(score[0], int)
    assert isinstance(score[1], int)

    probs = outcome_probabilities(1.2, 1.0, max_goals=6)
    total = probs["home_win"] + probs["draw"] + probs["away_win"]
    assert 0.9 <= total <= 1.0


def test_convert_expected_goals_to_scores():
    preds = np.array([[1.2, 0.8], [2.1, 2.0]])
    rounded = convert_expected_goals_to_scores(preds, method="round")
    poisson_scores = convert_expected_goals_to_scores(preds, method="poisson", max_goals=6)
    assert rounded.shape == (2, 2)
    assert poisson_scores.shape == (2, 2)
    assert (rounded >= 0).all()
    assert (poisson_scores >= 0).all()
