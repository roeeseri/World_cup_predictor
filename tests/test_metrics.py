import numpy as np

from src.evaluation.metrics import (
    exact_score_accuracy,
    goal_difference_mae,
    goal_mae_raw,
    goal_rmse_raw,
    result_accuracy,
    result_label,
    rounded_score_mae,
    winner_aware_error,
)


def test_metrics_small_example():
    y_true = np.array([[1, 0], [0, 2], [2, 2]])
    y_pred_expected = np.array([[1.2, 0.4], [0.3, 1.8], [2.1, 1.9]])
    y_pred_scores = np.array([[1, 0], [0, 2], [2, 1]])

    assert goal_mae_raw(y_true, y_pred_expected) > 0
    assert goal_rmse_raw(y_true, y_pred_expected) > 0
    assert rounded_score_mae(y_true, y_pred_scores) >= 0
    assert exact_score_accuracy(y_true, y_pred_scores) == 2 / 3
    assert result_accuracy(y_true, y_pred_scores) == 2 / 3
    assert goal_difference_mae(y_true, y_pred_scores) >= 0
    assert winner_aware_error(y_true, y_pred_scores, alpha=0.5) >= 0


def test_result_label():
    assert result_label(2, 1) == "A"
    assert result_label(1, 2) == "B"
    assert result_label(1, 1) == "D"
