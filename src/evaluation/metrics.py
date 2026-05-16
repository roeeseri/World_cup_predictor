from __future__ import annotations

import numpy as np
import pandas as pd


def goal_mae_raw(y_true, y_pred) -> float:
    y_true_arr = _coerce_goal_array(y_true)
    y_pred_arr = _coerce_goal_array(y_pred)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def goal_rmse_raw(y_true, y_pred) -> float:
    y_true_arr = _coerce_goal_array(y_true)
    y_pred_arr = _coerce_goal_array(y_pred)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def rounded_score_mae(y_true, y_pred_rounded) -> float:
    y_true_arr = _coerce_goal_array(y_true)
    y_pred_arr = _coerce_goal_array(y_pred_rounded)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def exact_score_accuracy(y_true, y_pred_rounded) -> float:
    y_true_arr = _coerce_goal_array(y_true)
    y_pred_arr = _coerce_goal_array(y_pred_rounded)
    return float(np.mean(np.all(y_true_arr == y_pred_arr, axis=1)))


def result_accuracy(y_true, y_pred_rounded) -> float:
    y_true_arr = _coerce_goal_array(y_true)
    y_pred_arr = _coerce_goal_array(y_pred_rounded)
    true_outcome = np.sign(y_true_arr[:, 0] - y_true_arr[:, 1])
    pred_outcome = np.sign(y_pred_arr[:, 0] - y_pred_arr[:, 1])
    return float(np.mean(true_outcome == pred_outcome))


def goal_difference_mae(y_true, y_pred_rounded) -> float:
    y_true_arr = _coerce_goal_array(y_true)
    y_pred_arr = _coerce_goal_array(y_pred_rounded)
    true_diff = y_true_arr[:, 0] - y_true_arr[:, 1]
    pred_diff = y_pred_arr[:, 0] - y_pred_arr[:, 1]
    return float(np.mean(np.abs(true_diff - pred_diff)))


def winner_aware_error(y_true, y_pred_rounded, alpha: float = 0.5) -> float:
    y_true_arr = _coerce_goal_array(y_true)
    y_pred_arr = _coerce_goal_array(y_pred_rounded)
    true_diff = y_true_arr[:, 0] - y_true_arr[:, 1]
    pred_diff = y_pred_arr[:, 0] - y_pred_arr[:, 1]
    base_error = np.abs(true_diff - pred_diff)
    mismatch = np.sign(true_diff) != np.sign(pred_diff)
    return float(np.mean(base_error + alpha * mismatch))


def _coerce_goal_array(y) -> np.ndarray:
    if isinstance(y, pd.DataFrame):
        if {"home_goals", "away_goals"}.issubset(y.columns):
            return y[["home_goals", "away_goals"]].to_numpy()
        return y.iloc[:, :2].to_numpy()
    y_arr = np.asarray(y)
    if y_arr.ndim != 2 or y_arr.shape[1] < 2:
        raise ValueError("Expected y with shape (n_samples, 2) for home/away goals.")
    return y_arr[:, :2]
