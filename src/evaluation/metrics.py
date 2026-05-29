from __future__ import annotations

import numpy as np
import pandas as pd


def detect_goal_anomalies(y_true) -> np.ndarray:
    """
    Detect anomalous matches (extreme scorelines like 9-0).
    
    Anomalies are matches where the goal difference is unusually large.
    These rare blowouts can skew evaluation metrics if given equal weight.
    
    Threshold: Goal difference > 4 (e.g., 5-0, 6-1, etc.)
    These occur in ~1-2% of matches but affect metrics heavily if errors are large.
    
    Returns:
        Boolean array where True = anomaly (extreme scoreline)
    """
    y_true_arr = _coerce_goal_array(y_true)
    goal_diff = np.abs(y_true_arr[:, 0] - y_true_arr[:, 1])
    return goal_diff > 4


def weight_anomalies(errors, anomaly_mask, anomaly_weight: float = 0.3):
    """
    Reduce the impact of anomalous matches on metrics.
    
    Approach: Downweight errors for anomalies before averaging.
    - Normal matches: weight = 1.0
    - Anomalies: weight = anomaly_weight (default 0.3)
    
    Example:
        errors = [0.5, 0.3, 2.1, 0.4]  # 2.1 is anomaly error
        weights = [1.0, 1.0, 0.3, 1.0]
        weighted_mean = sum(errors * weights) / sum(weights)
    
    Args:
        errors: Array of error magnitudes
        anomaly_mask: Boolean array indicating anomalies
        anomaly_weight: Weight to apply to anomalies (0.0-1.0)
    
    Returns:
        Weighted mean error
    """
    weights = np.where(anomaly_mask, anomaly_weight, 1.0)
    return float(np.average(errors, weights=weights))


def goal_mae_raw(y_true, y_pred) -> float:
    """
    Mean Absolute Error for raw expected goals (before rounding).
    
    Anomaly-aware: Large misses on rare scorelines (9-0) don't dominate the metric.
    
    Returns the weighted MAE where anomalous matches count less.
    """
    y_true_arr = _coerce_goal_array(y_true)
    y_pred_arr = _coerce_goal_array(y_pred)
    errors = np.abs(y_true_arr - y_pred_arr).mean(axis=1)
    anomalies = detect_goal_anomalies(y_true_arr)
    return weight_anomalies(errors, anomalies, anomaly_weight=0.3)


def goal_rmse_raw(y_true, y_pred) -> float:
    """
    Root Mean Squared Error for raw expected goals.
    
    Anomaly-aware: Reduces impact of huge outliers.
    """
    y_true_arr = _coerce_goal_array(y_true)
    y_pred_arr = _coerce_goal_array(y_pred)
    sq_errors = ((y_true_arr - y_pred_arr) ** 2).mean(axis=1)
    anomalies = detect_goal_anomalies(y_true_arr)
    weighted_mse = weight_anomalies(sq_errors, anomalies, anomaly_weight=0.3)
    return float(np.sqrt(weighted_mse))


def rounded_score_mae(y_true, y_pred_rounded) -> float:
    """
    Mean Absolute Error for rounded scores (post-rounding accuracy).
    
    Anomaly-aware: Reduces penalty for wrong predictions on extreme blowouts.
    """
    y_true_arr = _coerce_goal_array(y_true)
    y_pred_arr = _coerce_goal_array(y_pred_rounded)
    errors = np.abs(y_true_arr - y_pred_arr).mean(axis=1)
    anomalies = detect_goal_anomalies(y_true_arr)
    return weight_anomalies(errors, anomalies, anomaly_weight=0.3)


def exact_score_accuracy(y_true, y_pred_rounded) -> float:
    """
    Exact score accuracy: % of matches where predicted score matches actual.
    
    Anomaly-aware: Doesn't penalize missing an exact 8-0 forecast.
    """
    y_true_arr = _coerce_goal_array(y_true)
    y_pred_arr = _coerce_goal_array(y_pred_rounded)
    exact_matches = np.all(y_true_arr == y_pred_arr, axis=1).astype(float)
    anomalies = detect_goal_anomalies(y_true_arr)
    return weight_anomalies(exact_matches, anomalies, anomaly_weight=0.1)


def result_accuracy(y_true, y_pred_rounded) -> float:
    """
    Result accuracy: % of matches where win/draw/loss prediction is correct.
    
    Anomaly-aware: Same weight for correct/incorrect result, but anomalies count less overall.
    """
    y_true_arr = _coerce_goal_array(y_true)
    y_pred_arr = _coerce_goal_array(y_pred_rounded)
    true_outcome = np.sign(y_true_arr[:, 0] - y_true_arr[:, 1])
    pred_outcome = np.sign(y_pred_arr[:, 0] - y_pred_arr[:, 1])
    correct = (true_outcome == pred_outcome).astype(float)
    anomalies = detect_goal_anomalies(y_true_arr)
    return weight_anomalies(correct, anomalies, anomaly_weight=0.3)


def goal_difference_mae(y_true, y_pred_rounded) -> float:
    """
    Mean Absolute Error of goal difference (useful for tournament tiebreaker).
    
    Anomaly-aware: Reduces penalty for extreme blowout forecasts.
    """
    y_true_arr = _coerce_goal_array(y_true)
    y_pred_arr = _coerce_goal_array(y_pred_rounded)
    true_diff = y_true_arr[:, 0] - y_true_arr[:, 1]
    pred_diff = y_pred_arr[:, 0] - y_pred_arr[:, 1]
    errors = np.abs(true_diff - pred_diff)
    anomalies = detect_goal_anomalies(y_true_arr)
    return weight_anomalies(errors, anomalies, anomaly_weight=0.3)


def winner_aware_error(y_true, y_pred_rounded, alpha: float = 0.5) -> float:
    """
    Weighted error metric: penalizes wrong winner prediction more than wrong scoreline.
    
    Formula: error = |goal_diff| + alpha * (1 if winner_mismatch else 0)
    
    Anomaly-aware: Reduces penalty on anomalies.
    """
    y_true_arr = _coerce_goal_array(y_true)
    y_pred_arr = _coerce_goal_array(y_pred_rounded)
    true_diff = y_true_arr[:, 0] - y_true_arr[:, 1]
    pred_diff = y_pred_arr[:, 0] - y_pred_arr[:, 1]
    base_error = np.abs(true_diff - pred_diff)
    mismatch = (np.sign(true_diff) != np.sign(pred_diff)).astype(float)
    total_error = base_error + alpha * mismatch
    anomalies = detect_goal_anomalies(y_true_arr)
    return weight_anomalies(total_error, anomalies, anomaly_weight=0.3)


def result_label(goals_a: int, goals_b: int) -> str:
    """
    Convert goals to result label (A = home win, B = away win, D = draw).
    """
    if goals_a > goals_b:
        return "A"
    if goals_a < goals_b:
        return "B"
    return "D"


def _coerce_goal_array(y) -> np.ndarray:
    """
    Convert various goal formats to standardized (n_samples, 2) array.
    
    Handles:
    - DataFrame with columns: goals_A, goals_B or home_goals, away_goals
    - Numpy array (n, 2)
    - Other iterable formats
    """
    if isinstance(y, pd.DataFrame):
        if {"goals_A", "goals_B"}.issubset(y.columns):
            return y[["goals_A", "goals_B"]].to_numpy()
        if {"home_goals", "away_goals"}.issubset(y.columns):
            return y[["home_goals", "away_goals"]].to_numpy()
        return y.iloc[:, :2].to_numpy()
    y_arr = np.asarray(y)
    if y_arr.ndim != 2 or y_arr.shape[1] < 2:
        raise ValueError("Expected y with shape (n_samples, 2) for A/B goals.")
    return y_arr[:, :2]
