from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..prediction.score_conversion import convert_expected_goals_to_scores
from .metrics import (
    detect_goal_anomalies,
    exact_score_accuracy,
    goal_difference_mae,
    goal_mae_raw,
    goal_rmse_raw,
    result_accuracy,
    rounded_score_mae,
    winner_aware_error,
)


def validate_model_dataset(
    feature_df: pd.DataFrame,
    feature_columns: Iterable[str],
    target_columns: Iterable[str],
) -> None:
    """
    Validate that dataset has required columns and no missing/invalid values.
    
    Args:
        feature_df: DataFrame with features and targets
        feature_columns: Required feature column names
        target_columns: Required target column names (goals_A, goals_B)
    
    Raises:
        ValueError: If columns missing, NaN values present, or goals negative
    """
    missing_features = [col for col in feature_columns if col not in feature_df.columns]
    missing_targets = [col for col in target_columns if col not in feature_df.columns]

    if missing_features or missing_targets:
        raise ValueError(f"Missing columns. Features: {missing_features}. Targets: {missing_targets}.")

    features = feature_df[list(feature_columns)]
    targets = feature_df[list(target_columns)]

    if features.isna().any().any():
        raise ValueError("Feature columns contain missing values.")
    if targets.isna().any().any():
        raise ValueError("Target columns contain missing values.")
    if (targets < 0).any().any():
        raise ValueError("Target columns contain negative goals.")


def evaluate_predictions(
    y_true,
    y_pred_expected,
    y_pred_scores,
    alpha: float = 0.5,
) -> dict:
    """
    Evaluate model predictions across 7 metrics.
    
    All metrics are anomaly-aware: extreme scorelines (goal_diff > 4) have reduced weight
    in error calculations. This prevents rare blowouts like 9-0 from dominating evaluation.
    
    Args:
        y_true: Actual goals (n, 2) array
        y_pred_expected: Expected goals before rounding (n, 2) array
        y_pred_scores: Rounded predicted scores (n, 2) array
        alpha: Weight for winner mismatch in winner_aware_error
    
    Returns:
        Dictionary with 7 metric values and anomaly count
    """
    y_true_arr = np.asarray(y_true)
    if y_true_arr.ndim == 1:
        y_true_arr = y_true_arr.reshape(1, -1)
    
    anomalies = detect_goal_anomalies(y_true_arr)
    n_anomalies = int(np.sum(anomalies))
    
    return {
        "goal_mae_raw": goal_mae_raw(y_true, y_pred_expected),
        "goal_rmse_raw": goal_rmse_raw(y_true, y_pred_expected),
        "rounded_score_mae": rounded_score_mae(y_true, y_pred_scores),
        "exact_score_accuracy": exact_score_accuracy(y_true, y_pred_scores),
        "result_accuracy": result_accuracy(y_true, y_pred_scores),
        "goal_difference_mae": goal_difference_mae(y_true, y_pred_scores),
        "winner_aware_error": winner_aware_error(y_true, y_pred_scores, alpha=alpha),
        "n_anomalies": n_anomalies,
    }


def compare_models(models: dict, X_test, y_test, score_method: str = "poisson") -> pd.DataFrame:
    """
    Train and evaluate multiple models on test data.
    
    Args:
        models: Dictionary of {name: model_instance}
        X_test: Test feature matrix
        y_test: Test target (goals_A, goals_B)
        score_method: "round" or "poisson" for score conversion
    
    Returns:
        DataFrame with model results sorted by MAE (lower is better)
    """
    rows = []
    for name, model in models.items():
        y_pred_expected = model.predict(X_test)
        y_pred_scores = convert_expected_goals_to_scores(y_pred_expected, method=score_method)
        metrics = evaluate_predictions(y_test, y_pred_expected, y_pred_scores)
        metrics["model"] = name
        rows.append(metrics)

    results = pd.DataFrame(rows)
    if not results.empty:
        metric_columns = [col for col in results.columns if col != "model"]
        results = results[["model", *metric_columns]]
        results = results.sort_values("goal_mae_raw")
    return results
