from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import (
    exact_score_accuracy,
    goal_difference_mae,
    goal_mae_raw,
    goal_rmse_raw,
    result_accuracy,
    rounded_score_mae,
    winner_aware_error,
)


def summarize_evaluation(predictions_df: pd.DataFrame) -> pd.DataFrame:
    y_true = _extract_goals(predictions_df, kind="true")
    y_pred = _extract_goals(predictions_df, kind="pred")
    y_pred_rounded = np.rint(y_pred).astype(int)

    summary = {
        "goal_mae_raw": goal_mae_raw(y_true, y_pred),
        "goal_rmse_raw": goal_rmse_raw(y_true, y_pred),
        "rounded_score_mae": rounded_score_mae(y_true, y_pred_rounded),
        "exact_score_accuracy": exact_score_accuracy(y_true, y_pred_rounded),
        "result_accuracy": result_accuracy(y_true, y_pred_rounded),
        "goal_difference_mae": goal_difference_mae(y_true, y_pred_rounded),
        "winner_aware_error": winner_aware_error(y_true, y_pred_rounded),
    }

    return pd.DataFrame([summary])


def _extract_goals(predictions_df: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "true":
        candidates = [
            ("home_goals", "away_goals"),
            ("home_score", "away_score"),
            ("home", "away"),
        ]
    else:
        candidates = [
            ("pred_home_goals", "pred_away_goals"),
            ("pred_home", "pred_away"),
            ("expected_home_goals", "expected_away_goals"),
        ]

    for home_col, away_col in candidates:
        if {home_col, away_col}.issubset(predictions_df.columns):
            return predictions_df[[home_col, away_col]].to_numpy()

    return predictions_df.iloc[:, :2].to_numpy()
