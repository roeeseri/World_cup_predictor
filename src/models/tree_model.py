from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class TreeGoalModel:
    """
    Random Forest Regressor for goal prediction.
    
    Separate models for goals_A and goals_B.
    Regularization (reduced n_estimators, max_depth constraint) prevents overfitting.
    
    Args:
        n_estimators: Number of trees. Reduced from 200 to 100 for stronger regularization.
        max_depth: Maximum tree depth. Default 15 prevents deep memorization.
        random_state: For reproducibility.
    
    Dependency mechanism:
        Similar to Poisson: goals are modeled separately, but correlation emerges
        through shared feature importance (elo_diff, form, rest, etc.).
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 15, random_state: int = 42) -> None:
        self.home_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.away_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X, y):
        """Fit separate Random Forest models for A and B goals."""
        y_arr = _coerce_goal_array(y)
        self.home_model.fit(X, y_arr[:, 0])
        self.away_model.fit(X, y_arr[:, 1])
        return self

    def predict(self, X):
        """Predict expected goals for A and B."""
        home_pred = self.home_model.predict(X)
        away_pred = self.away_model.predict(X)
        return np.column_stack([home_pred, away_pred])


def _coerce_goal_array(y) -> np.ndarray:
    """Convert various goal formats to standardized (n_samples, 2) array."""
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
