from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .base import coerce_goal_array, ensure_non_negative


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

    def fit(self, X, y, sample_weight=None):
        """Fit separate Random Forest models for A and B goals."""
        y_arr = coerce_goal_array(y)
        self.home_model.fit(X, y_arr[:, 0], sample_weight=sample_weight)
        self.away_model.fit(X, y_arr[:, 1], sample_weight=sample_weight)
        return self

    def predict(self, X):
        """Predict expected goals for A and B."""
        home_pred = self.home_model.predict(X)
        away_pred = self.away_model.predict(X)
        return ensure_non_negative(np.column_stack([home_pred, away_pred]))
