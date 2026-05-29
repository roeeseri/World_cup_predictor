from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor


class PoissonGoalModel:
    """
    Poisson regression model for goal prediction.
    
    Separate models for goals_A and goals_B.
    Regularization (alpha > 0) prevents overfitting to noise and global mean collapse.
    
    Formula:
        goals_A ~ Poisson(lambda_A), where lambda_A = exp(X @ beta_A)
        goals_B ~ Poisson(lambda_B), where lambda_B = exp(X @ beta_B)
    
    Args:
        alpha: L2 regularization strength. Higher values prevent overfitting.
            Default 10.0 strongly regularizes toward global mean.
        max_iter: Max iterations for solver. Default 1000 usually sufficient.
    
    Dependency mechanism:
        Goals are modeled separately, but correlation is naturally encoded through
        shared features (elo_diff, form, etc). When elo_diff is high, the models
        are trained to predict higher lambda_A and lower lambda_B.
    """
    def __init__(self, alpha: float = 10.0, max_iter: int = 1000) -> None:
        self.home_model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
        self.away_model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
        self.alpha = alpha

    def fit(self, X, y):
        """Fit separate Poisson models for A and B goals."""
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
