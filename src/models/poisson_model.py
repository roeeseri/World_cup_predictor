from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from .base import coerce_goal_array, ensure_non_negative


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

    def fit(self, X, y, sample_weight=None):
        """Fit separate Poisson models for A and B goals."""
        y_arr = coerce_goal_array(y)
        self.home_model.fit(X, y_arr[:, 0], sample_weight=sample_weight)
        self.away_model.fit(X, y_arr[:, 1], sample_weight=sample_weight)
        return self

    def predict(self, X):
        """Predict expected goals for A and B."""
        home_pred = self.home_model.predict(X)
        away_pred = self.away_model.predict(X)
        return ensure_non_negative(np.column_stack([home_pred, away_pred]))
