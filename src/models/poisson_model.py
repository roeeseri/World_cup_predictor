from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor


class PoissonGoalModel:
    def __init__(self, alpha: float = 1.0, max_iter: int = 1000) -> None:
        self.home_model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
        self.away_model = PoissonRegressor(alpha=alpha, max_iter=max_iter)

    def fit(self, X, y):
        y_arr = _coerce_goal_array(y)
        self.home_model.fit(X, y_arr[:, 0])
        self.away_model.fit(X, y_arr[:, 1])
        return self

    def predict(self, X):
        home_pred = self.home_model.predict(X)
        away_pred = self.away_model.predict(X)
        return np.column_stack([home_pred, away_pred])


def _coerce_goal_array(y) -> np.ndarray:
    if isinstance(y, pd.DataFrame):
        if {"home_goals", "away_goals"}.issubset(y.columns):
            return y[["home_goals", "away_goals"]].to_numpy()
        return y.iloc[:, :2].to_numpy()
    y_arr = np.asarray(y)
    if y_arr.ndim != 2 or y_arr.shape[1] < 2:
        raise ValueError("Expected y with shape (n_samples, 2) for home/away goals.")
    return y_arr[:, :2]
