"""XGBoost model for goal prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .base import coerce_goal_array, ensure_non_negative


class XGBGoalModel:
    """
    XGBoost model for goal prediction.

    Separate models for goals_A and goals_B using Poisson objective,
    which naturally handles count data and non-negative outputs.

    Default params from notebook 06 Optuna search (starting point only —
    retune with proper WC cross-validation before using in production).
    """

    def __init__(
        self,
        n_estimators: int = 329,
        max_depth: int = 9,
        learning_rate: float = 0.011940116953483713,
        subsample: float = 0.9611784849268618,
        colsample_bytree: float = 0.9059272632696443,
        gamma: float = 2.306771313314438,
        min_child_weight: int = 7,
        reg_alpha: float = 0.372730975908547,
        reg_lambda: float = 0.46923840424489816,
        random_state: int = 42,
    ) -> None:
        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            objective="count:poisson",
            tree_method="hist",
            verbosity=0,
            n_jobs=-1,
        )
        self.home_model = xgb.XGBRegressor(**self._params)
        self.away_model = xgb.XGBRegressor(**self._params)

    def fit(self, X, y, sample_weight=None):
        y_arr = coerce_goal_array(y)
        self.home_model.fit(X, y_arr[:, 0], sample_weight=sample_weight)
        self.away_model.fit(X, y_arr[:, 1], sample_weight=sample_weight)
        return self

    def predict(self, X) -> np.ndarray:
        home_pred = self.home_model.predict(X)
        away_pred = self.away_model.predict(X)
        return ensure_non_negative(np.column_stack([home_pred, away_pred]))

    def feature_importances(self, feature_names: list[str] | None = None) -> pd.DataFrame:
        imp_a = self.home_model.feature_importances_
        imp_b = self.away_model.feature_importances_
        mean_imp = (imp_a + imp_b) / 2
        names = feature_names if feature_names is not None else list(range(len(mean_imp)))
        return pd.DataFrame({"feature": names, "importance": mean_imp}).sort_values(
            "importance", ascending=False
        )
