from __future__ import annotations

import numpy as np
import lightgbm as lgb

from .base import coerce_goal_array, ensure_non_negative


class LGBMGoalModel:
    """
    LightGBM model for goal prediction.

    Separate models for goals_A and goals_B using Poisson objective,
    which naturally handles count data and non-negative outputs.

    Default params are from Optuna search on 2022 WC (notebook 06).
    For generalization, consider tuning on a held-out WC year.
    """

    def __init__(
        self,
        n_estimators: int = 119,
        max_depth: int = 9,
        learning_rate: float = 0.08693801662636208,
        num_leaves: int = 71,
        min_child_samples: int = 38,
        subsample: float =  0.6748588482689979,
        colsample_bytree: float = 0.9474488082792801,
        reg_alpha: float = 0.006176220605605113,
        reg_lambda: float = 0.00040731368039932604,
        random_state: int = 42,
        verbose: int = -1,
    ) -> None:
        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            verbose=verbose,
            objective="poisson",
            n_jobs=-1,
        )
        self.home_model = lgb.LGBMRegressor(**self._params)
        self.away_model = lgb.LGBMRegressor(**self._params)

    def fit(self, X, y, sample_weight=None):
        y_arr = coerce_goal_array(y)
        self.home_model.fit(X, y_arr[:, 0], sample_weight=sample_weight)
        self.away_model.fit(X, y_arr[:, 1], sample_weight=sample_weight)
        return self

    def predict(self, X):
        home_pred = self.home_model.predict(X)
        away_pred = self.away_model.predict(X)
        return ensure_non_negative(np.column_stack([home_pred, away_pred]))

    def feature_importances(self, feature_names: list[str] | None = None) -> dict:
        """Return mean feature importance across home/away models."""
        imp_a = self.home_model.feature_importances_
        imp_b = self.away_model.feature_importances_
        mean_imp = (imp_a + imp_b) / 2
        if feature_names is None:
            return {"feature": list(range(len(mean_imp))), "importance": mean_imp.tolist()}
        return dict(zip(feature_names, mean_imp.tolist()))
