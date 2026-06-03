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
        n_estimators: int = 246,
        max_depth: int = 12,
        learning_rate: float = 0.02235344330741584,
        num_leaves: int = 153,
        min_child_samples: int = 53,
        subsample: float =  0.7131805151434094,
        colsample_bytree: float = 0.9003288705141621,
        reg_alpha: float = 0.0005891440604933838,
        reg_lambda: float = 0.0010391363783449235,
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
