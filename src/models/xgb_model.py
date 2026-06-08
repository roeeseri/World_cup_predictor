"""XGBoost model for goal prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .base import coerce_goal_array, ensure_non_negative
from src.features.feature_columns import mirror_features


class XGBGoalModel:
    """
    XGBoost model for goal prediction with symmetric training.

    A single model is trained on augmented data: each match appears twice —
    once as (team_a, features) → goals_A, and once as (team_b, mirrored_features)
    → goals_B. Prediction is symmetric: goals_A = model(X), goals_B = model(mirror(X)).

    This ensures predict(A vs B) gives the same lambda values as predict(B vs A)
    with teams swapped, regardless of which team is listed first.

    Default params from Optuna 3-fold WC CV (notebook 07).
    """

    def __init__(
        self,
        n_estimators: int = 268,
        max_depth: int = 8,
        learning_rate: float = 0.055263064405123206,
        subsample: float = 0.9888455338600703,
        colsample_bytree: float = 0.9528625653858853,
        gamma: float = 2.3702565701276304,
        min_child_weight: int = 9,
        reg_alpha: float = 0.1400481305977997,
        reg_lambda: float = 2.2312748382635074e-05,
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
        self.model = xgb.XGBRegressor(**self._params)

    def fit(self, X, y, sample_weight=None):
        y_arr = coerce_goal_array(y)
        X_pd = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=range(X.shape[1]))

        # Augment: original rows predict goals_A, mirrored rows predict goals_B
        X_aug = pd.concat([X_pd, mirror_features(X_pd)], ignore_index=True)
        y_aug = np.concatenate([y_arr[:, 0], y_arr[:, 1]])
        w_aug = None
        if sample_weight is not None:
            w = np.asarray(sample_weight)
            w_aug = np.concatenate([w, w])

        self.model.fit(X_aug, y_aug, sample_weight=w_aug)
        return self

    def predict(self, X) -> np.ndarray:
        X_pd = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=range(X.shape[1]))
        goals_a = self.model.predict(X_pd)
        goals_b = self.model.predict(mirror_features(X_pd))
        return ensure_non_negative(np.column_stack([goals_a, goals_b]))

    def feature_importances(self, feature_names: list[str] | None = None) -> pd.DataFrame:
        imp = self.model.feature_importances_
        names = feature_names if feature_names is not None else list(range(len(imp)))
        return pd.DataFrame({"feature": names, "importance": imp}).sort_values(
            "importance", ascending=False
        )
