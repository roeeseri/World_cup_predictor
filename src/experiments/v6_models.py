"""
V6 goal models: same architecture/hyperparams as production (LGBM 0.9 + XGB 0.1,
mirror-augmented symmetric training) but with the V6 mirror function so every
feature — including rest_diff and the new tournament-form diffs — stays
team-swap consistent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import coerce_goal_array, ensure_non_negative
from src.models.ensemble import EnsembleGoalModel
from src.experiments.v6_features import mirror_features_v6


class V6LGBMGoalModel:
    """LGBMGoalModel clone using mirror_features_v6 (hyperparams identical to prod)."""

    def __init__(self, **overrides) -> None:
        import lightgbm as lgb
        params = dict(
            n_estimators=276, max_depth=9, learning_rate=0.02597955300094567,
            num_leaves=233, min_child_samples=33, subsample=0.7334473346895813,
            colsample_bytree=0.835882051416841, reg_alpha=0.07980827874410094,
            reg_lambda=0.0017299303935923262, random_state=42, verbose=-1,
            objective="poisson", n_jobs=-1,
        )
        params.update(overrides)
        self.model = lgb.LGBMRegressor(**params)

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        y_arr = coerce_goal_array(y)
        X_aug = pd.concat([X, mirror_features_v6(X)], ignore_index=True)
        y_aug = np.concatenate([y_arr[:, 0], y_arr[:, 1]])
        w_aug = None
        if sample_weight is not None:
            w = np.asarray(sample_weight)
            w_aug = np.concatenate([w, w])
        self.model.fit(X_aug, y_aug, sample_weight=w_aug)
        return self

    def predict(self, X: pd.DataFrame):
        goals_a = self.model.predict(X)
        goals_b = self.model.predict(mirror_features_v6(X))
        return ensure_non_negative(np.column_stack([goals_a, goals_b]))


class V6XGBGoalModel:
    """XGBGoalModel clone using mirror_features_v6 (hyperparams identical to prod)."""

    def __init__(self, **overrides) -> None:
        import xgboost as xgb
        params = dict(
            n_estimators=268, max_depth=8, learning_rate=0.055263064405123206,
            subsample=0.9888455338600703, colsample_bytree=0.9528625653858853,
            gamma=2.3702565701276304, min_child_weight=9,
            reg_alpha=0.1400481305977997, reg_lambda=2.2312748382635074e-05,
            random_state=42, objective="count:poisson", tree_method="hist",
            verbosity=0, n_jobs=-1,
        )
        params.update(overrides)
        self.model = xgb.XGBRegressor(**params)

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        y_arr = coerce_goal_array(y)
        X_aug = pd.concat([X, mirror_features_v6(X)], ignore_index=True)
        y_aug = np.concatenate([y_arr[:, 0], y_arr[:, 1]])
        w_aug = None
        if sample_weight is not None:
            w = np.asarray(sample_weight)
            w_aug = np.concatenate([w, w])
        self.model.fit(X_aug, y_aug, sample_weight=w_aug)
        return self

    def predict(self, X: pd.DataFrame):
        goals_a = self.model.predict(X)
        goals_b = self.model.predict(mirror_features_v6(X))
        return ensure_non_negative(np.column_stack([goals_a, goals_b]))


def build_v6_ensemble() -> EnsembleGoalModel:
    """Production-matching ensemble (0.9 LGBM / 0.1 XGB) with V6 mirror."""
    return EnsembleGoalModel([V6LGBMGoalModel(), V6XGBGoalModel()], weights=[0.9, 0.1])
