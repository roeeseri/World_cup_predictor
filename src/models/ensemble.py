from __future__ import annotations

import numpy as np

from .base import ensure_non_negative


class EnsembleGoalModel:
    def __init__(self, models: list, weights: list[float] | None = None) -> None:
        if not models:
            raise ValueError("EnsembleGoalModel requires at least one model.")
        self.models = models
        self.weights = weights

    def fit(self, X, y, sample_weight=None):
        for model in self.models:
            try:
                model.fit(X, y, sample_weight=sample_weight)
            except TypeError:
                model.fit(X, y)
        return self

    def predict(self, X):
        preds = [model.predict(X) for model in self.models]
        pred_stack = np.stack(preds, axis=0)

        if self.weights is None:
            return np.mean(pred_stack, axis=0)

        weights = np.asarray(self.weights, dtype=float)
        if weights.shape[0] != pred_stack.shape[0]:
            raise ValueError("Weights length must match number of models.")
        weights = weights / np.sum(weights)
        return ensure_non_negative(np.tensordot(weights, pred_stack, axes=(0, 0)))
