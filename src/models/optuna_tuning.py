"""
Optuna-based hyperparameter optimization for goal prediction models.

Finds best hyperparameters by maximizing exact score accuracy on validation set.
"""

from __future__ import annotations

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from .poisson_model import PoissonGoalModel
from .tree_model import TreeGoalModel
from .base import coerce_goal_array


def exact_score_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Exact score accuracy (most important metric)."""
    y_pred_int = np.round(y_pred).astype(int)
    matches = np.all(y_true == y_pred_int, axis=1)
    return np.mean(matches)


def result_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Match result accuracy (win/draw/loss)."""
    y_pred_int = np.round(y_pred).astype(int)
    true_result = np.sign(y_true[:, 0] - y_true[:, 1])
    pred_result = np.sign(y_pred_int[:, 0] - y_pred_int[:, 1])
    return np.mean(true_result == pred_result)


def goal_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in goals."""
    return np.mean(np.abs(y_true.flatten() - y_pred.flatten()))


class PoissonTuner:
    """Optuna-based tuning for PoissonGoalModel."""

    def __init__(self, X_train, y_train, X_val, y_val, weights_train=None, weights_val=None):
        self.X_train = X_train
        self.y_train = coerce_goal_array(y_train)
        self.X_val = X_val
        self.y_val = coerce_goal_array(y_val)
        self.weights_train = weights_train
        self.weights_val = weights_val
        self.best_model = None
        self.best_params = None

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna (maximize exact score accuracy)."""
        # Suggest hyperparameters
        alpha = trial.suggest_float("alpha", 0.01, 100.0, log=True)
        max_iter = trial.suggest_int("max_iter", 500, 2000, step=100)

        # Train model
        model = PoissonGoalModel(alpha=alpha, max_iter=max_iter)
        try:
            model.fit(self.X_train, self.y_train, sample_weight=self.weights_train)
        except Exception as e:
            print(f"Training failed: {e}")
            return 0.0

        # Predict on validation
        y_pred = model.predict(self.X_val)
        y_pred = np.clip(y_pred, 0, None)

        # Calculate exact score accuracy (primary metric)
        exact_acc = exact_score_accuracy(self.y_val, y_pred)

        # Store best model
        if not hasattr(self, "best_exact_acc"):
            self.best_exact_acc = exact_acc
            self.best_model = model
            self.best_params = trial.params
        elif exact_acc > self.best_exact_acc:
            self.best_exact_acc = exact_acc
            self.best_model = model
            self.best_params = trial.params

        return exact_acc

    def optimize(self, n_trials: int = 50, timeout: int = 600, verbose: bool = True) -> dict:
        """Run Optuna optimization."""
        sampler = TPESampler(seed=42)
        pruner = MedianPruner()
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=verbose,
        )

        if verbose:
            print(f"\n{'='*70}")
            print(f"Best exact score accuracy: {self.best_exact_acc:.4f}")
            print(f"Best parameters: {self.best_params}")
            print(f"{'='*70}")

        return {
            "best_params": self.best_params,
            "best_model": self.best_model,
            "best_exact_acc": self.best_exact_acc,
            "study": study,
        }


class TreeTuner:
    """Optuna-based tuning for TreeGoalModel."""

    def __init__(self, X_train, y_train, X_val, y_val, weights_train=None, weights_val=None):
        self.X_train = X_train
        self.y_train = coerce_goal_array(y_train)
        self.X_val = X_val
        self.y_val = coerce_goal_array(y_val)
        self.weights_train = weights_train
        self.weights_val = weights_val
        self.best_model = None
        self.best_params = None

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna (maximize exact score accuracy)."""
        # Suggest hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 50, 300, step=10)
        max_depth = trial.suggest_int("max_depth", 5, 30)

        # Train model
        model = TreeGoalModel(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        try:
            model.fit(self.X_train, self.y_train, sample_weight=self.weights_train)
        except Exception as e:
            print(f"Training failed: {e}")
            return 0.0

        # Predict on validation
        y_pred = model.predict(self.X_val)
        y_pred = np.clip(y_pred, 0, None)

        # Calculate exact score accuracy (primary metric)
        exact_acc = exact_score_accuracy(self.y_val, y_pred)

        # Store best model
        if not hasattr(self, "best_exact_acc"):
            self.best_exact_acc = exact_acc
            self.best_model = model
            self.best_params = trial.params
        elif exact_acc > self.best_exact_acc:
            self.best_exact_acc = exact_acc
            self.best_model = model
            self.best_params = trial.params

        return exact_acc

    def optimize(self, n_trials: int = 50, timeout: int = 600, verbose: bool = True) -> dict:
        """Run Optuna optimization."""
        sampler = TPESampler(seed=42)
        pruner = MedianPruner()
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=verbose,
        )

        if verbose:
            print(f"\n{'='*70}")
            print(f"Best exact score accuracy: {self.best_exact_acc:.4f}")
            print(f"Best parameters: {self.best_params}")
            print(f"{'='*70}")

        return {
            "best_params": self.best_params,
            "best_model": self.best_model,
            "best_exact_acc": self.best_exact_acc,
            "study": study,
        }
