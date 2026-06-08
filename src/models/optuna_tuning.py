"""
Optuna-based hyperparameter optimization for goal prediction models.

Finds best hyperparameters by maximizing exact score accuracy, averaged
across multiple World Cup cross-validation folds.

CV strategy: for each WC fold (2014, 2018, 2022), train on ALL non-fold
rows (including post-fold data), test on that WC. Hyperparameters are
chosen to maximize average accuracy across folds.
"""

from __future__ import annotations

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from .lgbm_model import LGBMGoalModel
from .poisson_model import PoissonGoalModel
from .tree_model import TreeGoalModel
from .xgb_model import XGBGoalModel
from .base import coerce_goal_array


def exact_score_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Exact score accuracy using round — best calibration target for Poisson lambdas."""
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


# ---------------------------------------------------------------------------
# Base tuner mixin — shared bookkeeping used by all single-model tuners
# ---------------------------------------------------------------------------

class _BaseTuner:
    def __init__(self, X_train, y_train, X_val, y_val, weights_train=None):
        self.X_train = X_train
        self.y_train = coerce_goal_array(y_train)
        self.X_val = X_val
        self.y_val = coerce_goal_array(y_val)
        self.weights_train = weights_train
        self.best_model = None
        self.best_params = None
        self.best_exact_acc = -1.0

    def _record(self, model, params, acc):
        if acc > self.best_exact_acc:
            self.best_exact_acc = acc
            self.best_model = model
            self.best_params = params

    def optimize(self, n_trials: int = 50, timeout: int = 600, verbose: bool = True) -> dict:
        sampler = TPESampler(seed=42)
        pruner = MedianPruner()
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout,
                       show_progress_bar=verbose)
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


# ---------------------------------------------------------------------------
# Single-fold tuners
# ---------------------------------------------------------------------------

class PoissonTuner(_BaseTuner):
    """Optuna-based tuning for PoissonGoalModel."""

    def objective(self, trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 0.01, 100.0, log=True)
        max_iter = trial.suggest_int("max_iter", 500, 2000, step=100)
        model = PoissonGoalModel(alpha=alpha, max_iter=max_iter)
        try:
            model.fit(self.X_train, self.y_train, sample_weight=self.weights_train)
        except Exception:
            return 0.0
        y_pred = np.clip(model.predict(self.X_val), 0, None)
        acc = exact_score_accuracy(self.y_val, y_pred)
        self._record(model, trial.params, acc)
        return acc


class TreeTuner(_BaseTuner):
    """Optuna-based tuning for TreeGoalModel."""

    def objective(self, trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int("n_estimators", 50, 300, step=10)
        max_depth = trial.suggest_int("max_depth", 5, 30)
        model = TreeGoalModel(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        try:
            model.fit(self.X_train, self.y_train, sample_weight=self.weights_train)
        except Exception:
            return 0.0
        y_pred = np.clip(model.predict(self.X_val), 0, None)
        acc = exact_score_accuracy(self.y_val, y_pred)
        self._record(model, trial.params, acc)
        return acc


class LGBMTuner(_BaseTuner):
    """Optuna-based tuning for LGBMGoalModel."""

    def objective(self, trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 4, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 1.0, log=True),
        }
        model = LGBMGoalModel(**params)
        try:
            model.fit(self.X_train, self.y_train, sample_weight=self.weights_train)
        except Exception:
            return 0.0
        y_pred = np.clip(model.predict(self.X_val), 0, None)
        acc = exact_score_accuracy(self.y_val, y_pred)
        self._record(model, trial.params, acc)
        return acc


class XGBTuner(_BaseTuner):
    """Optuna-based tuning for XGBGoalModel."""

    def objective(self, trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
        }
        model = XGBGoalModel(**params)
        try:
            model.fit(self.X_train, self.y_train, sample_weight=self.weights_train)
        except Exception:
            return 0.0
        y_pred = np.clip(model.predict(self.X_val), 0, None)
        acc = exact_score_accuracy(self.y_val, y_pred)
        self._record(model, trial.params, acc)
        return acc


# ---------------------------------------------------------------------------
# Multi-fold CV tuner wrapper
# ---------------------------------------------------------------------------

class WCCVTuner:
    """
    Tunes any model by averaging exact score accuracy across multiple WC folds.

    cv_splits: dict of {year: (X_train, y_train, X_val, y_val)}
    base_tuner_class: one of LGBMTuner, XGBTuner, PoissonTuner, TreeTuner

    For each Optuna trial, the same hyperparameters are evaluated on every
    fold and averaged. This finds parameters that generalize across WC years
    rather than overfitting to a single tournament.
    """

    def __init__(
        self,
        base_tuner_class,
        cv_splits: dict,
        weights_per_fold: dict | None = None,
    ):
        self.base_tuner_class = base_tuner_class
        self.cv_splits = cv_splits
        self.weights_per_fold = weights_per_fold or {}
        self.best_params: dict | None = None
        self.best_avg_acc: float = -1.0
        self._dummy_tuner: _BaseTuner | None = None

    def _make_dummy_tuner(self) -> _BaseTuner:
        """Build a tuner instance just to re-use its objective param-suggestion logic."""
        first = next(iter(self.cv_splits.values()))
        X_tr, y_tr, X_val, y_val = first
        return self.base_tuner_class(X_tr, y_tr, X_val, y_val)

    def _build_model_from_params(self, params: dict):
        dummy = self._dummy_tuner
        if isinstance(dummy, LGBMTuner):
            return LGBMGoalModel(**params)
        if isinstance(dummy, XGBTuner):
            return XGBGoalModel(**params)
        if isinstance(dummy, PoissonTuner):
            return PoissonGoalModel(**{k: v for k, v in params.items()
                                       if k in ("alpha", "max_iter")})
        if isinstance(dummy, TreeTuner):
            return TreeGoalModel(**{k: v for k, v in params.items()
                                    if k in ("n_estimators", "max_depth")},
                                  random_state=42)
        raise TypeError(f"Unknown tuner class: {type(dummy)}")

    def objective(self, trial: optuna.Trial) -> float:
        # Let dummy tuner suggest params (reuse same search space)
        _ = self._dummy_tuner.objective(trial)  # side-effect: trial gets params suggested
        params = trial.params

        fold_accs = []
        for year, (X_tr, y_tr, X_val, y_val) in self.cv_splits.items():
            weights = self.weights_per_fold.get(year)
            model = self._build_model_from_params(params)
            try:
                model.fit(X_tr, coerce_goal_array(y_tr), sample_weight=weights)
            except Exception:
                fold_accs.append(0.0)
                continue
            y_pred = np.clip(model.predict(X_val), 0, None)
            fold_accs.append(exact_score_accuracy(coerce_goal_array(y_val), y_pred))

        avg_acc = float(np.mean(fold_accs)) if fold_accs else 0.0

        if avg_acc > self.best_avg_acc:
            self.best_avg_acc = avg_acc
            self.best_params = params

        return avg_acc

    def optimize(self, n_trials: int = 100, timeout: int = 1800, verbose: bool = True) -> dict:
        self._dummy_tuner = self._make_dummy_tuner()

        # We need a fresh dummy each trial (param suggestion is stateful per trial),
        # so we override objective to rebuild dummy each time.
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def _objective(trial):
            # Fresh dummy tuner for param suggestion (shares search space definition)
            first = next(iter(self.cv_splits.values()))
            X_tr, y_tr, X_val, y_val = first
            dummy = self.base_tuner_class(X_tr, y_tr, X_val, y_val)

            params = {}
            # Manually replicate param suggestion by calling the dummy's objective
            # on a proxy trial — this is cleaner via a separate suggest method.
            # Instead, we call the actual tuner logic inline:
            if self.base_tuner_class == LGBMTuner:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 4, 15),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 16, 256),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 1.0, log=True),
                }
            elif self.base_tuner_class == XGBTuner:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
                }
            elif self.base_tuner_class == PoissonTuner:
                params = {
                    "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                    "max_iter": trial.suggest_int("max_iter", 500, 2000, step=100),
                }
            elif self.base_tuner_class == TreeTuner:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=10),
                    "max_depth": trial.suggest_int("max_depth", 5, 30),
                }
            else:
                raise ValueError(f"Unsupported tuner class: {self.base_tuner_class}")

            fold_accs = []
            for year, (X_tr, y_tr, X_val, y_val) in self.cv_splits.items():
                weights = self.weights_per_fold.get(year)
                if self.base_tuner_class == LGBMTuner:
                    model = LGBMGoalModel(**params)
                elif self.base_tuner_class == XGBTuner:
                    model = XGBGoalModel(**params)
                elif self.base_tuner_class == PoissonTuner:
                    model = PoissonGoalModel(**params)
                else:
                    model = TreeGoalModel(**params, random_state=42)
                try:
                    model.fit(X_tr, coerce_goal_array(y_tr), sample_weight=weights)
                except Exception:
                    fold_accs.append(0.0)
                    continue
                y_pred = np.clip(model.predict(X_val), 0, None)
                fold_accs.append(exact_score_accuracy(coerce_goal_array(y_val), y_pred))

            avg_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
            if avg_acc > self.best_avg_acc:
                self.best_avg_acc = avg_acc
                self.best_params = params
            return avg_acc

        study.optimize(_objective, n_trials=n_trials, timeout=timeout,
                       show_progress_bar=verbose)

        if verbose:
            print(f"\n{'='*70}")
            print(f"Best avg exact score accuracy across folds: {self.best_avg_acc:.4f}")
            print(f"Best parameters: {self.best_params}")
            print(f"{'='*70}")

        return {
            "best_params": self.best_params,
            "best_avg_acc": self.best_avg_acc,
            "study": study,
        }
