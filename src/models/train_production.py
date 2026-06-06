"""
Production training script for World Cup score predictor.

Uses the 21 production features from feature_columns.py, proper multi-fold
WC cross-validation for evaluation, and saves model + config for inference.

Usage:
    python -m src.models.train_production --model-type lgbm --evaluate
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.feature_columns import FEATURE_COLS
from src.models.base import load_model_dataset
from src.models.ensemble import EnsembleGoalModel
from src.models.lgbm_model import LGBMGoalModel
from src.models.optuna_tuning import exact_score_accuracy, result_accuracy
from src.models.poisson_model import PoissonGoalModel
from src.models.score_conversion import win_draw_loss_probs
from src.models.train import save_model
from src.models.weighting import apply_competition_weights
from src.models.world_cup_utils import create_wc_cv_splits
from src.models.xgb_model import XGBGoalModel

TARGET_COLS = ["goals_A", "goals_B"]
DEFAULT_SAVE_PATH = "models/production_model.joblib"
DEFAULT_CONFIG_PATH = "models/production_config.json"
WC_CV_YEARS = [2014, 2018, 2022]


ENSEMBLE_W_LGBM = 0.8
ENSEMBLE_W_XGB  = 0.2


def _build_model(model_type: str):
    if model_type == "lgbm":
        return LGBMGoalModel()
    if model_type == "xgboost":
        return XGBGoalModel()
    if model_type == "poisson":
        return PoissonGoalModel()
    if model_type == "ensemble":
        return EnsembleGoalModel(
            [LGBMGoalModel(), XGBGoalModel()],
            weights=[ENSEMBLE_W_LGBM, ENSEMBLE_W_XGB],
        )
    raise ValueError(f"Unknown model_type: {model_type!r}. "
                     "Choose from: lgbm, xgboost, poisson, ensemble")


def _rps_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute average RPS using Poisson grid W/D/L probs from expected goals."""
    from src.evaluation.metrics import rps_batch
    probs = np.array([win_draw_loss_probs(la, lb) for la, lb in y_pred])
    return rps_batch(y_true, probs)


def run_wc_cv_evaluation(
    df: pd.DataFrame,
    model_type: str,
    weights: np.ndarray | None,
    fold_years: list[int] = WC_CV_YEARS,
) -> dict:
    """
    Evaluate model on each WC fold (train=all non-fold rows, val=WC matches).
    Returns per-fold and average metrics.
    """
    print(f"\nRunning WC cross-validation ({fold_years})...")
    splits = create_wc_cv_splits(df, fold_years, FEATURE_COLS, TARGET_COLS)

    cv_results = {}
    for year, (X_train, y_train, X_val, y_val) in splits.items():
        # Align weights to training rows (same mask used in create_wc_cv_splits)
        w_train = None
        if weights is not None:
            val_mask = (
                (df["tournament_year"] == year) &
                (df["competition"].str.strip().str.lower() == "world cup")
            )
            w_train = weights[~val_mask]

        model = _build_model(model_type)
        model.fit(X_train, y_train, sample_weight=w_train)

        y_pred = np.clip(model.predict(X_val), 0, None)
        exact = exact_score_accuracy(y_val, y_pred)
        result = result_accuracy(y_val, y_pred)
        rps = _rps_from_predictions(y_val, y_pred)

        cv_results[year] = {
            "exact_score_accuracy": float(exact),
            "result_accuracy": float(result),
            "rps": float(rps),
            "n_val": int(len(y_val)),
        }

        print(f"  WC {year}: exact={exact*100:.1f}%  result={result*100:.1f}%  "
              f"RPS={rps:.4f}  (n={len(y_val)})")

    avg_exact = np.mean([v["exact_score_accuracy"] for v in cv_results.values()])
    avg_result = np.mean([v["result_accuracy"] for v in cv_results.values()])
    avg_rps = np.mean([v["rps"] for v in cv_results.values()])

    cv_results["average"] = {
        "exact_score_accuracy": float(avg_exact),
        "result_accuracy": float(avg_result),
        "rps": float(avg_rps),
    }

    print(f"  Average: exact={avg_exact*100:.1f}%  result={avg_result*100:.1f}%  "
          f"RPS={avg_rps:.4f}")

    return cv_results


def train_production_model(
    model_type: str = "lgbm",
    use_weights: bool = True,
    save_path: str = DEFAULT_SAVE_PATH,
    config_path: str = DEFAULT_CONFIG_PATH,
    evaluate: bool = True,
) -> tuple:
    """
    Train production model on all available data and save.

    Args:
        model_type: "lgbm", "xgboost", "poisson", or "ensemble"
        use_weights: Apply competition-based sample weighting
        save_path: Path to save the model artifact
        config_path: Path to save model config JSON
        evaluate: Run WC cross-validation before final training

    Returns:
        (model, config_dict)
    """
    print("Loading dataset...")
    dataset_path = Path("data/processed/model_dataset.csv")
    if not dataset_path.exists():
        dataset_path = Path("../data/processed/model_dataset.csv")
    df = load_model_dataset(path=dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Validate required columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required feature columns: {missing}")
    missing_tgt = [c for c in TARGET_COLS if c not in df.columns]
    if missing_tgt:
        raise ValueError(f"Dataset missing target columns: {missing_tgt}")

    print(f"Dataset: {len(df)} matches, {len(FEATURE_COLS)} features")

    weights = None
    if use_weights:
        weights = apply_competition_weights(df)
        print("Applied competition-based sample weighting.")

    cv_metrics = {}
    if evaluate:
        cv_metrics = run_wc_cv_evaluation(df, model_type, weights)

    # Train final model on ALL data
    print(f"\nTraining final {model_type} model on all {len(df)} matches...")
    X_all = df[FEATURE_COLS].fillna(0)
    y_all = df[TARGET_COLS].values

    model = _build_model(model_type)
    model.fit(X_all, y_all, sample_weight=weights)
    print("Model trained.")

    # Save artifacts
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, save_path)
    print(f"Model saved -> {save_path}")

    config = {
        "model_type": model_type,
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "trained_on_date": datetime.today().strftime("%Y-%m-%d"),
        "dataset_info": {
            "n_samples": len(df),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
        },
        "use_weights": use_weights,
        "cv_metrics": cv_metrics,
    }

    config_path_obj = Path(config_path)
    config_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path_obj, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved  -> {config_path}")

    return model, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WC predictor production model")
    parser.add_argument(
        "--model-type",
        default="lgbm",
        choices=["lgbm", "xgboost", "poisson", "ensemble"],
        help="Model architecture",
    )
    parser.add_argument("--no-weights", action="store_true", help="Skip competition weighting")
    parser.add_argument("--no-evaluate", action="store_true", help="Skip WC cross-validation")
    parser.add_argument("--save-path", default=DEFAULT_SAVE_PATH, help="Model save path")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH, help="Config save path")

    args = parser.parse_args()

    train_production_model(
        model_type=args.model_type,
        use_weights=not args.no_weights,
        save_path=args.save_path,
        config_path=args.config_path,
        evaluate=not args.no_evaluate,
    )
