"""
train_v5.py — Final V5 model retrain on ALL data + calibration fit.

Saves:
  models/production_model_v5.joblib
  models/production_config_v5.json

V4 artifacts are never modified. Run ONLY after Block 4 holdout confirms v5 > v4.

Usage:
    python scripts/train_v5.py [--decay 0.95] [--blend 0.7] [--alpha 0.0]
                                [--et-scale 0.333] [--blend-weight 0.0]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.protocol import (
    DATASET_PATH,
    HOLDOUT,
    TARGET_COLS,
    TUNING_FOLDS,
    generate_oof_lambdas,
    load_and_prepare_dataset,
)
from src.features.feature_columns import FEATURE_COLS_V5_PROD
from src.models.base import load_model_dataset
from src.models.ensemble import EnsembleGoalModel
from src.models.lgbm_model import LGBMGoalModel
from src.models.predictor_v5 import ScorePredictorV5
from src.models.score_grid import fit_calibration_params
from src.models.train import save_model
from src.models.weighting import apply_combined_weighting
from src.models.xgb_model import XGBGoalModel

DEFAULT_SAVE_PATH = "models/production_model_v5.joblib"
DEFAULT_CONFIG_PATH = "models/production_config_v5.json"
ENSEMBLE_W_LGBM = 0.9
ENSEMBLE_W_XGB = 0.1


def build_ensemble() -> EnsembleGoalModel:
    return EnsembleGoalModel(
        [LGBMGoalModel(), XGBGoalModel()],
        weights=[ENSEMBLE_W_LGBM, ENSEMBLE_W_XGB],
    )


def train_v5(
    decay_rate: float = 0.9,
    competition_blend: float = 0.5,
    alpha: float = 0.0,
    et_scale: float = 30.0 / 90.0,
    blend_weight: float = 0.0,
    save_path: str = DEFAULT_SAVE_PATH,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> tuple:
    """
    Train final V5 model on ALL data (including WC 2022).

    Steps:
    1. Load + prepare dataset (add V5 features)
    2. Fit calibration params via OOF on tuning folds (WC 2022 still excluded here)
    3. Retrain on ALL data with chosen weights
    4. Build ScorePredictorV5 and save model + config

    Returns (predictor, config_dict)
    """
    print("=== V5 Production Training ===")
    print(f"Params: decay={decay_rate}  comp_blend={competition_blend}  "
          f"alpha={alpha}  et_scale={et_scale:.3f}  blend_weight={blend_weight}")

    print("\nLoading dataset...")
    df = load_and_prepare_dataset(DATASET_PATH)
    print(f"Dataset: {len(df)} rows after V5 feature engineering")

    missing_v5 = [c for c in FEATURE_COLS_V5_PROD if c not in df.columns]
    if missing_v5:
        raise ValueError(f"Missing V5 feature columns: {missing_v5}")

    # ── Step 1: fit calibration on OOF lambdas (tuning folds, WC 2022 excluded) ──
    print("\nStep 1: Fitting calibration params on OOF tuning folds...")
    use_decay = decay_rate < 1.0

    def weights_fn(train_df):
        return apply_combined_weighting(
            train_df,
            apply_decay=use_decay,
            decay_rate=decay_rate,
            reference_year=2026,
            competition_weight=competition_blend,
            temporal_weight=1.0 - competition_blend,
        )

    oof_df = generate_oof_lambdas(
        df=df,
        model_factory=build_ensemble,
        feature_cols=FEATURE_COLS_V5_PROD,
        weights_fn=weights_fn,
        folds=TUNING_FOLDS,
        holdout=HOLDOUT,
    )

    la_oof = oof_df["pred_lambda_a"].values
    lb_oof = oof_df["pred_lambda_b"].values
    ga_oof = oof_df["goals_A"].values.astype(int)
    gb_oof = oof_df["goals_B"].values.astype(int)

    calib = fit_calibration_params(la_oof, lb_oof, ga_oof, gb_oof)
    print(f"  scale_c={calib['scale_c']:.4f}  rho={calib['rho']:.4f}  "
          f"affine=({calib.get('affine_a', 0):.3f}, {calib.get('affine_b', 1):.3f})")

    # ── Step 2: retrain on ALL data ───────────────────────────────────────────
    print("\nStep 2: Retraining on ALL data (including WC 2022)...")
    X_all = df[FEATURE_COLS_V5_PROD].fillna(0)   # DataFrame — mirror_features needs named cols
    y_all = df[TARGET_COLS].values

    w_all = weights_fn(df)

    model = build_ensemble()
    try:
        model.fit(X_all, y_all, sample_weight=w_all)
    except TypeError:
        model.fit(X_all, y_all)
    print("Model trained.")

    # ── Step 3: save model + config ────────────────────────────────────────────
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, save_path)
    print(f"Model saved → {save_path}")

    predictor = ScorePredictorV5(
        model=model,
        feature_cols=FEATURE_COLS_V5_PROD,
        rho=calib["rho"],
        scale_c=calib["scale_c"],
        affine_a=calib.get("affine_a"),
        affine_b=calib.get("affine_b"),
        alpha=alpha,
        et_scale=et_scale,
        blend_weight=blend_weight,
        competition_importance=4.0,
    )

    config = {
        "version": "v5",
        "feature_cols": FEATURE_COLS_V5_PROD,
        "target_cols": TARGET_COLS,
        "trained_on_date": datetime.today().strftime("%Y-%m-%d"),
        "dataset_info": {
            "n_samples": len(df),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
        },
        "training_params": {
            "decay_rate": decay_rate,
            "competition_blend": competition_blend,
            "ensemble_weights": {"lgbm": ENSEMBLE_W_LGBM, "xgb": ENSEMBLE_W_XGB},
        },
        "calibration": calib,
        "predictor_config": predictor.config(),
        "tuning_folds": [list(f) for f in TUNING_FOLDS],
        "holdout": list(HOLDOUT),
    }

    config_path_obj = Path(config_path)
    config_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path_obj, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved  → {config_path}")

    return predictor, config


def main():
    parser = argparse.ArgumentParser(description="Train V5 production model")
    parser.add_argument("--decay", type=float, default=0.9,
                        help="Temporal decay rate (1.0 = off)")
    parser.add_argument("--blend", type=float, default=0.5,
                        help="Competition weight blend ratio")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Outcome bonus in pick_score")
    parser.add_argument("--et-scale", type=float, default=30.0 / 90.0,
                        help="Extra-time lambda scale factor")
    parser.add_argument("--blend-weight", type=float, default=0.0,
                        help="Tournament lambda blending weight (0 = disabled)")
    parser.add_argument("--save-path", default=DEFAULT_SAVE_PATH)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    train_v5(
        decay_rate=args.decay,
        competition_blend=args.blend,
        alpha=args.alpha,
        et_scale=args.et_scale,
        blend_weight=args.blend_weight,
        save_path=args.save_path,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
