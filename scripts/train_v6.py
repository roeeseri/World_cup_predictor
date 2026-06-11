"""
train_v6.py — Final V6 model retrain on ALL data + drawband calibration fit.

V6 = V5-prod feature set (20 features) + corrected mirror (rest_diff fix)
   + drawband decision rule (cond_floor tb=0.5 fallback; most-likely draw score
     when the calibrated DC grid draw mass >= 0.33).

Saves:
  models/production_model_v6.joblib
  models/production_config_v6.json

V4/V5 artifacts are never modified.

Steps:
1. Generate OOF lambdas on the 4 tuning folds (WC 2022 excluded) with the V6
   ensemble → fit scale_c (Poisson MLE) + Dixon-Coles rho. These freeze the
   drawband calibration; they are never fit on WC 2022 or 2026 data.
2. Retrain on ALL data (incl. WC 2022 + any 2026 results in the dataset).
3. Save model + config.

Usage:
    python scripts/train_v6.py [--decay 0.9] [--blend 0.5]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.protocol import (
    DATASET_PATH,
    HOLDOUT,
    TARGET_COLS,
    TUNING_FOLDS,
    generate_oof_lambdas,
    load_and_prepare_dataset,
)
from src.features.feature_columns import FEATURE_COLS_V5_PROD
from src.models.goal_models_v6 import build_v6_ensemble
from src.models.score_grid import apply_lambda_scale, fit_lambda_scale, fit_rho
from src.models.train import save_model
from src.models.weighting import apply_combined_weighting

DEFAULT_SAVE_PATH = "models/production_model_v6.joblib"
DEFAULT_CONFIG_PATH = "models/production_config_v6.json"

# Drawband decision params (tuned on the 4 OOF folds; see outputs/experiments/v6/V6_REPORT.md)
DRAW_THRESHOLD = 0.33
THRESHOLD_B = 0.5


def train_v6(
    decay_rate: float = 0.9,
    competition_blend: float = 0.5,
    save_path: str = DEFAULT_SAVE_PATH,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> dict:
    print("=== V6 Production Training ===")
    print(f"Params: decay={decay_rate}  comp_blend={competition_blend}  "
          f"features={len(FEATURE_COLS_V5_PROD)} (V5_PROD, corrected mirror)")

    print("\nLoading dataset...")
    df = load_and_prepare_dataset(DATASET_PATH)
    print(f"Dataset: {len(df)} rows")

    missing = [c for c in FEATURE_COLS_V5_PROD if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    def weights_fn(train_df):
        return apply_combined_weighting(
            train_df,
            apply_decay=decay_rate < 1.0,
            decay_rate=decay_rate,
            reference_year=2026,
            competition_weight=competition_blend,
            temporal_weight=1.0 - competition_blend,
        )

    # ── Step 1: OOF lambdas → drawband calibration (WC 2022 excluded) ─────────
    print("\nStep 1: OOF lambdas on tuning folds (WC 2022 excluded)...")
    oof = generate_oof_lambdas(
        df=df,
        model_factory=build_v6_ensemble,
        feature_cols=FEATURE_COLS_V5_PROD,
        weights_fn=weights_fn,
        folds=TUNING_FOLDS,
        holdout=HOLDOUT,
    )
    la = oof["pred_lambda_a"].values
    lb = oof["pred_lambda_b"].values
    ga = oof["goals_A"].values
    gb = oof["goals_B"].values

    scale_c = fit_lambda_scale(la, lb, ga, gb)
    la_s, lb_s = apply_lambda_scale(la, lb, scale_c)
    rho = fit_rho(la_s, lb_s, ga.astype(int), gb.astype(int))
    print(f"  Calibration: scale_c={scale_c:.4f}  rho={rho:.4f}")

    # ── Step 2: retrain on ALL data ───────────────────────────────────────────
    print("\nStep 2: Retraining on ALL data (incl. WC 2022)...")
    X_all = df[FEATURE_COLS_V5_PROD].fillna(0)
    y_all = df[TARGET_COLS].values
    model = build_v6_ensemble()
    model.fit(X_all, y_all, sample_weight=weights_fn(df))
    print("Model trained.")

    # ── Step 3: save ──────────────────────────────────────────────────────────
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    save_model(model, save_path)
    print(f"Model saved: {save_path}")

    config = {
        "version": "v6",
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
            "ensemble_weights": {"lgbm": 0.9, "xgb": 0.1},
            "mirror": "mirror_features_v6 (rest_diff fix)",
        },
        "drawband": {
            "draw_threshold": DRAW_THRESHOLD,
            "threshold_b": THRESHOLD_B,
            "scale_c": float(scale_c),
            "rho": float(rho),
        },
        "tuning_folds": [list(f) for f in TUNING_FOLDS],
        "holdout": list(HOLDOUT),
        "holdout_result": "39.1% exact / 90.6% outcome (see outputs/experiments/v6/V6_REPORT.md)",
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_path}")
    return config


def main():
    parser = argparse.ArgumentParser(description="Train V6 production model")
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--blend", type=float, default=0.5)
    parser.add_argument("--save-path", default=DEFAULT_SAVE_PATH)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    train_v6(
        decay_rate=args.decay,
        competition_blend=args.blend,
        save_path=args.save_path,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
