"""
evaluate_v5.py — Leak-free V5 candidate evaluation on tuning folds.

Usage:
    python scripts/evaluate_v5.py [--features v4|v5] [--decay 0.95] [--blend 0.7]
                                   [--alpha 0.0] [--et-scale 0.333]
                                   [--holdout]  # one-shot WC 2022 evaluation

Outputs JSON + CSV to outputs/evaluation/v5/

NOTE: never uses the production_model_v4 or v5 artifacts — trains fresh per fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.protocol import (
    DATASET_PATH,
    HOLDOUT,
    TARGET_COLS,
    TUNING_FOLDS,
    aggregate_oof_results,
    create_holdout_split,
    generate_oof_lambdas,
    load_and_prepare_dataset,
    log_holdout_evaluation,
    v4_probs_fn,
    v4_score_fn,
    compute_fold_metrics,
)
from src.models.ensemble import EnsembleGoalModel
from src.models.lgbm_model import LGBMGoalModel
from src.models.score_grid import (
    fit_calibration_params,
    make_score_fn,
    apply_lambda_scale,
)
from src.models.weighting import apply_combined_weighting
from src.features.feature_columns import FEATURE_COLS, FEATURE_COLS_V5

OUTPUT_DIR = Path("outputs/evaluation/v5")


def build_model_factory(ensemble_w_lgbm: float = 0.9, ensemble_w_xgb: float = 0.1):
    from src.models.xgb_model import XGBGoalModel

    def factory():
        return EnsembleGoalModel(
            [LGBMGoalModel(), XGBGoalModel()],
            weights=[ensemble_w_lgbm, ensemble_w_xgb],
        )

    return factory


def build_weights_fn(decay_rate: float, competition_blend: float, reference_year: int = 2026):
    def weights_fn(train_df: pd.DataFrame) -> np.ndarray:
        use_decay = decay_rate < 1.0
        return apply_combined_weighting(
            train_df,
            apply_decay=use_decay,
            decay_rate=decay_rate,
            reference_year=reference_year,
            competition_weight=competition_blend,
            temporal_weight=1.0 - competition_blend,
        )

    return weights_fn


def run_tuning_evaluation(
    df: pd.DataFrame,
    feature_cols: list[str],
    weights_fn,
    alpha: float,
    et_scale: float,
    strategy_name: str,
) -> dict:
    """Run 4-fold OOF evaluation, fit calibration params, score with DC grid."""
    print(f"\n=== {strategy_name} ===")
    print(f"Features: {len(feature_cols)}  alpha={alpha}  et_scale={et_scale:.3f}")

    factory = build_model_factory()

    # Generate OOF lambdas
    oof_df = generate_oof_lambdas(
        df=df,
        model_factory=factory,
        feature_cols=feature_cols,
        weights_fn=weights_fn,
        folds=TUNING_FOLDS,
        holdout=HOLDOUT,
    )

    la_oof = oof_df["pred_lambda_a"].values
    lb_oof = oof_df["pred_lambda_b"].values
    ga_oof = oof_df["goals_A"].values.astype(int)
    gb_oof = oof_df["goals_B"].values.astype(int)

    # V4 baseline: floor(λ+0.1)
    print("\n  V4 decision rule (floor+0.1) on same OOF lambdas:")
    v4_results = aggregate_oof_results(oof_df, v4_score_fn, v4_probs_fn)
    print(f"    exact={v4_results['overall']['exact_score_accuracy']:.3f}  "
          f"outcome={v4_results['overall']['outcome_accuracy']:.3f}  "
          f"draw_rate_pred={v4_results['overall']['predicted_draw_rate']:.3f}  "
          f"draw_rate_actual={v4_results['overall']['actual_draw_rate']:.3f}")

    # Fit calibration on OOF
    calib = fit_calibration_params(la_oof, lb_oof, ga_oof, gb_oof)
    print(f"\n  Calibration: scale_c={calib['scale_c']:.4f}  "
          f"rho={calib['rho']:.4f}  "
          f"affine=({calib.get('affine_a', 0):.3f}, {calib.get('affine_b', 1):.3f})")

    # V5 score / probs with calibrated DC grid (scale calibration)
    score_fn_scale, probs_fn_scale = make_score_fn(
        rho=calib["rho"],
        scale_c=calib["scale_c"],
        alpha=alpha,
        knockout=False,
        et_scale=et_scale,
    )

    print("\n  V5 DC grid (scale calib):")
    v5_scale_results = aggregate_oof_results(oof_df, score_fn_scale, probs_fn_scale)
    print(f"    exact={v5_scale_results['overall']['exact_score_accuracy']:.3f}  "
          f"outcome={v5_scale_results['overall']['outcome_accuracy']:.3f}  "
          f"draw_rate_pred={v5_scale_results['overall']['predicted_draw_rate']:.3f}")

    # V5 with affine calibration
    score_fn_affine, probs_fn_affine = make_score_fn(
        rho=calib["rho"],
        affine_a=calib.get("affine_a"),
        affine_b=calib.get("affine_b"),
        alpha=alpha,
        knockout=False,
        et_scale=et_scale,
    )

    print("  V5 DC grid (affine calib):")
    v5_affine_results = aggregate_oof_results(oof_df, score_fn_affine, probs_fn_affine)
    print(f"    exact={v5_affine_results['overall']['exact_score_accuracy']:.3f}  "
          f"outcome={v5_affine_results['overall']['outcome_accuracy']:.3f}  "
          f"draw_rate_pred={v5_affine_results['overall']['predicted_draw_rate']:.3f}")

    # Per-fold breakdown
    print("\n  Per-fold exact score accuracy (scale calib):")
    for fold_name, metrics in v5_scale_results["per_fold"].items():
        print(f"    {fold_name}: exact={metrics['exact_score_accuracy']:.3f}  "
              f"outcome={metrics['outcome_accuracy']:.3f}  "
              f"draw_rate={metrics['predicted_draw_rate']:.3f}")

    return {
        "strategy": strategy_name,
        "feature_cols": feature_cols,
        "calibration": calib,
        "v4_baseline": v4_results["overall"],
        "v5_scale": v5_scale_results["overall"],
        "v5_scale_per_fold": v5_scale_results["per_fold"],
        "v5_affine": v5_affine_results["overall"],
        "v5_affine_per_fold": v5_affine_results["per_fold"],
    }


def run_holdout_evaluation(
    df: pd.DataFrame,
    feature_cols: list[str],
    weights_fn,
    calib_params: dict,
    alpha: float,
    strategy_name: str,
) -> dict:
    """One-shot WC 2022 holdout evaluation. Call at most once per candidate."""
    print(f"\n=== HOLDOUT EVALUATION: {strategy_name} ===")
    print("*** WC 2022 — evaluate at most once per candidate ***")

    train_df, holdout_df = create_holdout_split(df, HOLDOUT)

    factory = build_model_factory()
    model = factory()

    X_train = train_df[feature_cols].fillna(0)   # DataFrame — mirror_features needs named cols
    y_train = train_df[TARGET_COLS].values
    w_train = weights_fn(train_df) if weights_fn is not None else None

    if w_train is not None:
        try:
            model.fit(X_train, y_train, sample_weight=w_train)
        except TypeError:
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    X_holdout = holdout_df[feature_cols].fillna(0)   # DataFrame — mirror_features needs named cols
    y_true = holdout_df[TARGET_COLS].values.astype(int)
    preds = np.clip(model.predict(X_holdout), 0.0, None)
    la, lb = preds[:, 0], preds[:, 1]

    # V4 decision rule
    v4_scores = np.array([v4_score_fn(a, b) for a, b in zip(la, lb)])
    v4_metrics = compute_fold_metrics(
        y_true, v4_scores,
        probs=np.array([v4_probs_fn(a, b) for a, b in zip(la, lb)])
    )

    # V5 with calibrated DC grid
    score_fn, probs_fn = make_score_fn(
        rho=calib_params["rho"],
        scale_c=calib_params["scale_c"],
        alpha=alpha,
    )
    v5_scores = np.array([score_fn(a, b) for a, b in zip(la, lb)])
    v5_metrics = compute_fold_metrics(
        y_true, v5_scores,
        probs=np.array([probs_fn(a, b) for a, b in zip(la, lb)])
    )

    print(f"  V4 exact={v4_metrics['exact_score_accuracy']:.3f}  "
          f"outcome={v4_metrics['outcome_accuracy']:.3f}  "
          f"draw_rate_pred={v4_metrics['predicted_draw_rate']:.3f}")
    print(f"  V5 exact={v5_metrics['exact_score_accuracy']:.3f}  "
          f"outcome={v5_metrics['outcome_accuracy']:.3f}  "
          f"draw_rate_pred={v5_metrics['predicted_draw_rate']:.3f}")

    results = {
        "holdout": str(HOLDOUT),
        "n_matches": len(holdout_df),
        "v4_floor_rule": v4_metrics,
        "v5_dc_grid": v5_metrics,
    }

    log_holdout_evaluation(results, strategy_name)
    return results


def main():
    parser = argparse.ArgumentParser(description="V5 leak-free evaluation on tuning folds")
    parser.add_argument("--features", choices=["v4", "v5", "both"], default="both",
                        help="Feature set to evaluate")
    parser.add_argument("--decay", type=float, default=0.0,
                        help="Temporal decay rate (0 = off, 0.95 = 5%/yr)")
    parser.add_argument("--blend", type=float, default=0.7,
                        help="Competition weight blend ratio (0.7 = 70% comp / 30% temporal)")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Outcome bonus weight in pick_score (0 = pure exact-score)")
    parser.add_argument("--et-scale", type=float, default=30.0 / 90.0,
                        help="ET scale factor (default 30/90 ≈ 0.333)")
    parser.add_argument("--holdout", action="store_true",
                        help="Run WC 2022 holdout (one-shot — use wisely)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = load_and_prepare_dataset(DATASET_PATH)
    print(f"Dataset: {len(df)} rows, V5 features added")

    weights_fn = build_weights_fn(
        decay_rate=args.decay if args.decay > 0 else 1.0,
        competition_blend=args.blend,
    )

    all_results = []

    if args.features in ("v4", "both"):
        result = run_tuning_evaluation(
            df=df,
            feature_cols=FEATURE_COLS,
            weights_fn=weights_fn,
            alpha=args.alpha,
            et_scale=args.et_scale,
            strategy_name=f"v4_features_decay{args.decay}_blend{args.blend}",
        )
        all_results.append(result)

    if args.features in ("v5", "both"):
        result = run_tuning_evaluation(
            df=df,
            feature_cols=FEATURE_COLS_V5,
            weights_fn=weights_fn,
            alpha=args.alpha,
            et_scale=args.et_scale,
            strategy_name=f"v5_features_decay{args.decay}_blend{args.blend}",
        )
        all_results.append(result)

        if args.holdout:
            # Use calibration from the v5 tuning run
            calib = result["calibration"]
            run_holdout_evaluation(
                df=df,
                feature_cols=FEATURE_COLS_V5,
                weights_fn=weights_fn,
                calib_params=calib,
                alpha=args.alpha,
                strategy_name=f"v5_features_decay{args.decay}_blend{args.blend}_HOLDOUT",
            )

    # Save results
    out_path = OUTPUT_DIR / "tuning_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved → {out_path}")

    # Summary table
    rows = []
    for r in all_results:
        for key, label in [("v4_baseline", "V4 floor"), ("v5_scale", "V5 DC scale"), ("v5_affine", "V5 DC affine")]:
            if key in r:
                m = r[key]
                rows.append({
                    "strategy": r["strategy"],
                    "conversion": label,
                    "exact_acc": round(m["exact_score_accuracy"] * 100, 1),
                    "outcome_acc": round(m["outcome_accuracy"] * 100, 1),
                    "pred_draw_rate": round(m["predicted_draw_rate"] * 100, 1),
                    "actual_draw_rate": round(m.get("actual_draw_rate", 0) * 100, 1),
                })
    summary_df = pd.DataFrame(rows)
    summary_path = OUTPUT_DIR / "tuning_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table → {summary_path}")
    print("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
