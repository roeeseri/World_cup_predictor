"""
V6 isolated improvement experiments — orchestrator.

Stages (run in order; each caches its outputs so reruns are cheap):

  python scripts/v6_experiments.py prepare
      Build the V6 feature dataset → outputs/experiments/v6/dataset_v6.parquet

  python scripts/v6_experiments.py oof [F_V5 F_FORM ...]
      Generate OOF lambdas on the 4 tuning folds (WC 2022 untouched)
      → outputs/experiments/v6/oof_<fset>.csv

  python scripts/v6_experiments.py sweep
      Decision-rule/calibration sweeps on all cached OOF lambdas
      → outputs/experiments/v6/results_folds.csv

  python scripts/v6_experiments.py holdout <fset:rule> [...]
      ONE-SHOT WC 2022 evaluation for chosen variants + V4/V5 baselines
      → outputs/experiments/v6/results_holdout.csv + matchbymatch_<name>.csv

V4/V5 production artifacts are never touched. Models are trained fresh per fold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.protocol import (
    HOLDOUT,
    TUNING_FOLDS,
    create_holdout_split,
    generate_oof_lambdas,
    load_and_prepare_dataset,
)
from src.experiments.v6_eval import (
    TARGET_COLS,
    crossfit_grid_params,
    empirical_score_prior,
    fit_grid_params,
    full_metrics,
    grid_pick,
    knockout_mask,
    rule_cond_floor,
    rule_draw_band,
    rule_floor01,
    score_lambda_frame,
)
from src.experiments.v6_features import add_v6_features, get_v6_feature_sets
from src.experiments.v6_models import build_v6_ensemble
from src.models.weighting import apply_combined_weighting

OUT_DIR = ROOT / "outputs" / "experiments" / "v6"
DATASET_CACHE = OUT_DIR / "dataset_v6.parquet"

META_COLS = [
    "date", "team_A", "team_B", "competition", "tournament_year",
    "goals_A", "goals_B",
    "team_a_tournament_matches_played", "team_b_tournament_matches_played",
    "rank_diff", "elo_diff",
]

# Production V5 weighting (decay 0.9 / blend 0.5, reference 2026)
def weights_fn(train_df: pd.DataFrame) -> np.ndarray:
    return apply_combined_weighting(
        train_df,
        apply_decay=True,
        decay_rate=0.9,
        reference_year=2026,
        competition_weight=0.5,
        temporal_weight=0.5,
    )


def load_v6_dataset() -> pd.DataFrame:
    if DATASET_CACHE.exists():
        return pd.read_parquet(DATASET_CACHE)
    print("Building V6 dataset (load + V5 features + V6 features)...")
    df = load_and_prepare_dataset()
    df = add_v6_features(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_CACHE)
    print(f"Cached → {DATASET_CACHE}  ({len(df)} rows)")
    return df


# ── Stage: oof ─────────────────────────────────────────────────────────────────
def stage_oof(fset_names: list[str]) -> None:
    df = load_v6_dataset()
    fsets = get_v6_feature_sets()

    for name in fset_names:
        cache = OUT_DIR / f"oof_{name}.csv"
        if cache.exists():
            print(f"[{name}] cached — skip ({cache})")
            continue
        cols = fsets[name]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"[{name}] missing columns: {missing}")
        print(f"\n[{name}] generating OOF lambdas ({len(cols)} features)...")
        oof = generate_oof_lambdas(
            df=df,
            model_factory=build_v6_ensemble,
            feature_cols=cols,
            weights_fn=weights_fn,
            folds=TUNING_FOLDS,
            holdout=HOLDOUT,
        )
        keep = [c for c in META_COLS if c in oof.columns] + [
            "pred_lambda_a", "pred_lambda_b", "fold_competition", "fold_year",
        ]
        oof[keep].to_csv(cache, index=False)
        print(f"[{name}] cached → {cache}")


# ── Stage: sweep ───────────────────────────────────────────────────────────────
def stage_sweep() -> None:
    fsets = get_v6_feature_sets()
    rows = []

    for name in fsets:
        cache = OUT_DIR / f"oof_{name}.csv"
        if not cache.exists():
            print(f"[{name}] no OOF cache — skip")
            continue
        oof = pd.read_csv(cache)
        y_true = oof[TARGET_COLS].values
        ko = knockout_mask(oof)

        def record(rule_name: str, preds: np.ndarray) -> None:
            m = full_metrics(y_true, preds, ko_mask=ko)
            m["fset"] = name
            m["rule"] = rule_name
            rows.append(m)
            print(f"  {name:10s} {rule_name:28s} exact={m['exact_%']:.2f}%  "
                  f"outcome={m['outcome_%']:.2f}%  draws={m['pred_draw_%']:.1f}%  "
                  f"btts={m['pred_btts_%']:.1f}%")

        # 1) simple floor rules
        record("floor01", score_lambda_frame(oof, rule_floor01))
        for tb in (0.4, 0.5, 0.6):
            record(f"cond_floor_tb{tb}", score_lambda_frame(oof, rule_cond_floor, tb=tb))

        # 2) DC grid with cross-fitted scale_c + rho
        cf = crossfit_grid_params(oof)

        # empirical prior per fold from OTHER folds' actual major-tournament scores
        priors = {}
        for key in cf:
            comp, year = key.rsplit(" ", 1)
            other = oof[~((oof["fold_competition"] == comp) & (oof["fold_year"] == int(year)))]
            priors[key] = empirical_score_prior(other["goals_A"].values, other["goals_B"].values)

        for draw_beta in (0.0, 0.01, 0.02, 0.03):
            for prior_w in (0.0, 0.15, 0.3):
                for use_ko in (False, True):
                    preds = np.zeros((len(oof), 2), dtype=int)
                    for i, (_, r) in enumerate(oof.iterrows()):
                        key = f"{r['fold_competition']} {r['fold_year']}"
                        preds[i] = grid_pick(
                            r["pred_lambda_a"], r["pred_lambda_b"],
                            rho=cf[key]["rho"], scale_c=cf[key]["scale_c"],
                            draw_beta=draw_beta,
                            prior=priors[key] if prior_w > 0 else None,
                            prior_w=prior_w,
                            is_ko=bool(ko[i]) and use_ko,
                        )
                    tag = f"dc_b{draw_beta}_p{prior_w}" + ("_ko" if use_ko else "")
                    record(tag, preds)

    res = pd.DataFrame(rows)
    out = OUT_DIR / "results_folds.csv"
    res.to_csv(out, index=False)
    print(f"\nFold sweep results → {out}  ({len(res)} configs)")

    # leaderboard
    cols = ["fset", "rule", "exact_%", "outcome_%", "gd_acc_%", "pred_draw_%",
            "actual_draw_%", "pred_btts_%", "actual_btts_%"]
    top = res.sort_values("exact_%", ascending=False).head(15)
    print("\nTop 15 by exact accuracy (4 tuning folds, n=211):")
    print(top[cols].to_string(index=False))


# ── Stage: holdout ─────────────────────────────────────────────────────────────
def _holdout_lambdas(df: pd.DataFrame, cols: list[str], model_factory) -> tuple[pd.DataFrame, np.ndarray]:
    """Train on all-except-WC2022 (fresh), predict holdout lambdas."""
    train_df, hold_df = create_holdout_split(df)
    X_tr = train_df[cols].fillna(0)
    y_tr = train_df[TARGET_COLS].values
    w = weights_fn(train_df)
    model = model_factory()
    model.fit(X_tr, y_tr, sample_weight=w)
    preds = np.clip(model.predict(hold_df[cols].fillna(0)), 0.0, None)
    return hold_df.reset_index(drop=True), preds


def stage_holdout(variant_specs: list[str]) -> None:
    """
    variant_specs: list of "<fset>:<rule>" where rule is one of
      floor01 | cond_floor_tb<0.X> | dc_b<beta>_p<w>[_ko]
    Baselines V4_prod and V5_prod are always evaluated alongside.
    """
    df = load_v6_dataset()
    fsets = get_v6_feature_sets()
    results = []

    def evaluate(name: str, hold_df: pd.DataFrame, lam: np.ndarray, preds: np.ndarray) -> None:
        y_true = hold_df[TARGET_COLS].values
        ko = knockout_mask(hold_df)
        m = full_metrics(y_true, preds, ko_mask=ko)
        m["variant"] = name
        results.append(m)
        mb = hold_df[["date", "team_A", "team_B", "goals_A", "goals_B"]].copy()
        mb["lambda_a"] = lam[:, 0].round(3)
        mb["lambda_b"] = lam[:, 1].round(3)
        mb["pred_a"] = preds[:, 0]
        mb["pred_b"] = preds[:, 1]
        mb["exact"] = (mb["goals_A"] == mb["pred_a"]) & (mb["goals_B"] == mb["pred_b"])
        safe = name.replace(":", "_").replace(".", "")
        mb.to_csv(OUT_DIR / f"matchbymatch_{safe}.csv", index=False)
        print(f"  {name:34s} exact={m['exact_%']:.1f}%  outcome={m['outcome_%']:.1f}%  "
              f"gd={m['gd_acc_%']:.1f}%  draws={m['pred_draw_%']:.0f}%/{m['actual_draw_%']:.0f}%  "
              f"btts={m['pred_btts_%']:.0f}%/{m['actual_btts_%']:.0f}%")

    # ── Baselines with PRODUCTION model classes (prod mirror, prod features) ──
    from src.models.lgbm_model import LGBMGoalModel
    from src.models.xgb_model import XGBGoalModel
    from src.models.ensemble import EnsembleGoalModel
    from src.features.feature_columns import FEATURE_COLS, FEATURE_COLS_V5_PROD

    def prod_ensemble():
        return EnsembleGoalModel([LGBMGoalModel(), XGBGoalModel()], weights=[0.9, 0.1])

    print("\nBaseline V4 (prod features + floor01)...")
    hold_df, lam = _holdout_lambdas(df, FEATURE_COLS, prod_ensemble)
    preds = np.array([rule_floor01(a, b) for a, b in lam])
    evaluate("V4_baseline", hold_df, lam, preds)

    print("Baseline V5 (prod features + cond_floor 0.5, prod mirror)...")
    hold_df, lam = _holdout_lambdas(df, FEATURE_COLS_V5_PROD, prod_ensemble)
    preds = np.array([rule_cond_floor(a, b, tb=0.5) for a, b in lam])
    evaluate("V5_baseline", hold_df, lam, preds)

    # ── V6 variants ───────────────────────────────────────────────────────────
    lam_cache: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}

    for spec in variant_specs:
        fset, rule = spec.split(":", 1)
        if fset not in lam_cache:
            print(f"\n[{fset}] training holdout model...")
            lam_cache[fset] = _holdout_lambdas(df, fsets[fset], build_v6_ensemble)
        hold_df, lam = lam_cache[fset]

        if rule == "floor01":
            preds = np.array([rule_floor01(a, b) for a, b in lam])
        elif rule.startswith("cond_floor_tb"):
            tb = float(rule.replace("cond_floor_tb", ""))
            preds = np.array([rule_cond_floor(a, b, tb=tb) for a, b in lam])
        elif rule.startswith("drawband_t"):
            # drawband_t<threshold>_tb<tb>; grid params frozen from 4-fold OOF
            body = rule.replace("drawband_t", "")
            t_str, tb_str = body.split("_tb")
            oof = pd.read_csv(OUT_DIR / f"oof_{fset}.csv")
            gp = fit_grid_params(oof)
            preds = np.array([
                rule_draw_band(a, b, rho=gp["rho"], scale_c=gp["scale_c"],
                               draw_threshold=float(t_str), tb=float(tb_str))
                for a, b in lam
            ])
        elif rule.startswith("dc_"):
            # grid params frozen from this fset's 4-fold OOF (never from holdout)
            oof = pd.read_csv(OUT_DIR / f"oof_{fset}.csv")
            gp = fit_grid_params(oof)
            parts = rule.split("_")
            draw_beta = float(parts[1][1:])
            prior_w = float(parts[2][1:])
            use_ko = rule.endswith("_ko")
            prior = (empirical_score_prior(oof["goals_A"].values, oof["goals_B"].values)
                     if prior_w > 0 else None)
            ko = knockout_mask(hold_df)
            preds = np.array([
                grid_pick(a, b, rho=gp["rho"], scale_c=gp["scale_c"],
                          draw_beta=draw_beta, prior=prior, prior_w=prior_w,
                          is_ko=bool(k) and use_ko)
                for (a, b), k in zip(lam, ko)
            ])
        else:
            raise ValueError(f"Unknown rule: {rule}")

        evaluate(spec, hold_df, lam, preds)

    res = pd.DataFrame(results)
    out = OUT_DIR / "results_holdout.csv"
    res.to_csv(out, index=False)
    print(f"\nHoldout results → {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    args = sys.argv[2:]

    if stage == "prepare":
        load_v6_dataset()
    elif stage == "oof":
        names = args if args else list(get_v6_feature_sets().keys())
        stage_oof(names)
    elif stage == "sweep":
        stage_sweep()
    elif stage == "holdout":
        if not args:
            raise SystemExit("holdout requires variant specs like F_ELOUPD:cond_floor_tb0.5")
        stage_holdout(args)
    else:
        raise SystemExit(f"Unknown stage: {stage}")
