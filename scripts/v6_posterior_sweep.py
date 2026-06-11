"""
Tune the Poisson-Gamma posterior blend (k) on cached OOF lambdas (folds only).
Joins raw tournament counts onto OOF rows, blends lambdas, applies decision rules.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.v6_eval import (
    TARGET_COLS, crossfit_grid_params, full_metrics, knockout_mask,
    rule_cond_floor, rule_draw_band,
)
from src.experiments.v6_features import add_raw_tournament_counts, posterior_blend_lambda

df = pd.read_parquet("outputs/experiments/v6/dataset_v6.parquet")
if "t_m_a" not in df.columns:
    print("adding raw tournament counts...")
    df = add_raw_tournament_counts(df)
    df.to_parquet("outputs/experiments/v6/dataset_v6.parquet")

count_cols = ["t_m_a", "t_gf_a", "t_ga_a", "t_m_b", "t_gf_b", "t_ga_b"]
key_cols = ["date", "team_A", "team_B"]
counts = df[key_cols + count_cols].copy()
counts["date"] = pd.to_datetime(counts["date"]).astype(str)

rows = []
for fset in ["F_V5", "F_ELOUPD"]:
    oof = pd.read_csv(f"outputs/experiments/v6/oof_{fset}.csv")
    oof["date"] = pd.to_datetime(oof["date"]).astype(str)
    oof = oof.merge(counts, on=key_cols, how="left", validate="one_to_one")
    assert oof[count_cols].notna().all().all(), "count join failed"

    y_true = oof[TARGET_COLS].values
    ko = knockout_mask(oof)
    cf = crossfit_grid_params(oof)

    for k in (2.0, 3.0, 4.0, 6.0, 8.0, 1e9):
        la_p = np.array([posterior_blend_lambda(r.pred_lambda_a, r.t_gf_a, r.t_m_a, k)
                         for r in oof.itertuples()])
        lb_p = np.array([posterior_blend_lambda(r.pred_lambda_b, r.t_gf_b, r.t_m_b, k)
                         for r in oof.itertuples()])

        for rule_name in ("cond05", "drawband"):
            preds = np.zeros((len(oof), 2), dtype=int)
            for i, (_, r) in enumerate(oof.iterrows()):
                key = f"{r['fold_competition']} {r['fold_year']}"
                if rule_name == "cond05":
                    preds[i] = rule_cond_floor(la_p[i], lb_p[i], tb=0.5)
                else:
                    preds[i] = rule_draw_band(la_p[i], lb_p[i],
                                              rho=cf[key]["rho"], scale_c=cf[key]["scale_c"],
                                              draw_threshold=0.33, tb=0.5)
            m = full_metrics(y_true, preds, ko_mask=ko)
            ktag = "inf" if k > 1e8 else k
            m["fset"], m["rule"], m["k"] = fset, rule_name, ktag
            rows.append(m)
            print(f"{fset:9s} k={str(ktag):4s} {rule_name:8s}  exact={m['exact_%']:.2f}%  "
                  f"outcome={m['outcome_%']:.2f}%  gd={m['gd_acc_%']:.1f}%  "
                  f"draws={m['pred_draw_%']:.1f}%  btts={m['pred_btts_%']:.1f}%  "
                  f"ko_exact={m.get('ko_exact_%', float('nan')):.1f}%")

pd.DataFrame(rows).to_csv("outputs/experiments/v6/results_folds_posterior.csv", index=False)
print("\nsaved -> outputs/experiments/v6/results_folds_posterior.csv")

