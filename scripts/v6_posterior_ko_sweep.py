"""Posterior blend applied ONLY when a team has >= m_min tournament matches (KO gate)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.v6_eval import (
    TARGET_COLS, crossfit_grid_params, full_metrics, knockout_mask, rule_draw_band,
)
from src.experiments.v6_features import posterior_blend_lambda

df = pd.read_parquet("outputs/experiments/v6/dataset_v6.parquet")
count_cols = ["t_m_a", "t_gf_a", "t_ga_a", "t_m_b", "t_gf_b", "t_ga_b"]
key_cols = ["date", "team_A", "team_B"]
counts = df[key_cols + count_cols].copy()
counts["date"] = pd.to_datetime(counts["date"]).astype(str)

for fset in ["F_V5"]:
    oof = pd.read_csv(f"outputs/experiments/v6/oof_{fset}.csv")
    oof["date"] = pd.to_datetime(oof["date"]).astype(str)
    oof = oof.merge(counts, on=key_cols, how="left", validate="one_to_one")

    y_true = oof[TARGET_COLS].values
    ko = knockout_mask(oof)
    cf = crossfit_grid_params(oof)

    for m_min in (2, 3):
        for k in (4.0, 6.0, 8.0, 10.0):
            preds = np.zeros((len(oof), 2), dtype=int)
            for i, (_, r) in enumerate(oof.iterrows()):
                la = (posterior_blend_lambda(r.pred_lambda_a, r.t_gf_a, r.t_m_a, k)
                      if r.t_m_a >= m_min else r.pred_lambda_a)
                lb = (posterior_blend_lambda(r.pred_lambda_b, r.t_gf_b, r.t_m_b, k)
                      if r.t_m_b >= m_min else r.pred_lambda_b)
                key = f"{r['fold_competition']} {r['fold_year']}"
                preds[i] = rule_draw_band(la, lb, rho=cf[key]["rho"], scale_c=cf[key]["scale_c"],
                                          draw_threshold=0.33, tb=0.5)
            m = full_metrics(y_true, preds, ko_mask=ko)
            print(f"{fset} m_min={m_min} k={k:5.1f} drawband  exact={m['exact_%']:.2f}%  "
                  f"outcome={m['outcome_%']:.2f}%  gd={m['gd_acc_%']:.1f}%  "
                  f"draws={m['pred_draw_%']:.1f}%  btts={m['pred_btts_%']:.1f}%  "
                  f"grp_exact={m.get('group_exact_%', float('nan')):.1f}%  ko_exact={m.get('ko_exact_%', float('nan')):.1f}%")

