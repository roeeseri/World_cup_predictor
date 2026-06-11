"""Sweep the draw-band hybrid rule on cached OOF lambdas (folds only)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.v6_eval import (
    TARGET_COLS, crossfit_grid_params, full_metrics, knockout_mask, rule_draw_band,
)

rows = []
for fset in ["F_V5", "F_CAP", "F_ELOUPD", "F_FORM"]:
    oof = pd.read_csv(f"outputs/experiments/v6/oof_{fset}.csv")
    y_true = oof[TARGET_COLS].values
    ko = knockout_mask(oof)
    cf = crossfit_grid_params(oof)

    for t in (0.26, 0.28, 0.30, 0.33):
        for tb in (0.5, 0.6):
            preds = np.zeros((len(oof), 2), dtype=int)
            for i, (_, r) in enumerate(oof.iterrows()):
                key = f"{r['fold_competition']} {r['fold_year']}"
                preds[i] = rule_draw_band(
                    r["pred_lambda_a"], r["pred_lambda_b"],
                    rho=cf[key]["rho"], scale_c=cf[key]["scale_c"],
                    draw_threshold=t, tb=tb,
                )
            m = full_metrics(y_true, preds, ko_mask=ko)
            m["fset"] = fset
            m["rule"] = f"drawband_t{t}_tb{tb}"
            rows.append(m)
            print(f"{fset:9s} t={t} tb={tb}  exact={m['exact_%']:.2f}%  outcome={m['outcome_%']:.2f}%  "
                  f"gd={m['gd_acc_%']:.1f}%  draws={m['pred_draw_%']:.1f}% (act {m['actual_draw_%']:.1f}%)  "
                  f"btts={m['pred_btts_%']:.1f}% (act {m['actual_btts_%']:.1f}%)")

res = pd.DataFrame(rows)
res.to_csv("outputs/experiments/v6/results_folds_drawband.csv", index=False)
print("\nsaved -> outputs/experiments/v6/results_folds_drawband.csv")

