"""ONE-SHOT WC 2022 holdout for the final V6 candidate:
F_V5 features (corrected mirror) + drawband(t=0.33, tb=0.5) + KO-gated posterior blend (m_min=3, k=8).
All params were tuned on the 4 OOF folds only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.evaluation.protocol import create_holdout_split
from src.experiments.v6_eval import (
    TARGET_COLS, fit_grid_params, full_metrics, knockout_mask, rule_draw_band,
)
from src.experiments.v6_features import get_v6_feature_sets, posterior_blend_lambda
from src.experiments.v6_models import build_v6_ensemble
from v6_experiments import load_v6_dataset, weights_fn

K_BLEND = 8.0
M_MIN = 3
DRAW_T = 0.33
TB = 0.5

df = load_v6_dataset()
cols = get_v6_feature_sets()["F_V5"]

train_df, hold_df = create_holdout_split(df)
model = build_v6_ensemble()
model.fit(train_df[cols].fillna(0), train_df[TARGET_COLS].values,
          sample_weight=weights_fn(train_df))
lam = np.clip(model.predict(hold_df[cols].fillna(0)), 0.0, None)
hold_df = hold_df.reset_index(drop=True)

# Frozen grid params from the F_V5 4-fold OOF (never from holdout)
oof = pd.read_csv("outputs/experiments/v6/oof_F_V5.csv")
gp = fit_grid_params(oof)
print(f"grid params (frozen from folds): scale_c={gp['scale_c']:.4f}  rho={gp['rho']:.4f}")

preds = np.zeros((len(hold_df), 2), dtype=int)
lam_eff = lam.copy()
for i, r in enumerate(hold_df.itertuples()):
    la, lb = lam[i]
    if r.t_m_a >= M_MIN:
        la = posterior_blend_lambda(la, r.t_gf_a, r.t_m_a, K_BLEND)
    if r.t_m_b >= M_MIN:
        lb = posterior_blend_lambda(lb, r.t_gf_b, r.t_m_b, K_BLEND)
    lam_eff[i] = (la, lb)
    preds[i] = rule_draw_band(la, lb, rho=gp["rho"], scale_c=gp["scale_c"],
                              draw_threshold=DRAW_T, tb=TB)

y_true = hold_df[TARGET_COLS].values
ko = knockout_mask(hold_df)
m = full_metrics(y_true, preds, ko_mask=ko)
print("\nWC 2022 HOLDOUT â€” V6 final (drawband + KO-gated posterior blend k=8)")
for key in ("exact_%", "outcome_%", "gd_acc_%", "team_goals_mae", "total_goals_mae",
            "pred_draw_%", "actual_draw_%", "pred_btts_%", "actual_btts_%",
            "group_exact_%", "ko_exact_%", "group_outcome_%", "ko_outcome_%",
            "ko_pred_draw_%", "ko_actual_draw_%"):
    v = m.get(key)
    print(f"  {key:18s} {v:.2f}" if isinstance(v, float) else f"  {key:18s} {v}")
print("  top_pred_scores  ", m["top_pred_scores"])

mb = hold_df[["date", "team_A", "team_B", "goals_A", "goals_B"]].copy()
mb["lambda_a"] = lam[:, 0].round(3)
mb["lambda_b"] = lam[:, 1].round(3)
mb["lambda_a_eff"] = lam_eff[:, 0].round(3)
mb["lambda_b_eff"] = lam_eff[:, 1].round(3)
mb["pred_a"] = preds[:, 0]
mb["pred_b"] = preds[:, 1]
mb["exact"] = (mb["goals_A"] == mb["pred_a"]) & (mb["goals_B"] == mb["pred_b"])
mb.to_csv("outputs/experiments/v6/matchbymatch_V6_final.csv", index=False)
print("\nmatch-by-match -> outputs/experiments/v6/matchbymatch_V6_final.csv")
print(f"\nExact: {int(m['exact_%']*len(hold_df)/100)}/{len(hold_df)}")

