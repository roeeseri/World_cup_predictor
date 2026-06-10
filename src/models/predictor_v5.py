"""
ScorePredictorV5: bundled ensemble + calibration params + decision policy.

Produces the same output dict shape as predict_match_from_features() (v4),
so it can be used as a drop-in replacement by any consumer that checks for it.

V4 path stays untouched. All v5 logic lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.score_grid import (
    dixon_coles_grid,
    knockout_grid,
    make_score_fn,
    pick_score,
    win_draw_loss_from_grid,
    apply_lambda_scale,
    apply_lambda_affine,
)
from src.features.feature_columns import FEATURE_COLS_V5


class ScorePredictorV5:
    """
    V5 predictor: DC grid + calibrated lambdas + optional tournament blending.

    Parameters
    ----------
    model            : fitted goal model (EnsembleGoalModel / LGBMGoalModel / ...)
    feature_cols     : list of feature column names (default FEATURE_COLS_V5)
    rho              : Dixon-Coles correlation param (negative → more draws)
    scale_c          : multiplicative lambda scale (fit on OOF lambdas)
    affine_a/b       : log-affine calibration params; if provided, take precedence over scale_c
    alpha            : outcome-bonus weight in pick_score (0 = pure exact-score mode)
    et_scale         : fraction of ET minutes / 90 (default 30/90 for 30 ET minutes)
    blend_weight     : w in λ_final = (1-w)*λ_model + w*obs_rate when matches_played >= 2;
                       0 disables tournament blending
    competition_importance : override for competition_importance feature at inference time;
                       None = read from feature_row if present, else default 4.0 (WC)
    """

    def __init__(
        self,
        model,
        feature_cols: list[str] | None = None,
        rho: float = 0.0,
        scale_c: float = 1.0,
        affine_a: float | None = None,
        affine_b: float | None = None,
        alpha: float = 0.0,
        et_scale: float = 30.0 / 90.0,
        blend_weight: float = 0.0,
        competition_importance: float | None = None,
    ) -> None:
        self.model = model
        self.feature_cols = feature_cols if feature_cols is not None else FEATURE_COLS_V5
        self.rho = rho
        self.scale_c = scale_c
        self.affine_a = affine_a
        self.affine_b = affine_b
        self.alpha = alpha
        self.et_scale = et_scale
        self.blend_weight = blend_weight
        self.competition_importance = competition_importance

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _calibrate(self, lambda_a: float, lambda_b: float) -> tuple[float, float]:
        la = np.array([lambda_a])
        lb = np.array([lambda_b])
        if self.affine_a is not None and self.affine_b is not None:
            la_c, lb_c = apply_lambda_affine(la, lb, self.affine_a, self.affine_b)
        else:
            la_c, lb_c = apply_lambda_scale(la, lb, self.scale_c)
        return float(la_c[0]), float(lb_c[0])

    def _blend_with_tournament(
        self,
        lambda_a: float,
        lambda_b: float,
        team_a: str,
        team_b: str,
        team_states: dict,
    ) -> tuple[float, float]:
        """
        Blend λ_model with observed tournament scoring rate when matches_played >= 2.

        λ_final = (1 - w) * λ_model + w * obs_rate

        w = self.blend_weight (0 disables blending)
        """
        if self.blend_weight <= 0:
            return lambda_a, lambda_b

        def _blend(lam: float, state: dict) -> float:
            if state["matches"] < 2:
                return lam
            obs = state["goals_for"] / state["matches"]
            return (1 - self.blend_weight) * lam + self.blend_weight * obs

        sa = team_states.get(team_a, {"matches": 0, "goals_for": 0, "goals_against": 0})
        sb = team_states.get(team_b, {"matches": 0, "goals_for": 0, "goals_against": 0})
        return _blend(lambda_a, sa), _blend(lambda_b, sb)

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict_lambdas(
        self,
        feature_row: pd.DataFrame,
        team_a: str | None = None,
        team_b: str | None = None,
        team_states: dict | None = None,
    ) -> tuple[float, float]:
        """
        Raw model prediction → calibrated lambdas, optionally tournament-blended.

        Args:
            feature_row:  single-row DataFrame with FEATURE_COLS_V5 features
            team_a/b:     team names (required if team_states provided)
            team_states:  live tournament state dict
        """
        X = feature_row[self.feature_cols].fillna(0)
        pred = np.clip(self.model.predict(X), 0.0, None)
        la_raw, lb_raw = float(pred[0, 0]), float(pred[0, 1])

        la, lb = self._calibrate(la_raw, lb_raw)

        if team_states is not None and team_a is not None and team_b is not None:
            la, lb = self._blend_with_tournament(la, lb, team_a, team_b, team_states)

        return la, lb

    def predict(
        self,
        feature_row: pd.DataFrame,
        knockout: bool = False,
        team_a: str | None = None,
        team_b: str | None = None,
        team_states: dict | None = None,
    ) -> dict[str, Any]:
        """
        Full prediction: lambdas → DC grid → score + probs.

        Returns the same dict shape as predict_match_from_features():
            lambda_a, lambda_b, win_prob, draw_prob, loss_prob,
            most_likely_score (tuple), top_scores (list of tuples)

        Args:
            feature_row:  single-row DataFrame with V5 feature columns
            knockout:     True → apply ET mixture (for knockout rounds)
            team_a/b:     team names for tournament blending (optional)
            team_states:  live tournament state dict (optional)
        """
        lambda_a, lambda_b = self.predict_lambdas(feature_row, team_a, team_b, team_states)

        if knockout:
            grid = knockout_grid(lambda_a, lambda_b, rho=self.rho, et_scale=self.et_scale)
        else:
            grid = dixon_coles_grid(lambda_a, lambda_b, rho=self.rho)

        best_score = pick_score(grid, alpha=self.alpha)
        win_prob, draw_prob, loss_prob = win_draw_loss_from_grid(grid)

        # Top 5 scores
        flat_indices = np.argsort(grid, axis=None)[::-1][:5]
        top_scores = [
            (int(r), int(c), float(grid[r, c]))
            for r, c in zip(*np.unravel_index(flat_indices, grid.shape))
        ]

        return {
            "lambda_a": lambda_a,
            "lambda_b": lambda_b,
            "win_prob": win_prob,
            "draw_prob": draw_prob,
            "loss_prob": loss_prob,
            "most_likely_score": best_score,
            "top_scores": top_scores,
        }

    def config(self) -> dict:
        """Serialisable config dict for saving alongside the model artifact."""
        return {
            "version": "v5",
            "feature_cols": self.feature_cols,
            "rho": self.rho,
            "scale_c": self.scale_c,
            "affine_a": self.affine_a,
            "affine_b": self.affine_b,
            "alpha": self.alpha,
            "et_scale": self.et_scale,
            "blend_weight": self.blend_weight,
            "competition_importance": self.competition_importance,
        }

    @classmethod
    def from_config(cls, model, config: dict) -> "ScorePredictorV5":
        """Reconstruct a ScorePredictorV5 from a saved config dict."""
        return cls(
            model=model,
            feature_cols=config.get("feature_cols"),
            rho=config.get("rho", 0.0),
            scale_c=config.get("scale_c", 1.0),
            affine_a=config.get("affine_a"),
            affine_b=config.get("affine_b"),
            alpha=config.get("alpha", 0.0),
            et_scale=config.get("et_scale", 30.0 / 90.0),
            blend_weight=config.get("blend_weight", 0.0),
            competition_importance=config.get("competition_importance"),
        )
