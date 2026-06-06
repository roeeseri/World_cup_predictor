"""
2022 World Cup Backtest Demo
============================
Trains the production ensemble (0.8 LightGBM + 0.2 XGBoost) on all pre-2022
data, then simulates the 2022 WC match-by-match using the production pipeline.

Also compares LightGBM-only and XGBoost-only as reference.
Hyperparameters come directly from the model class defaults (lgbm_model.py,
xgb_model.py) — no hardcoded values here.

Src files used:
  src/features/feature_columns.py           FEATURE_COLS (21 features)
  src/features/tournament_state_features.py live state updated after each match
  src/models/lgbm_model.py                  LGBMGoalModel (Poisson objective)
  src/models/xgb_model.py                   XGBGoalModel  (Poisson objective)
  src/models/ensemble.py                    EnsembleGoalModel (weighted avg)
  src/models/weighting.py                   competition-based sample weights
  src/models/score_conversion.py            Poisson grid -> discrete score

Run from the project root:
    python scripts/demo_2022_wc.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.features.feature_columns import FEATURE_COLS
from src.features.tournament_state_features import (
    compute_tournament_state_features,
    initialize_team_states,
    update_state_after_match,
)
from src.evaluation.metrics import (
    exact_score_accuracy,
    goal_difference_mae,
    goal_mae_raw,
    goal_rmse_raw,
    result_accuracy,
    rounded_score_mae,
    rps_batch,
    winner_aware_error,
)
from src.models.ensemble import EnsembleGoalModel
from src.models.lgbm_model import LGBMGoalModel
from src.models.score_conversion import most_likely_score, win_draw_loss_probs
from src.models.weighting import apply_competition_weights, COMPETITION_WEIGHTS
from src.models.xgb_model import XGBGoalModel

SEP = "=" * 100

ENSEMBLE_W_LGBM = 0.8
ENSEMBLE_W_XGB  = 0.2


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def load_dataset(path: str = "data/processed/model_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def split_2022_wc(df: pd.DataFrame):
    """
    Test  = actual 64-match Qatar 2022 tournament (tournament_key == 'World Cup_2022').
    Train = everything before the tournament start (includes 2022 qualifiers).
    """
    test_mask = df["tournament_key"] == "World Cup_2022"
    cutoff = df.loc[test_mask, "date"].min()
    train_df = df[df["date"] < cutoff].copy()
    test_df = df[test_mask].sort_values("date").reset_index(drop=True)
    return train_df, test_df, cutoff


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_lgbm(train_df: pd.DataFrame) -> LGBMGoalModel:
    X = train_df[FEATURE_COLS]
    y = train_df[["target_goals_a", "target_goals_b"]]
    w = apply_competition_weights(train_df)
    model = LGBMGoalModel()
    model.fit(X, y, sample_weight=w)
    return model


def train_xgb(train_df: pd.DataFrame) -> XGBGoalModel:
    X = train_df[FEATURE_COLS]
    y = train_df[["target_goals_a", "target_goals_b"]]
    w = apply_competition_weights(train_df)
    model = XGBGoalModel()
    model.fit(X, y, sample_weight=w)
    return model


def train_ensemble(train_df: pd.DataFrame) -> EnsembleGoalModel:
    X = train_df[FEATURE_COLS]
    y = train_df[["target_goals_a", "target_goals_b"]]
    w = apply_competition_weights(train_df)
    lgbm = LGBMGoalModel()
    xgb  = XGBGoalModel()
    model = EnsembleGoalModel(
        [lgbm, xgb],
        weights=[ENSEMBLE_W_LGBM, ENSEMBLE_W_XGB],
    )
    model.fit(X, y, sample_weight=w)
    return model


# ---------------------------------------------------------------------------
# Tournament simulation (model-agnostic)
# ---------------------------------------------------------------------------

def run_simulation(model, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate the 2022 WC match by match.

    For each match:
      1. Retrieve pre-computed features from the dataset (ELO, form, market value)
      2. Override 4 tournament-state cols with live state from tournament_state_features.py
      3. Predict (lambda_a, lambda_b) from model
      4. Poisson grid -> discrete score + W/D/L probs
      5. Update tournament state with actual result
    """
    team_states = initialize_team_states([])
    records = []

    for _, row in test_df.iterrows():
        team_a = row["team_a"]
        team_b = row["team_b"]

        features = row[FEATURE_COLS].copy()

        # Override with live tournament state
        state_feats = compute_tournament_state_features(team_a, team_b, team_states)
        for col, val in state_feats.items():
            if col in features.index:
                features[col] = val

        X = pd.DataFrame([features])
        pred = np.clip(model.predict(X), 0, None)

        lambda_a, lambda_b = float(pred[0, 0]), float(pred[0, 1])
        predicted = most_likely_score(lambda_a, lambda_b)
        probs = win_draw_loss_probs(lambda_a, lambda_b)

        actual_a = int(row["target_goals_a"])
        actual_b = int(row["target_goals_b"])
        pred_result   = int(np.sign(predicted[0] - predicted[1]))
        actual_result = int(np.sign(actual_a - actual_b))

        records.append({
            "date":         row["date"].date(),
            "team_a":       team_a,
            "team_b":       team_b,
            "pred_score":   f"{predicted[0]}-{predicted[1]}",
            "actual_score": f"{actual_a}-{actual_b}",
            "lambda_a":     round(lambda_a, 2),
            "lambda_b":     round(lambda_b, 2),
            "p_win_a":      round(probs[0] * 100, 1),
            "p_draw":       round(probs[1] * 100, 1),
            "p_win_b":      round(probs[2] * 100, 1),
            "exact_match":  predicted[0] == actual_a and predicted[1] == actual_b,
            "result_match": pred_result == actual_result,
        })

        team_states = update_state_after_match(
            team_states, team_a=team_a, team_b=team_b,
            goals_a=actual_a, goals_b=actual_b,
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _arrays_from_results(results: pd.DataFrame):
    """Extract numpy arrays needed by the evaluation metrics functions."""
    def parse_score(s):
        a, b = s.split("-")
        return int(a), int(b)

    actual   = np.array([parse_score(s) for s in results["actual_score"]])
    rounded  = np.array([parse_score(s) for s in results["pred_score"]])
    expected = results[["lambda_a", "lambda_b"]].values
    probs    = results[["p_win_a", "p_draw", "p_win_b"]].values / 100.0
    return actual, expected, rounded, probs


def compute_all_metrics(results: pd.DataFrame) -> dict:
    """Compute the full set of evaluation metrics for one model's results."""
    actual, expected, rounded, probs = _arrays_from_results(results)
    return {
        # Score prediction quality
        "Exact score %":         exact_score_accuracy(actual, rounded) * 100,
        "Result accuracy %":     result_accuracy(actual, rounded) * 100,
        # Goal quantity accuracy
        "Goal MAE (expected)":   goal_mae_raw(actual, expected),
        "Goal RMSE (expected)":  goal_rmse_raw(actual, expected),
        "Rounded score MAE":     rounded_score_mae(actual, rounded),
        "Goal diff MAE":         goal_difference_mae(actual, rounded),
        # Composite / probability
        "Winner-aware error":    winner_aware_error(actual, rounded),
        "RPS (lower=better)":    rps_batch(actual, probs),
    }


def print_all_metrics(results: pd.DataFrame, label: str) -> None:
    metrics = compute_all_metrics(results)

    print(f"\n--- Metrics: {label} ---")
    print(f"  {'Metric':<26} {'Value':>10}  Note")
    print(f"  {'-'*26} {'-'*10}  {'-'*42}")

    notes = {
        "Exact score %":        "exact (goals_A AND goals_B correct)  [higher better]",
        "Result accuracy %":    "W/D/L correct                         [higher better]",
        "Goal MAE (expected)":  "mean |lambda - actual| per goal       [lower better]",
        "Goal RMSE (expected)": "root mean squared error on lambdas    [lower better]",
        "Rounded score MAE":    "mean |rounded_pred - actual| per goal [lower better]",
        "Goal diff MAE":        "mean |goal_diff_pred - actual|        [lower better]",
        "Winner-aware error":   "MAE + 0.5 penalty for wrong winner    [lower better]",
        "RPS (lower=better)":   "probabilistic W/D/L score 0-1        [lower better]",
    }
    for name, val in metrics.items():
        unit = "%" if "%" in name else ""
        print(f"  {name:<26} {val:>9.3f}{unit}  {notes[name]}")

    # Draw accuracy breakdown
    actual_draws = results["actual_score"].apply(lambda s: s.split("-")[0] == s.split("-")[1])
    pred_draws   = results["pred_score"].apply(lambda s: s.split("-")[0] == s.split("-")[1])
    n_actual_draws = actual_draws.sum()
    n_pred_draws   = pred_draws.sum()
    n_correct_draws = (actual_draws & pred_draws).sum()
    print(f"\n  Draw prediction: {n_correct_draws}/{n_actual_draws} actual draws caught"
          f"  ({n_pred_draws} draws predicted total)")


def print_model_info(train_df: pd.DataFrame, cutoff: pd.Timestamp) -> None:
    print(f"\n{SEP}")
    print("MODEL CONFIGURATION")
    print(SEP)

    lgbm_defaults = LGBMGoalModel()
    xgb_defaults  = XGBGoalModel()

    print(f"\n--- Ensemble: {ENSEMBLE_W_LGBM} x LightGBM + {ENSEMBLE_W_XGB} x XGBoost ---")

    print("\n--- LightGBM hyperparameters (from lgbm_model.py defaults) ---")
    for k, v in lgbm_defaults._params.items():
        if k not in ("objective", "n_jobs", "verbose", "random_state"):
            print(f"    {k:<28} {v}")

    print("\n--- XGBoost hyperparameters (from xgb_model.py defaults) ---")
    for k, v in xgb_defaults._params.items():
        if k not in ("objective", "tree_method", "verbosity", "n_jobs", "random_state"):
            print(f"    {k:<28} {v}")

    print("\n--- Training data ---")
    print(f"  Total matches : {len(train_df):,}  ({train_df['date'].min().date()} to {cutoff.date()})")
    by_comp = train_df.groupby("tournament_key").size().sort_values(ascending=False)
    print("  Top competitions by match count:")
    for key, cnt in by_comp.head(8).items():
        print(f"    {key:<42} {cnt:>5}")

    print("\n--- Sample weights ---")
    for comp, w in sorted(COMPETITION_WEIGHTS.items(), key=lambda x: -x[1]):
        print(f"    {comp:<40} {w:.1f}x")
    print("  Normalized to mean=1.0 across training set.")


def print_results(results: pd.DataFrame, label: str) -> None:
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 140)

    print(f"\n{SEP}")
    print(f"PREDICTIONS: {label}")
    print(SEP)

    display_cols = [
        "date", "team_a", "team_b",
        "pred_score", "actual_score",
        "lambda_a", "lambda_b",
        "p_win_a", "p_draw", "p_win_b",
        "exact_match", "result_match",
    ]
    print(results[display_cols].to_string(index=False))

    print_all_metrics(results, label)

    pred_counts   = results["pred_score"].value_counts().head(7)
    actual_counts = results["actual_score"].value_counts().head(7)
    print(f"\n     {'Predicted':>12}  cnt          {'Actual':>12}  cnt")
    for (ps, pc), (as_, ac) in zip(pred_counts.items(), actual_counts.items()):
        print(f"     {ps:>12}  {pc:<6}       {as_:>12}  {ac}")


def print_comparison(all_results: dict[str, pd.DataFrame]) -> None:
    print(f"\n{SEP}")
    print("MODEL COMPARISON SUMMARY")
    print(SEP)

    metrics_rows = {label: compute_all_metrics(r) for label, r in all_results.items()}
    metric_names = list(next(iter(metrics_rows.values())).keys())

    col_w = 32
    print(f"\n  {'Metric':<26}", end="")
    for label in all_results:
        print(f"  {label:>{col_w}}", end="")
    print()
    print(f"  {'-'*26}", end="")
    for _ in all_results:
        print(f"  {'-'*col_w}", end="")
    print()

    higher_better = {"Exact score %", "Result accuracy %"}
    for name in metric_names:
        vals = [metrics_rows[label][name] for label in all_results]
        best_idx = vals.index(max(vals) if name in higher_better else min(vals))
        print(f"  {name:<26}", end="")
        for i, (label, val) in enumerate(zip(all_results, vals)):
            unit = "%" if "%" in name else ""
            marker = " *" if i == best_idx else "  "
            print(f"  {val:>{col_w-3}.3f}{unit}{marker}", end="")
        print()

    print("\n  * = best value for that metric")

    print(f"\n  {'Score diversity':<26}", end="")
    for label, r in all_results.items():
        unique = r["pred_score"].nunique()
        two_zero = (r["pred_score"] == "2-0").sum()
        print(f"  {f'{unique} unique, {two_zero}x 2-0':>{col_w}}", end="")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading dataset...")
    df = load_dataset()
    print(f"  {len(df):,} total matches")

    print("Splitting train / 2022 WC test...")
    train_df, test_df, cutoff = split_2022_wc(df)
    print(f"  Train: {len(train_df):,} matches  (up to {cutoff.date()})")
    print(f"  Test : {len(test_df)} matches  (2022 WC, Nov 20 - Dec 18)")

    print_model_info(train_df, cutoff)

    print("\nTraining models...")
    print("  [1/3] LightGBM...")
    lgbm_model = train_lgbm(train_df)

    print("  [2/3] XGBoost...")
    xgb_model = train_xgb(train_df)

    print("  [3/3] Ensemble (0.8 LGB + 0.2 XGB)...")
    ensemble = train_ensemble(train_df)
    print("  Done.\n")

    print("Running simulations...")
    r_lgbm     = run_simulation(lgbm_model, test_df)
    r_xgb      = run_simulation(xgb_model, test_df)
    r_ensemble = run_simulation(ensemble, test_df)

    all_results = {
        "LightGBM":              r_lgbm,
        "XGBoost":               r_xgb,
        f"Ensemble ({ENSEMBLE_W_LGBM} LGB + {ENSEMBLE_W_XGB} XGB)": r_ensemble,
    }

    print_comparison(all_results)

    best_label = max(all_results, key=lambda k: all_results[k]["exact_match"].mean())
    print_results(all_results[best_label], label=f"Best model: {best_label}")

    r_ensemble.to_csv("data/processed/demo_2022_wc_predictions.csv", index=False)
    print(f"\nEnsemble predictions saved to data/processed/demo_2022_wc_predictions.csv")


if __name__ == "__main__":
    main()
