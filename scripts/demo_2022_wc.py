"""
2022 World Cup Backtest Demo
============================
Trains LightGBM on all pre-2022 data, then simulates the 2022 WC
match-by-match using the production src pipeline.

Compares four model variants:
  1. Baseline          - LightGBM trained on difference features
  2. Calibrated        - same + isotonic calibration of lambdas
  3. Symmetric         - per-team symmetric model (single model, doubled data)
  4. Sym + Calibrated  - both improvements combined

Src files used:
  src/features/feature_columns.py           FEATURE_COLS (21 features)
  src/features/tournament_state_features.py live state updated after each match
  src/models/lgbm_model.py                  LGBMGoalModel (Poisson objective)
  src/models/weighting.py                   competition-based sample weights
  src/prediction/score_conversion.py        Poisson grid -> discrete score

Run from the project root:
    python scripts/demo_2022_wc.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.features.feature_columns import FEATURE_COLS
from src.features.tournament_state_features import (
    compute_tournament_state_features,
    initialize_team_states,
    update_state_after_match,
)
from src.models.lgbm_model import LGBMGoalModel
from src.models.weighting import apply_competition_weights, COMPETITION_WEIGHTS
from src.prediction.score_conversion import convert_expected_goals_to_scores, outcome_probabilities

SEP = "=" * 100

# ---------------------------------------------------------------------------
# Hyperparameters (Optuna-found on 2022 WC — use as a ceiling estimate)
# ---------------------------------------------------------------------------
LGBM_PARAMS = dict(
    n_estimators=246,
    max_depth=12,
    learning_rate=0.02235344330741584,
    num_leaves=153,
    min_child_samples=53,
    subsample=0.7131805151434094,
    colsample_bytree=0.9003288705141621,
    reg_alpha=0.0005891440604933838,
    reg_lambda=0.0010391363783449235,
)

# ---------------------------------------------------------------------------
# Symmetric model: per-team feature definitions
#
# Why per-team?
# The current model uses *difference* features (elo_diff = elo_a - elo_b).
# This means it cannot distinguish:
#   - high-attacking vs high-defending team  (net diff = 0)
#   - moderate-attacking vs moderate-defending team  (net diff = 0)
# A symmetric model learns "goals by the team in the attacking slot" by training
# on BOTH team_a and team_b perspectives simultaneously.
# Benefits:
#   - Forces symmetry: f(team_a, team_b) is consistent with f(team_b, team_a)
#   - Doubles effective training data
#   - Separates attack from defense signal via absolute rating columns
# ---------------------------------------------------------------------------

# Difference features: negate when flipping (elo_diff for team_a is -elo_diff for team_b)
DIFF_COLS = [
    "rank_diff",
    "elo_diff",
    "avg_player_value_diff",
    "opponent_strength_diff_last5",
    "weighted_goals_for_diff_last5",
    "weighted_goals_against_diff_last5",
    "market_value_rel_mean_diff",
    "rating_change_diff_last5",
    "defender_share_diff",
    "goalkeeper_share_diff",
    "tournament_goal_diff_diff",
    "tournament_points_diff",
]

# Absolute per-team features: swap a<->b when flipping
SWAP_PAIRS = [
    ("rating_a_before", "rating_b_before"),
    ("team_a_matches_played_before", "team_b_matches_played_before"),
    ("team_a_days_since_last_match", "team_b_days_since_last_match"),
    ("team_a_tournament_matches_played", "team_b_tournament_matches_played"),
]

# log_market_value_a has no team_b counterpart in the dataset,
# so drop it from the symmetric model (market_value_rel_mean_diff still covers it)
SYMMETRIC_FEATURE_COLS = [c for c in FEATURE_COLS if c != "log_market_value_a"]


def flip_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return X from team_b's perspective (swap a<->b, negate diffs)."""
    X_flip = X.copy()
    for col in DIFF_COLS:
        if col in X_flip.columns:
            X_flip[col] = -X_flip[col]
    for col_a, col_b in SWAP_PAIRS:
        if col_a in X_flip.columns and col_b in X_flip.columns:
            tmp = X_flip[col_a].copy()
            X_flip[col_a] = X_flip[col_b]
            X_flip[col_b] = tmp
    return X_flip


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
# Lambda calibration
# ---------------------------------------------------------------------------

class LambdaCalibrator:
    """
    Maps raw model lambdas to calibrated lambdas using isotonic regression.

    Fit on a held-out calibration set (chronologically last portion of training data).
    IsotonicRegression learns a monotone mapping: raw_lambda -> actual_mean_goals.

    Why isotonic?
    - Preserves order: higher raw lambda -> higher calibrated lambda
    - Non-parametric: no assumption about the shape of miscalibration
    - Corrects systematic over-prediction at extreme lambda values (e.g. lambda_a=3 -> actual 2.2)
    """

    def __init__(self):
        self.cal_a = IsotonicRegression(increasing=True, out_of_bounds="clip")
        self.cal_b = IsotonicRegression(increasing=True, out_of_bounds="clip")
        self._fitted = False

    def fit(self, raw_preds: np.ndarray, y_actual: np.ndarray):
        """raw_preds: (n, 2), y_actual: (n, 2)"""
        self.cal_a.fit(raw_preds[:, 0], y_actual[:, 0])
        self.cal_b.fit(raw_preds[:, 1], y_actual[:, 1])
        self._fitted = True
        return self

    def transform(self, raw_preds: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Fit calibrator before transforming.")
        cal_a = self.cal_a.predict(raw_preds[:, 0])
        cal_b = self.cal_b.predict(raw_preds[:, 1])
        return np.column_stack([cal_a, cal_b])


# ---------------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------------

def train_baseline(train_df: pd.DataFrame) -> LGBMGoalModel:
    """Standard two-model LightGBM on FEATURE_COLS."""
    X = train_df[FEATURE_COLS]
    y = train_df[["target_goals_a", "target_goals_b"]]
    w = apply_competition_weights(train_df)
    model = LGBMGoalModel(**LGBM_PARAMS)
    model.fit(X, y, sample_weight=w)
    return model


def train_with_calibration(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    cal_ratio: float = 0.25,
) -> tuple:
    """
    Train model + isotonic calibrator.

    1. Chronological split: first (1-cal_ratio) for model, last cal_ratio for calibration.
    2. Fit calibrator on predictions vs actuals in calibration set.
    3. Retrain final model on ALL training data.

    Returns (final_model, calibrator).
    """
    n_cal = int(len(train_df) * cal_ratio)
    n_train = len(train_df) - n_cal

    model_train_df = train_df.iloc[:n_train]
    cal_df = train_df.iloc[n_train:]

    # Step 1: train on 75% to get predictions for calibration
    X_mt = model_train_df[feature_cols]
    y_mt = model_train_df[["target_goals_a", "target_goals_b"]]
    w_mt = apply_competition_weights(model_train_df)
    cal_model = LGBMGoalModel(**LGBM_PARAMS)
    cal_model.fit(X_mt, y_mt, sample_weight=w_mt)

    # Step 2: get calibration set predictions
    X_cal = cal_df[feature_cols]
    y_cal_actual = cal_df[["target_goals_a", "target_goals_b"]].values
    raw_cal_preds = cal_model.predict(X_cal)

    calibrator = LambdaCalibrator()
    calibrator.fit(raw_cal_preds, y_cal_actual)

    # Step 3: retrain on all training data
    X_all = train_df[feature_cols]
    y_all = train_df[["target_goals_a", "target_goals_b"]]
    w_all = apply_competition_weights(train_df)
    final_model = LGBMGoalModel(**LGBM_PARAMS)
    final_model.fit(X_all, y_all, sample_weight=w_all)

    return final_model, calibrator


class SymmetricGoalModel:
    """
    Per-team symmetric model.

    Trains a SINGLE LGBMRegressor on augmented data:
      - Original rows: SYMMETRIC_FEATURE_COLS -> goals_a
      - Flipped rows:  flip(SYMMETRIC_FEATURE_COLS) -> goals_b

    At prediction:
      - lambda_a = model.predict(X_orig)
      - lambda_b = model.predict(flip(X_orig))

    This forces symmetry: the model has no notion of "team_a vs team_b",
    only "attacking team vs defending team" based on features.
    """

    def __init__(self, **params):
        p = {**LGBM_PARAMS, **params, "objective": "poisson", "n_jobs": -1,
             "random_state": 42, "verbose": -1}
        self._lgb = lgb.LGBMRegressor(**p)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight=None):
        X_orig = X[SYMMETRIC_FEATURE_COLS]
        X_flip = flip_features(X_orig)

        goals_a = y["target_goals_a"].values if isinstance(y, pd.DataFrame) else y[:, 0]
        goals_b = y["target_goals_b"].values if isinstance(y, pd.DataFrame) else y[:, 1]

        X_aug = pd.concat([X_orig, X_flip], ignore_index=True)
        y_aug = np.concatenate([goals_a, goals_b])
        w_aug = (np.concatenate([sample_weight, sample_weight])
                 if sample_weight is not None else None)

        self._lgb.fit(X_aug, y_aug, sample_weight=w_aug)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_orig = X[SYMMETRIC_FEATURE_COLS]
        X_flip = flip_features(X_orig)
        lambda_a = np.clip(self._lgb.predict(X_orig), 0.0, None)
        lambda_b = np.clip(self._lgb.predict(X_flip), 0.0, None)
        return np.column_stack([lambda_a, lambda_b])

    def feature_importances(self, feature_names=None):
        imp = self._lgb.feature_importances_
        names = feature_names if feature_names is not None else list(range(len(imp)))
        return dict(zip(names, imp.tolist()))


def train_symmetric(train_df: pd.DataFrame) -> SymmetricGoalModel:
    X = train_df[SYMMETRIC_FEATURE_COLS]
    y = train_df[["target_goals_a", "target_goals_b"]]
    w = apply_competition_weights(train_df)
    model = SymmetricGoalModel()
    model.fit(X, y, sample_weight=w)
    return model


def train_symmetric_with_calibration(train_df: pd.DataFrame) -> tuple:
    n_cal = int(len(train_df) * 0.25)
    n_train = len(train_df) - n_cal

    model_train_df = train_df.iloc[:n_train]
    cal_df = train_df.iloc[n_train:]

    cal_model = SymmetricGoalModel()
    cal_model.fit(
        model_train_df[SYMMETRIC_FEATURE_COLS],
        model_train_df[["target_goals_a", "target_goals_b"]],
        sample_weight=apply_competition_weights(model_train_df),
    )
    raw_cal_preds = cal_model.predict(cal_df[SYMMETRIC_FEATURE_COLS])
    y_cal_actual = cal_df[["target_goals_a", "target_goals_b"]].values

    calibrator = LambdaCalibrator()
    calibrator.fit(raw_cal_preds, y_cal_actual)

    final_model = SymmetricGoalModel()
    final_model.fit(
        train_df[SYMMETRIC_FEATURE_COLS],
        train_df[["target_goals_a", "target_goals_b"]],
        sample_weight=apply_competition_weights(train_df),
    )
    return final_model, calibrator


# ---------------------------------------------------------------------------
# Tournament simulation (model-agnostic)
# ---------------------------------------------------------------------------

def run_simulation(
    model,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    calibrator: LambdaCalibrator | None = None,
) -> pd.DataFrame:
    """
    Simulate the 2022 WC match by match.

    For each match:
      1. Retrieve pre-computed features from the dataset (ELO, form, market value)
      2. Override 4 tournament-state cols with live state from tournament_state_features.py
      3. Predict (lambda_a, lambda_b) from model
      4. Apply calibration if provided
      5. Poisson grid -> discrete score (score_conversion.py)
      6. Update tournament state with actual result
    """
    team_states = initialize_team_states([])
    records = []

    for _, row in test_df.iterrows():
        team_a = row["team_a"]
        team_b = row["team_b"]

        features = row[feature_cols].copy()

        # Live tournament state
        state_feats = compute_tournament_state_features(team_a, team_b, team_states)
        for col, val in state_feats.items():
            if col in features.index:
                features[col] = val

        X = pd.DataFrame([features])
        raw_pred = model.predict(X)

        if calibrator is not None:
            pred = calibrator.transform(raw_pred)
        else:
            pred = raw_pred

        lambda_a, lambda_b = float(pred[0, 0]), float(pred[0, 1])
        predicted = convert_expected_goals_to_scores(pred)[0]
        probs = outcome_probabilities(lambda_a, lambda_b)

        actual_a = int(row["target_goals_a"])
        actual_b = int(row["target_goals_b"])
        pred_result = int(np.sign(predicted[0] - predicted[1]))
        actual_result = int(np.sign(actual_a - actual_b))

        records.append({
            "date": row["date"].date(),
            "team_a": team_a,
            "team_b": team_b,
            "pred_score": f"{predicted[0]}-{predicted[1]}",
            "actual_score": f"{actual_a}-{actual_b}",
            "lambda_a": round(lambda_a, 2),
            "lambda_b": round(lambda_b, 2),
            "p_win_a": round(probs["home_win"] * 100, 1),
            "p_draw": round(probs["draw"] * 100, 1),
            "p_win_b": round(probs["away_win"] * 100, 1),
            "exact_match": predicted[0] == actual_a and predicted[1] == actual_b,
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

def print_model_info(
    baseline: LGBMGoalModel,
    symmetric: SymmetricGoalModel,
    train_df: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> None:
    print(f"\n{SEP}")
    print("MODEL CONFIGURATION")
    print(SEP)

    print("\n--- Src files used ---")
    print("  src/features/feature_columns.py           FEATURE_COLS (21 features)")
    print("  src/features/tournament_state_features.py live state overridden per match")
    print("  src/models/lgbm_model.py                  LGBMGoalModel (Poisson obj)")
    print("  src/models/weighting.py                   apply_competition_weights()")
    print("  src/prediction/score_conversion.py        Poisson grid -> discrete score")

    print("\n--- Training data ---")
    print(f"  Total matches : {len(train_df):,}  ({train_df['date'].min().date()} to {cutoff.date()})")
    by_comp = train_df.groupby("tournament_key").size().sort_values(ascending=False)
    print("  Top competitions by match count:")
    for key, cnt in by_comp.head(8).items():
        print(f"    {key:<42} {cnt:>5}")

    print("\n--- Sample weights (src/models/weighting.py) ---")
    for comp, w in sorted(COMPETITION_WEIGHTS.items(), key=lambda x: -x[1]):
        print(f"    {comp:<40} {w:.1f}x")
    print("  Normalized to mean=1.0 across training set.")

    print("\n--- Hyperparameters (Optuna on 2022 WC) ---")
    for k, v in LGBM_PARAMS.items():
        print(f"    {k:<28} {v}")

    print("\n--- Feature importance: Baseline vs Symmetric ---")
    imp_b = baseline.feature_importances(FEATURE_COLS)
    imp_s = symmetric.feature_importances(SYMMETRIC_FEATURE_COLS)
    total_b = sum(imp_b.values())
    total_s = sum(imp_s.values())

    all_feats = sorted(set(list(imp_b.keys()) + list(imp_s.keys())))
    print(f"  {'Feature':<42} {'Baseline':>9}  {'Symmetric':>10}")
    print(f"  {'-'*42} {'-'*9}  {'-'*10}")
    for feat in sorted(all_feats, key=lambda f: -imp_b.get(f, 0)):
        b_pct = 100 * imp_b.get(feat, 0) / total_b if total_b else 0
        s_pct = 100 * imp_s.get(feat, 0) / total_s if total_s else 0
        marker = " (*)" if feat not in imp_s else ""
        print(f"  {feat:<42} {b_pct:>8.1f}%  {s_pct:>9.1f}%{marker}")
    print("  (*) feature not in symmetric model (no team_b counterpart)")

    print("\n--- Why symmetric helps ---")
    print("  Baseline uses difference features: elo_diff = elo_a - elo_b")
    print("  Problem: elo_diff=0 looks the same whether both teams are strong or both weak.")
    print("  Symmetric model sees rating_a_before AND rating_b_before separately,")
    print("  so it learns 'high-rated attacker vs high-rated defender = ~1.4 goals'")
    print("  differently from 'low vs low = ~1.1 goals'.")
    print("  It trains on both orientations of each match, doubling effective data.")

    print("\n--- How calibration works ---")
    print("  1. Hold out last 25% of training data (chronologically) as calibration set.")
    print("  2. Train model on first 75%.")
    print("  3. Predict lambdas on the 25% calibration set.")
    print("  4. Fit IsotonicRegression: raw_lambda -> actual_goals (monotone, non-parametric).")
    print("  5. Retrain final model on all 100% of training data.")
    print("  6. At test time: raw_lambda -> calibrated_lambda -> Poisson grid -> score.")
    print("  Corrects systematic over-prediction at high lambda values.")


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

    exact = results["exact_match"].mean() * 100
    result_acc = results["result_match"].mean() * 100
    print(f"\nExact score accuracy : {exact:.1f}%  ({results['exact_match'].sum()}/{len(results)})")
    print(f"Result accuracy      : {result_acc:.1f}%  ({results['result_match'].sum()}/{len(results)})")

    pred_counts = results["pred_score"].value_counts().head(8)
    actual_counts = results["actual_score"].value_counts().head(8)
    print(f"\n  {'Predicted':>12}  cnt    {'Actual':>12}  cnt")
    for (ps, pc), (as_, ac) in zip(pred_counts.items(), actual_counts.items()):
        print(f"  {ps:>12}  {pc:<6} {as_:>12}  {ac}")


def print_comparison(all_results: dict[str, pd.DataFrame]) -> None:
    print(f"\n{SEP}")
    print("MODEL COMPARISON SUMMARY")
    print(SEP)
    print(f"\n  {'Model':<30} {'Exact%':>7}  {'Result%':>8}  {'Unique scores':>14}  {'2-0 count':>10}")
    print(f"  {'-'*30} {'-'*7}  {'-'*8}  {'-'*14}  {'-'*10}")
    for label, r in all_results.items():
        exact = r["exact_match"].mean() * 100
        result_acc = r["result_match"].mean() * 100
        unique = r["pred_score"].nunique()
        two_zero = (r["pred_score"] == "2-0").sum()
        print(f"  {label:<30} {exact:>6.1f}%  {result_acc:>7.1f}%  {unique:>14}  {two_zero:>10}")


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

    print("\nTraining models...")

    print("  [1/4] Baseline LightGBM...")
    baseline = train_baseline(train_df)

    print("  [2/4] LightGBM + calibration (25% hold-out)...")
    calibrated_model, calibrator = train_with_calibration(train_df, FEATURE_COLS)

    print("  [3/4] Symmetric LightGBM (per-team)...")
    symmetric = train_symmetric(train_df)

    print("  [4/4] Symmetric + calibration...")
    sym_cal_model, sym_calibrator = train_symmetric_with_calibration(train_df)

    print("  Done.\n")

    # Print full model info
    print_model_info(baseline, symmetric, train_df, cutoff)

    # Run all simulations
    print("\nRunning simulations...")
    r_baseline = run_simulation(baseline, test_df, FEATURE_COLS)
    r_calibrated = run_simulation(calibrated_model, test_df, FEATURE_COLS, calibrator=calibrator)
    r_symmetric = run_simulation(symmetric, test_df, SYMMETRIC_FEATURE_COLS)
    r_sym_cal = run_simulation(sym_cal_model, test_df, SYMMETRIC_FEATURE_COLS, calibrator=sym_calibrator)

    all_results = {
        "Baseline": r_baseline,
        "Calibrated": r_calibrated,
        "Symmetric": r_symmetric,
        "Symmetric + Calibrated": r_sym_cal,
    }

    # Summary comparison
    print_comparison(all_results)

    # Full match table for the best model
    best_label = max(all_results, key=lambda k: all_results[k]["exact_match"].mean())
    print_results(all_results[best_label], label=f"Best model: {best_label}")

    # Save best predictions
    all_results[best_label].to_csv("data/processed/demo_2022_wc_predictions.csv", index=False)
    print(f"\nBest predictions saved to data/processed/demo_2022_wc_predictions.csv")


if __name__ == "__main__":
    main()
