"""V5 evaluation protocol: leak-free fold splits, V5 feature engineering, OOF lambdas, metrics, holdout logging."""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.models.base import load_model_dataset
from src.models.weighting import apply_combined_weighting

# ── Fold / holdout constants ───────────────────────────────────────────────────
DATASET_PATH = "data/processed/updated_model_dataset.csv"
TARGET_COLS = ["goals_A", "goals_B"]

# Four tuning folds — WC 2022 never appears in any training set here
TUNING_FOLDS: list[tuple[str, int]] = [
    ("World Cup", 2014),
    ("World Cup", 2018),
    ("European Championship", 2024),
    ("Copa America", 2024),
]
HOLDOUT: tuple[str, int] = ("World Cup", 2022)

# Tournament context features (in-tournament goals rates) are non-zero only here
MAJOR_TOURNAMENTS: set[str] = {"World Cup", "European Championship", "Copa America"}

HOLDOUT_LOG_PATH = "outputs/evaluation/v5/holdout_log.json"

# ── Big-tournament sets (exact names) ─────────────────────────────────────────
_BIG_CHAMPIONSHIPS: frozenset[str] = frozenset({
    "European Championship",
    "Copa America",
    "African Nations Cup",
})

# Qualifiers for the top-tier competitions (WC + the 3 big championships above).
# Detection uses substring matching after the exact checks pass.
_BIG_QUAL_SUBSTRINGS: tuple[str, ...] = (
    "world cup",            # catches WC qualifier / WC and Asian Cup qualifier / etc.
    "european championship",
    "african nations cup",
    "copa america",
)


# ── Competition importance (4 / 3 / 2 / 1) ────────────────────────────────────
def get_competition_importance(competition: str) -> float:
    """
    4 = World Cup (exactly "World Cup")
    3 = European Championship / Copa America / African Nations Cup (exactly, not qualifiers)
    2 = Qualifiers for the big-4 competitions (WC, Euro, Copa, AfCON)
    1 = Everything else (friendlies, small tournaments, other qualifiers)

    WC detection uses startswith("World Cup") to exclude informal events like
    "Mini World Cup" or "VIVA World Cup" that contain "world cup" but aren't FIFA.
    """
    c = str(competition).strip()
    cl = c.lower()

    if c == "World Cup":
        return 4.0
    if c in _BIG_CHAMPIONSHIPS:
        return 3.0
    # WC qualifier variants all start with "World Cup " (after the exact match above)
    if cl.startswith("world cup"):
        return 2.0
    # Big-championship qualifiers (European Championship qualifier, etc.)
    for sub in ("european championship", "african nations cup", "copa america"):
        if sub in cl:
            return 2.0
    return 1.0


# ── Tournament key encoding ────────────────────────────────────────────────────
def _is_qualifier_string(cl: str) -> bool:
    """Return True if the lowercased competition name indicates a qualifier game."""
    return (
        "qualifier" in cl
        or cl.endswith(" qual")
        or " qual " in cl
        or cl.endswith(" q")
        or " q " in cl
    )


def get_tournament_key_numeric(competition: str) -> int:
    """
    Encode competition string as an ordinal category.

    1 = World Cup (exactly "World Cup")
    2 = European Championship / Copa America / African Nations Cup (exact, not qualifiers)
    3 = World Cup qualifiers — name starts with "World Cup" (catches all WC qual variants)
    4 = Other qualifier games (name contains "qualifier" / "qual" / trailing " q")
    5 = Other tournaments
    6 = Friendlies
    """
    c = str(competition).strip()
    cl = c.lower()

    if c == "World Cup":
        return 1
    if c in _BIG_CHAMPIONSHIPS:
        return 2
    # All WC-qualifier variants start with "World Cup " (exact WC already handled above)
    if cl.startswith("world cup"):
        return 3
    if _is_qualifier_string(cl):
        return 4
    if "friendly" in cl:
        return 6
    return 5


# ── Feature engineering ────────────────────────────────────────────────────────
def build_v5_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add V5-specific columns to a standardised model dataset.

    New columns added (non-destructive — existing columns untouched):
      competition_importance          float  4/3/2/1 based on competition tier
      tournament_key_numeric          int    ordinal category (1–6)
      rest_diff                       float  team_a_days - team_b_days (signed diff)
      tournament_goals_for_per_match_diff   float  0 for non-MAJOR_TOURNAMENTS
      tournament_goals_against_per_match_diff float  0 for non-MAJOR_TOURNAMENTS

    Args:
        df: standardised dataset (team_A/team_B, goals_A/goals_B, competition, date, tournament_year)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # competition_importance — clean 4/3/2/1 tiers, NOT the COMPETITION_WEIGHTS lookup
    df["competition_importance"] = df["competition"].apply(get_competition_importance).astype(float)

    # tournament_key_numeric
    df["tournament_key_numeric"] = df["competition"].apply(get_tournament_key_numeric).astype(int)

    # rest_diff (signed: positive = team_a had more rest)
    if "team_a_days_since_last_match" in df.columns and "team_b_days_since_last_match" in df.columns:
        df["rest_diff"] = (
            df["team_a_days_since_last_match"] - df["team_b_days_since_last_match"]
        ).astype(float)
    else:
        df["rest_diff"] = 0.0

    # Per-tournament goals_for/against rate diffs
    df = _add_tournament_rate_features(df)

    return df


def _add_tournament_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-team, per-tournament goals_for/against per match rates as pre-match snapshots.

    Only populated for MAJOR_TOURNAMENTS. Non-major rows get 0.
    Each tournament instance is identified by (competition, tournament_year) — no bleed-over
    between the same tournament in different years.
    """
    df = df.copy()
    df["tournament_goals_for_per_match_diff"] = 0.0
    df["tournament_goals_against_per_match_diff"] = 0.0

    # Resolve goal columns (after load_model_dataset standardisation)
    goal_a_col = "goals_A" if "goals_A" in df.columns else "target_goals_a"
    goal_b_col = "goals_B" if "goals_B" in df.columns else "target_goals_b"
    team_a_col = "team_A" if "team_A" in df.columns else "team_a"
    team_b_col = "team_B" if "team_B" in df.columns else "team_b"

    df_sorted = df.sort_values("date").reset_index(drop=False)  # preserves original index

    for (comp, year), group in df_sorted.groupby(["competition", "tournament_year"], sort=False):
        if comp not in MAJOR_TOURNAMENTS:
            continue

        team_state: dict[str, dict] = {}

        for _, row in group.sort_values("date").iterrows():
            orig_idx = row["index"]
            team_a = row[team_a_col]
            team_b = row[team_b_col]

            sa = team_state.get(team_a, {"matches": 0, "goals_for": 0, "goals_against": 0})
            sb = team_state.get(team_b, {"matches": 0, "goals_for": 0, "goals_against": 0})

            gf_a = sa["goals_for"] / sa["matches"] if sa["matches"] > 0 else 0.0
            ga_a = sa["goals_against"] / sa["matches"] if sa["matches"] > 0 else 0.0
            gf_b = sb["goals_for"] / sb["matches"] if sb["matches"] > 0 else 0.0
            ga_b = sb["goals_against"] / sb["matches"] if sb["matches"] > 0 else 0.0

            df.loc[orig_idx, "tournament_goals_for_per_match_diff"] = gf_a - gf_b
            df.loc[orig_idx, "tournament_goals_against_per_match_diff"] = ga_a - ga_b

            goals_a = int(row[goal_a_col])
            goals_b = int(row[goal_b_col])

            for team, gf, ga in [(team_a, goals_a, goals_b), (team_b, goals_b, goals_a)]:
                if team not in team_state:
                    team_state[team] = {"matches": 0, "goals_for": 0, "goals_against": 0}
                team_state[team]["matches"] += 1
                team_state[team]["goals_for"] += gf
                team_state[team]["goals_against"] += ga

    return df


# ── Fold creation ─────────────────────────────────────────────────────────────
def load_and_prepare_dataset(dataset_path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the updated dataset, standardise columns, and add V5 features."""
    df = load_model_dataset(path=dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = build_v5_features(df)
    return df


def create_v5_fold(
    df: pd.DataFrame,
    fold: tuple[str, int],
    holdout: tuple[str, int] = HOLDOUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (train_df, val_df) for a single fold.

    train = all rows EXCEPT this fold AND except the holdout (WC 2022).
    val   = exactly this fold's rows.

    This ensures WC 2022 never appears in any tuning fold's training set.
    """
    comp, year = fold
    holdout_comp, holdout_year = holdout

    val_mask = (
        (df["competition"].str.strip() == comp) &
        (df["tournament_year"] == year)
    )
    holdout_mask = (
        (df["competition"].str.strip() == holdout_comp) &
        (df["tournament_year"] == holdout_year)
    )

    train_mask = ~val_mask & ~holdout_mask

    return df[train_mask].copy(), df[val_mask].copy()


def create_holdout_split(
    df: pd.DataFrame,
    holdout: tuple[str, int] = HOLDOUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (train_df, holdout_df) for the one-shot WC 2022 evaluation.

    train = all rows except the holdout.
    """
    comp, year = holdout
    holdout_mask = (
        (df["competition"].str.strip() == comp) &
        (df["tournament_year"] == year)
    )
    return df[~holdout_mask].copy(), df[holdout_mask].copy()


# ── OOF lambda generation ─────────────────────────────────────────────────────
def generate_oof_lambdas(
    df: pd.DataFrame,
    model_factory: Callable,
    feature_cols: list[str],
    weights_fn: Callable[[pd.DataFrame], np.ndarray] | None = None,
    folds: list[tuple[str, int]] = TUNING_FOLDS,
    holdout: tuple[str, int] = HOLDOUT,
) -> pd.DataFrame:
    """
    Generate out-of-fold lambda predictions for all tuning folds.

    Each fold's model is trained on all data except that fold AND except the holdout.
    Returns a DataFrame with the original rows (tuning folds only) augmented with
    pred_lambda_a, pred_lambda_b columns.

    Args:
        df:              prepared dataset (output of load_and_prepare_dataset)
        model_factory:   callable() -> fresh unfitted model
        feature_cols:    list of feature column names
        weights_fn:      optional (df) -> np.ndarray sample weights; receives train_df
        folds:           list of (competition, year) tuning folds
        holdout:         (competition, year) to exclude from all training
    """
    all_oof = []

    for fold in folds:
        comp, year = fold
        train_df, val_df = create_v5_fold(df, fold, holdout)

        # Keep as DataFrames — LGBMGoalModel.predict uses mirror_features which
        # requires named columns to negate diff cols and swap paired cols.
        # Converting to numpy (.values) silently breaks mirroring → lambda_a ≈ lambda_b.
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[TARGET_COLS].values
        X_val = val_df[feature_cols].fillna(0)

        w_train = weights_fn(train_df) if weights_fn is not None else None

        model = model_factory()
        if w_train is not None:
            try:
                model.fit(X_train, y_train, sample_weight=w_train)
            except TypeError:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        preds = np.clip(model.predict(X_val), 0.0, None)

        oof_df = val_df.copy()
        oof_df["pred_lambda_a"] = preds[:, 0]
        oof_df["pred_lambda_b"] = preds[:, 1]
        oof_df["fold_competition"] = comp
        oof_df["fold_year"] = year
        all_oof.append(oof_df)

        n_train = len(train_df)
        n_val = len(val_df)
        print(f"  Fold {comp} {year}: train={n_train}, val={n_val}")

    return pd.concat(all_oof, ignore_index=True)


# ── Metrics ───────────────────────────────────────────────────────────────────
def raw_exact_score_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted exact-score accuracy (primary metric; never anomaly-downweights)."""
    exact = np.all(y_true.astype(int) == y_pred.astype(int), axis=1)
    return float(exact.mean())


def raw_outcome_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted W/D/L accuracy."""
    true_sign = np.sign(y_true[:, 0] - y_true[:, 1])
    pred_sign = np.sign(y_pred[:, 0] - y_pred[:, 1])
    return float((true_sign == pred_sign).mean())


def draw_rate(y: np.ndarray) -> float:
    """Fraction of draws (goals_A == goals_B) in an array of integer scores."""
    return float((y[:, 0].astype(int) == y[:, 1].astype(int)).mean())


def scoreline_distribution(y: np.ndarray, top_n: int = 10) -> dict[str, int]:
    """Count occurrences of each scoreline."""
    counts: dict[str, int] = {}
    for a, b in zip(y[:, 0].astype(int), y[:, 1].astype(int)):
        key = f"{a}-{b}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1])[:top_n])


def non_basic_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Count and accuracy of 'non-basic' scorelines: 2-1, 1-2, 3-1, 1-3, 3-2, 2-3, 2-2.
    These are the interesting scorelines the v4 conversion under-produces.
    """
    non_basic = {(2, 1), (1, 2), (3, 1), (1, 3), (3, 2), (2, 3), (2, 2)}
    actual_non_basic = np.array([
        (int(a), int(b)) in non_basic
        for a, b in zip(y_true[:, 0], y_true[:, 1])
    ])
    pred_non_basic = np.array([
        (int(a), int(b)) in non_basic
        for a, b in zip(y_pred[:, 0], y_pred[:, 1])
    ])
    n_actual = int(actual_non_basic.sum())
    n_pred = int(pred_non_basic.sum())
    correct_on_nb = int(np.all(y_true.astype(int) == y_pred.astype(int), axis=1)[actual_non_basic].sum())
    return {
        "n_actual_non_basic": n_actual,
        "n_pred_non_basic": n_pred,
        "correct_on_non_basic": correct_on_nb,
        "non_basic_exact_acc": correct_on_nb / max(n_actual, 1),
    }


def rps_batch(y_true: np.ndarray, probs_matrix: np.ndarray) -> float:
    """Ranked Probability Score for W/D/L (lower is better)."""
    from src.evaluation.metrics import rps_batch as _rps
    return _rps(y_true, probs_matrix)


def compute_fold_metrics(
    y_true: np.ndarray,
    pred_scores: np.ndarray,
    lambdas: np.ndarray | None = None,
    probs: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute the full suite of per-fold metrics.

    Args:
        y_true:      (n, 2) integer actual goals
        pred_scores: (n, 2) integer predicted scores
        lambdas:     (n, 2) raw predicted lambdas (optional)
        probs:       (n, 3) [win, draw, loss] probabilities (optional)
    """
    m: dict[str, Any] = {}
    m["n"] = len(y_true)
    m["exact_score_accuracy"] = raw_exact_score_accuracy(y_true, pred_scores)
    m["outcome_accuracy"] = raw_outcome_accuracy(y_true, pred_scores)
    m["actual_draw_rate"] = draw_rate(y_true)
    m["predicted_draw_rate"] = draw_rate(pred_scores)
    m["actual_distribution"] = scoreline_distribution(y_true)
    m["predicted_distribution"] = scoreline_distribution(pred_scores)
    m.update(non_basic_accuracy(y_true, pred_scores))

    if probs is not None:
        m["rps"] = rps_batch(y_true, probs)

    return m


# ── Aggregate OOF results ─────────────────────────────────────────────────────
def aggregate_oof_results(
    oof_df: pd.DataFrame,
    score_fn: Callable[[float, float], tuple[int, int]],
    probs_fn: Callable[[float, float], tuple[float, float, float]] | None = None,
) -> dict[str, Any]:
    """
    Score all OOF rows using a given score conversion function.

    Args:
        oof_df:   DataFrame from generate_oof_lambdas (has pred_lambda_a/b, goals_A/B)
        score_fn: (lambda_a, lambda_b) -> (pred_score_a, pred_score_b)
        probs_fn: (lambda_a, lambda_b) -> (win_p, draw_p, loss_p)  [optional]
    """
    la = oof_df["pred_lambda_a"].values
    lb = oof_df["pred_lambda_b"].values

    scores = np.array([score_fn(a, b) for a, b in zip(la, lb)])
    y_true = oof_df[TARGET_COLS].values.astype(int)

    probs = None
    if probs_fn is not None:
        probs = np.array([probs_fn(a, b) for a, b in zip(la, lb)])

    overall = compute_fold_metrics(y_true, scores, lambdas=np.stack([la, lb], axis=1), probs=probs)

    per_fold: dict[str, dict] = {}
    for (comp, year), grp in oof_df.groupby(["fold_competition", "fold_year"]):
        idx = grp.index
        per_fold[f"{comp} {year}"] = compute_fold_metrics(
            y_true[oof_df.index.get_indexer(idx)],
            scores[oof_df.index.get_indexer(idx)],
        )

    return {"overall": overall, "per_fold": per_fold}


# ── Holdout logging ───────────────────────────────────────────────────────────
def log_holdout_evaluation(
    results: dict[str, Any],
    strategy_name: str,
    path: str = HOLDOUT_LOG_PATH,
) -> None:
    """Append a holdout evaluation record to the JSON log file."""
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "strategy": strategy_name,
        "results": results,
    }

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    existing: list = []
    if log_path.exists():
        try:
            with open(log_path) as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = []

    existing.append(record)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)

    print(f"Holdout log written → {log_path}")


# ── Convenience: v4 baseline score function ──────────────────────────────────
def v4_score_fn(lambda_a: float, lambda_b: float, threshold: float = 0.9) -> tuple[int, int]:
    """V4 production decision rule: floor(λ + (1-threshold))."""
    shift = 1.0 - threshold
    return int(lambda_a + shift), int(lambda_b + shift)


def v4_probs_fn(lambda_a: float, lambda_b: float) -> tuple[float, float, float]:
    """W/D/L probs from independent Poisson (v4 path)."""
    from src.models.score_conversion import win_draw_loss_probs
    return win_draw_loss_probs(lambda_a, lambda_b)
