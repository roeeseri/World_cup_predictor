from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MODEL_DATASET_PATH = Path("../data/processed/model_dataset.csv")

STANDARD_TARGET_COLUMNS = ["goals_A", "goals_B"]
STANDARD_METADATA_COLUMNS = [
    "date",
    "team_A",
    "team_B",
    "competition",
    "location",
    "season_id",
    "tournament_year",
    "tournament_key",
]

DATASET_TARGET_COLUMNS = ["target_goals_a", "target_goals_b"]
# target_goal_diff and target_total_goals are derived from the targets — pure leakage.
# goals_a / goals_b are the lowercase variants produced by standardize_goal_columns().
LEAKAGE_COLUMNS = ["target_goal_diff", "target_total_goals", "goals_a", "goals_b"]
WEIGHT_COLUMNS = ["competition_weight"]

EXCLUDED_FEATURE_COLUMNS = set(
    STANDARD_METADATA_COLUMNS
    + DATASET_TARGET_COLUMNS
    + LEAKAGE_COLUMNS
    + WEIGHT_COLUMNS
    + STANDARD_TARGET_COLUMNS
)


def standardize_model_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the modeling-side column names normalized."""
    rename_map = {
        "team_a": "team_A",
        "team_b": "team_B",
        "target_goals_a": "goals_A",
        "target_goals_b": "goals_B",
    }
    present_map = {source: target for source, target in rename_map.items() if source in df.columns}
    return df.rename(columns=present_map).copy()


def load_model_dataset(path: str | Path = MODEL_DATASET_PATH, standardize: bool = True) -> pd.DataFrame:
    """Load the partner-generated dataset and optionally normalize key column names."""
    dataset = pd.read_csv(Path(path))
    return standardize_model_dataset(dataset) if standardize else dataset


def infer_feature_columns(
    df: pd.DataFrame,
    *,
    exclude_columns: set[str] | None = None,
) -> list[str]:
    """Infer numeric feature columns from a model dataset."""
    standardized = standardize_model_dataset(df)
    excluded = set(EXCLUDED_FEATURE_COLUMNS)
    if exclude_columns:
        excluded.update(exclude_columns)

    numeric_columns = standardized.select_dtypes(include=[np.number]).columns.tolist()
    features = [column for column in numeric_columns if column not in excluded]

    _LEAKAGE_GUARD = {
        "goals_a", "goals_b", "goals_A", "goals_B",
        "target_goals_a", "target_goals_b",
        "target_goal_diff", "target_total_goals",
    }
    leaked = [c for c in features if c in _LEAKAGE_GUARD]
    if leaked:
        raise ValueError(f"Data leakage: target columns in feature set: {leaked}")

    return features


def resolve_feature_columns(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> list[str]:
    """Return an explicit feature list or infer one from the dataset."""
    if feature_columns is not None:
        missing = [column for column in feature_columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return list(feature_columns)
    return infer_feature_columns(df)


def build_sample_weight(
    df: pd.DataFrame,
    *,
    weight_column: str = "competition_weight",
    normalize: bool = True,
) -> np.ndarray | None:
    """Build normalized sample weights from a dataset column when present."""
    if weight_column not in df.columns:
        return None

    weights = pd.to_numeric(df[weight_column], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    weights = np.clip(weights, 0.0, None)
    if normalize and np.any(weights > 0):
        weights = weights / np.mean(weights)
    return weights


def coerce_goal_array(y) -> np.ndarray:
    """Convert supported goal representations into a 2D numpy array."""
    if isinstance(y, pd.DataFrame):
        if {"goals_A", "goals_B"}.issubset(y.columns):
            return y[["goals_A", "goals_B"]].to_numpy(dtype=float)
        if {"target_goals_a", "target_goals_b"}.issubset(y.columns):
            return y[["target_goals_a", "target_goals_b"]].to_numpy(dtype=float)
        if {"home_goals", "away_goals"}.issubset(y.columns):
            return y[["home_goals", "away_goals"]].to_numpy(dtype=float)
        return y.iloc[:, :2].to_numpy(dtype=float)

    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim != 2 or y_arr.shape[1] < 2:
        raise ValueError("Expected y with shape (n_samples, 2) for goal targets.")
    return y_arr[:, :2]


def ensure_non_negative(predictions) -> np.ndarray:
    """Clip raw goal predictions to valid non-negative values."""
    return np.clip(np.asarray(predictions, dtype=float), 0.0, None)
