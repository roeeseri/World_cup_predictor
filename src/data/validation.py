"""Validation helpers for model datasets and live prediction feature rows."""

import pandas as pd


def validate_feature_columns(df: pd.DataFrame, expected_columns: list[str]) -> bool:
    """Validate that a DataFrame contains exactly the expected feature columns.

    Raises:
        ValueError: if columns are missing, extra, non-numeric, or contain all-NaN values.
    """
    missing = [col for col in expected_columns if col not in df.columns]
    extra = [col for col in df.columns if col not in expected_columns]

    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    if extra:
        raise ValueError(f"Unexpected extra columns: {extra}")

    non_numeric = [
        col for col in expected_columns
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    if non_numeric:
        raise ValueError(f"Non-numeric feature columns: {non_numeric}")

    all_nan = [
        col for col in expected_columns
        if df[col].isna().all()
    ]

    if all_nan:
        raise ValueError(f"Feature columns are entirely NaN: {all_nan}")

    return True


def validate_no_target_columns(df: pd.DataFrame) -> bool:
    """Ensure target/result columns are not present in a prediction feature row."""
    forbidden = [
        "goals_a",
        "goals_b",
        "target_goals_a",
        "target_goals_b",
        "target_goal_diff",
        "target_total_goals",
    ]

    found = [col for col in forbidden if col in df.columns]

    if found:
        raise ValueError(f"Target columns found in feature row: {found}")

    return True
