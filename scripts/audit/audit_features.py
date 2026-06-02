"""Audit production model features."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.features.feature_columns import FEATURE_COLS


DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])

    print("Dataset shape:", df.shape)
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Number of features:", len(FEATURE_COLS))

    print("\n=== Missing feature columns ===")
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    print(missing_cols)

    print("\n=== Non numeric features ===")
    non_numeric = [
        col for col in FEATURE_COLS
        if not pd.api.types.is_numeric_dtype(df[col])
    ]
    print(non_numeric)

    print("\n=== Missing values by feature ===")
    missing = df[FEATURE_COLS].isna().sum()
    print(missing[missing > 0])

    print("\n=== Basic feature describe ===")
    desc = df[FEATURE_COLS].describe().T
    desc["zero_share"] = (df[FEATURE_COLS] == 0).mean()
    desc["non_zero_share"] = (df[FEATURE_COLS] != 0).mean()
    print(desc.sort_values("zero_share", ascending=False))

    print("\n=== Suspicious constant / near constant features ===")
    low_variance = desc[desc["std"] < 1e-9]
    print(low_variance)

    print("\n=== High correlations >= 0.90 ===")
    corr = df[FEATURE_COLS].corr().abs()
    pairs = []

    for i, col1 in enumerate(FEATURE_COLS):
        for col2 in FEATURE_COLS[i + 1:]:
            value = corr.loc[col1, col2]
            if value >= 0.90:
                pairs.append((col1, col2, value))

    corr_df = pd.DataFrame(
        pairs,
        columns=["feature_1", "feature_2", "abs_corr"]
    )

    if corr_df.empty:
        print("No high correlations found.")
    else:
        print(corr_df.sort_values("abs_corr", ascending=False))

    print("\n=== Top absolute values by feature ===")
    for col in FEATURE_COLS:
        top_abs = df[["date", "team_a", "team_b", col]].copy()
        top_abs["abs_value"] = top_abs[col].abs()
        top_abs = top_abs.sort_values("abs_value", ascending=False).head(5)

        print(f"\n--- {col} ---")
        print(top_abs[["date", "team_a", "team_b", col]].to_string(index=False))

    print("\n=== World Cup only describe ===")
    wc = df[df["competition"].astype(str).str.lower().eq("world cup")]
    print(wc[FEATURE_COLS].describe().T)


if __name__ == "__main__":
    main()
