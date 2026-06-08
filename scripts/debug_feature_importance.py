from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.features.feature_columns import FEATURE_COLS


MODEL_PATH = "models/production_model_v3.joblib"


def normalize_importance(values):
    values = pd.Series(values, dtype=float)
    total = values.sum()
    if total == 0:
        return values
    return values / total


def main():
    model = joblib.load(MODEL_PATH)

    rows = []

    for i, submodel in enumerate(model.models):
        name = type(submodel).__name__

        imp_raw = submodel.feature_importances(FEATURE_COLS)

        if isinstance(imp_raw, dict):
            imp = pd.Series(imp_raw, dtype=float)
        else:
            imp = imp_raw.set_index("feature")["importance"].astype(float)

        imp = normalize_importance(imp)

        for feature, value in imp.items():
            rows.append(
                {
                    "model_index": i,
                    "model_name": name,
                    "feature": feature,
                    "importance_norm": value,
                }
            )

    df = pd.DataFrame(rows)

    pivot = (
        df.pivot_table(
            index="feature",
            columns="model_name",
            values="importance_norm",
            aggfunc="mean",
        )
        .fillna(0)
    )

    pivot["avg_importance"] = pivot.mean(axis=1)

    print(
        pivot.sort_values("avg_importance", ascending=False)
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
