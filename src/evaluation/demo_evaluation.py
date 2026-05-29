from __future__ import annotations

import pandas as pd

from ..models.baseline import AverageGoalsBaseline, ConstantScoreBaseline, EloHeuristicBaseline
from ..models.poisson_model import PoissonGoalModel
from ..models.tree_model import TreeGoalModel
from ..prediction.score_conversion import convert_expected_goals_to_scores
from .evaluate import compare_models, validate_model_dataset
from .mock_data import DEFAULT_FEATURE_COLUMNS, generate_mock_feature_table


FEATURE_COLUMNS = DEFAULT_FEATURE_COLUMNS
TARGET_COLUMNS = ["goals_A", "goals_B"]


def run_demo() -> pd.DataFrame:
    """
    Main demo: Generate synthetic data, train 5 models, evaluate with anomaly handling.
    
    Models tested:
    1. Constant: Always predicts average goals (baseline floor)
    2. Average: Same as constant (sanity check)
    3. Elo Heuristic: Uses Elo diff to predict (simple but effective)
    4. Random Forest: 100 trees with max_depth=15 (regularized)
    5. Poisson: L2 regularization with alpha=10.0
    
    Anomalies (goal_diff > 4) are downweighted 70% during evaluation to prevent
    extreme blowouts (9-0) from skewing metrics.
    """
    df = generate_mock_feature_table(n_matches=600, random_state=42)

    validate_model_dataset(df, FEATURE_COLUMNS, TARGET_COLUMNS)

    # Use 80% train / 20% test split
    test_mask = df.sample(frac=0.2, random_state=42).index
    test_df = df.loc[test_mask]
    train_df = df.drop(test_mask)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMNS]

    models = {
        "constant": ConstantScoreBaseline().fit(X_train, y_train),
        "average": AverageGoalsBaseline().fit(X_train, y_train),
        "elo_heuristic": EloHeuristicBaseline().fit(X_train, y_train),
        "tree": TreeGoalModel().fit(X_train, y_train),
        "poisson": PoissonGoalModel().fit(X_train, y_train),
    }

    comparison = compare_models(models, X_test, y_test, score_method="poisson")

    y_pred_expected = models["poisson"].predict(X_test)
    _ = convert_expected_goals_to_scores(y_pred_expected, method="round")

    return comparison


def main() -> None:
    comparison = run_demo()
    with pd.option_context("display.max_columns", None):
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
