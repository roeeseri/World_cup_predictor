import numpy as np

from src.evaluation.mock_data import DEFAULT_FEATURE_COLUMNS, generate_mock_feature_table
from src.models.baseline import AverageGoalsBaseline, ConstantScoreBaseline, EloHeuristicBaseline
from src.models.poisson_model import PoissonGoalModel
from src.models.tree_model import TreeGoalModel


def test_models_fit_predict_smoke():
    df = generate_mock_feature_table(n_matches=80, random_state=7)
    X = df[DEFAULT_FEATURE_COLUMNS]
    y = df[["goals_A", "goals_B"]]

    models = [
        ConstantScoreBaseline().fit(X, y),
        AverageGoalsBaseline().fit(X, y),
        EloHeuristicBaseline().fit(X, y),
        TreeGoalModel().fit(X, y),
        PoissonGoalModel().fit(X, y),
    ]

    for model in models:
        preds = model.predict(X.head(10))
        assert preds.shape == (10, 2)
        assert np.all(preds >= 0)
