"""Small real fit → predict → score → evaluate integration test (synthetic data)."""
import numpy as np
from src.evaluation.mock_data import DEFAULT_FEATURE_COLUMNS, generate_mock_feature_table
from src.evaluation.evaluate import compare_models, validate_model_dataset
from src.models.baseline import AverageGoalsBaseline

def test_fitted_model_evaluation_pipeline():
    df = generate_mock_feature_table(n_matches=50, random_state=7)
    targets = ["goals_A", "goals_B"]
    validate_model_dataset(df, DEFAULT_FEATURE_COLUMNS, targets)
    train, test = df.iloc[:35], df.iloc[35:]
    model = AverageGoalsBaseline().fit(train[DEFAULT_FEATURE_COLUMNS], train[targets])
    result = compare_models({"average": model}, test[DEFAULT_FEATURE_COLUMNS], test[targets])
    assert result["model"].tolist() == ["average"]
    assert np.isfinite(result.select_dtypes("number").to_numpy()).all()
    assert 0 <= result.iloc[0]["exact_score_accuracy"] <= 1
    assert 0 <= result.iloc[0]["result_accuracy"] <= 1
