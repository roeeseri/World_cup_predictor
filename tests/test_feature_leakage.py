"""Target exclusion checks; these do not certify the temporal research splits."""
import pandas as pd
import pytest
from src.models.base import infer_feature_columns
from src.data.validation import validate_no_target_columns

def test_inferred_features_exclude_targets_and_derived_results():
    df = pd.DataFrame({name: [1.0] for name in [
        "elo_diff", "goals_a", "goals_b", "goals_A", "goals_B",
        "target_goal_diff", "target_total_goals", "competition_weight",
    ]})
    assert infer_feature_columns(df) == ["elo_diff"]

@pytest.mark.parametrize("target", ["goals_a", "goals_b", "goals_A", "goals_B", "target_goals_a", "target_goals_b", "target_goal_diff", "target_total_goals"])
def test_live_feature_validation_rejects_result_columns(target):
    with pytest.raises(ValueError, match="Target columns"):
        validate_no_target_columns(pd.DataFrame({"elo_diff": [0.0], target: [1]}))
