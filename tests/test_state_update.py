"""Verify the actual live feature state, including points and pre-match values."""
from src.features.tournament_state_features import (
    initialize_team_states, compute_tournament_state_features, update_state_after_match,
)

def test_match_results_update_both_teams_and_next_match_features():
    states = initialize_team_states(["A", "B"])
    before = compute_tournament_state_features("A", "B", states)
    assert before["tournament_points_diff"] == 0
    update_state_after_match(states, "A", "B", 2, 0)
    assert states["A"]["points"] == 3
    assert states["B"]["points"] == 0
    assert states["A"]["clean_sheets"] == 1
    after = compute_tournament_state_features("A", "B", states, is_major_tournament=True)
    assert after["tournament_points_diff"] == 3
    assert after["tournament_goal_diff_diff"] == 4
    assert after["tournament_goals_for_per_match_diff"] == 2
    update_state_after_match(states, "A", "B", 1, 1)
    assert states["A"]["matches"] == states["B"]["matches"] == 2
    assert states["A"]["points"] == 4 and states["B"]["points"] == 1
    assert before["tournament_points_diff"] == 0
