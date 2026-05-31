"""Tournament-state feature helpers."""


def empty_team_state() -> dict:
    """Create an empty tournament state for one team."""
    return {
        "matches": 0,
        "points": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_diff": 0,
        "clean_sheets": 0,
    }


def initialize_team_states(teams: list[str]) -> dict:
    """Initialize tournament states for all teams."""
    return {team: empty_team_state() for team in teams}


def compute_tournament_state_features(
    team_a: str,
    team_b: str,
    team_states: dict,
) -> dict:
    """Compute tournament-state features before kickoff."""
    state_a = team_states.get(team_a, empty_team_state())
    state_b = team_states.get(team_b, empty_team_state())

    return {
        "team_a_tournament_matches_played": state_a["matches"],
        "team_b_tournament_matches_played": state_b["matches"],
        "tournament_points_diff": state_a["points"] - state_b["points"],
        "tournament_goal_diff_diff": state_a["goal_diff"] - state_b["goal_diff"],
    }


def update_state_after_match(
    team_states: dict,
    team_a: str,
    team_b: str,
    goals_a: int,
    goals_b: int,
) -> dict:
    """Update tournament states after a completed match."""
    team_states.setdefault(team_a, empty_team_state())
    team_states.setdefault(team_b, empty_team_state())

    state_a = team_states[team_a]
    state_b = team_states[team_b]

    if goals_a > goals_b:
        points_a, points_b = 3, 0
    elif goals_a < goals_b:
        points_a, points_b = 0, 3
    else:
        points_a, points_b = 1, 1

    state_a["matches"] += 1
    state_a["points"] += points_a
    state_a["goals_for"] += goals_a
    state_a["goals_against"] += goals_b
    state_a["goal_diff"] = state_a["goals_for"] - state_a["goals_against"]
    state_a["clean_sheets"] += int(goals_b == 0)

    state_b["matches"] += 1
    state_b["points"] += points_b
    state_b["goals_for"] += goals_b
    state_b["goals_against"] += goals_a
    state_b["goal_diff"] = state_b["goals_for"] - state_b["goals_against"]
    state_b["clean_sheets"] += int(goals_a == 0)

    return team_states
