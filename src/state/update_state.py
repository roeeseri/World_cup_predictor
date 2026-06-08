from __future__ import annotations

from copy import deepcopy


def derive_rankings_from_elo(elo_ratings: dict[str, float]) -> dict[str, int]:
    """Rank teams by ELO descending (highest ELO = rank 1)."""
    sorted_teams = sorted(elo_ratings.items(), key=lambda x: -x[1])
    return {team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)}


def _default_team_state() -> dict:
    return {
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_diff": 0,
        "points": 0,
    }


def update_state_after_match(team_state: dict, match_result: dict) -> dict:
    home_team = match_result["home_team"]
    away_team = match_result["away_team"]
    home_goals = int(match_result["home_goals"])
    away_goals = int(match_result["away_goals"])

    next_state = deepcopy(team_state) if team_state is not None else {}
    next_state.setdefault(home_team, _default_team_state())
    next_state.setdefault(away_team, _default_team_state())

    home = next_state[home_team]
    away = next_state[away_team]

    home["played"] += 1
    away["played"] += 1
    home["goals_for"] += home_goals
    home["goals_against"] += away_goals
    away["goals_for"] += away_goals
    away["goals_against"] += home_goals

    if home_goals > away_goals:
        home["wins"] += 1
        away["losses"] += 1
        home["points"] += 3
    elif home_goals < away_goals:
        away["wins"] += 1
        home["losses"] += 1
        away["points"] += 3
    else:
        home["draws"] += 1
        away["draws"] += 1
        home["points"] += 1
        away["points"] += 1

    home["goal_diff"] = home["goals_for"] - home["goals_against"]
    away["goal_diff"] = away["goals_for"] - away["goals_against"]

    return next_state
