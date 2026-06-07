"""Group-stage standings utilities."""

from __future__ import annotations

import pandas as pd


def match_points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def build_group_standings(group_matches: pd.DataFrame) -> pd.DataFrame:
    """Build standings from matches with actual/predicted goals.

    Required columns:
    - group
    - team_a
    - team_b
    - goals_a
    - goals_b
    """
    rows = []

    for _, match in group_matches.iterrows():
        group = match["group"]
        team_a = match["team_a"]
        team_b = match["team_b"]
        goals_a = int(match["goals_a"])
        goals_b = int(match["goals_b"])

        rows.append({
            "group": group,
            "team": team_a,
            "played": 1,
            "wins": int(goals_a > goals_b),
            "draws": int(goals_a == goals_b),
            "losses": int(goals_a < goals_b),
            "goals_for": goals_a,
            "goals_against": goals_b,
            "goal_diff": goals_a - goals_b,
            "points": match_points(goals_a, goals_b),
        })

        rows.append({
            "group": group,
            "team": team_b,
            "played": 1,
            "wins": int(goals_b > goals_a),
            "draws": int(goals_a == goals_b),
            "losses": int(goals_b < goals_a),
            "goals_for": goals_b,
            "goals_against": goals_a,
            "goal_diff": goals_b - goals_a,
            "points": match_points(goals_b, goals_a),
        })

    standings = pd.DataFrame(rows)

    standings = (
        standings
        .groupby(["group", "team"], as_index=False)
        .agg({
            "played": "sum",
            "wins": "sum",
            "draws": "sum",
            "losses": "sum",
            "goals_for": "sum",
            "goals_against": "sum",
            "goal_diff": "sum",
            "points": "sum",
        })
    )

    standings = (
        standings
        .sort_values(
            ["group", "points", "goal_diff", "goals_for", "team"],
            ascending=[True, False, False, False, True],
        )
        .reset_index(drop=True)
    )

    standings["position"] = (
        standings
        .groupby("group")
        .cumcount()
        + 1
    )

    return standings


def get_group_position_map(standings: pd.DataFrame) -> dict[str, str]:
    """Map slots like 1A, 2A, 3A to team names."""
    slot_map = {}

    for _, row in standings.iterrows():
        group_letter = str(row["group"]).replace("Group ", "").strip()
        position = int(row["position"])
        team = row["team"]

        slot_map[f"{position}{group_letter}"] = team

    return slot_map


def get_best_third_placed_teams(standings: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Return best third-placed teams.

    World Cup 2026 has 12 groups. The best 8 third-placed teams advance.
    """
    thirds = standings[standings["position"] == 3].copy()

    thirds = (
        thirds
        .sort_values(
            ["points", "goal_diff", "goals_for", "team"],
            ascending=[False, False, False, True],
        )
        .head(n)
        .reset_index(drop=True)
    )

    return thirds
