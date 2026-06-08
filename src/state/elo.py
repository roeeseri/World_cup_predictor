"""ELO rating update calculations for live tournament state."""

from __future__ import annotations


# World Cup venue city keywords → host nation
_MEXICO_VENUES = {"mexico city", "guadalajara", "monterrey"}
_USA_VENUES = {
    "los angeles", "san francisco", "seattle", "dallas", "houston",
    "kansas city", "miami", "philadelphia", "new york", "boston",
    "nashville", "atlanta", "detroit",
}
_CANADA_VENUES = {"toronto", "vancouver", "edmonton"}


def _determine_home(team_a: str, team_b: str, location: str) -> str | None:
    """Return 'a', 'b', or None for home-team advantage."""
    loc = location.lower()
    if any(c in loc for c in _MEXICO_VENUES):
        if team_a == "Mexico":
            return "a"
        if team_b == "Mexico":
            return "b"
    if any(c in loc for c in _USA_VENUES):
        if team_a in ("USA", "United States"):
            return "a"
        if team_b in ("USA", "United States"):
            return "b"
    if any(c in loc for c in _CANADA_VENUES):
        if team_a == "Canada":
            return "a"
        if team_b == "Canada":
            return "b"
    return None


def _k_base(competition: str) -> float:
    comp = competition.lower()
    if "world cup" in comp and "qualifier" not in comp and "qualification" not in comp:
        return 60.0
    if any(
        x in comp
        for x in (
            "euro", "copa america", "africa cup", "african cup",
            "asian cup", "gold cup", "concacaf championship",
            "nations league final", "intercontinental",
        )
    ):
        return 50.0
    if "qualifier" in comp or "qualification" in comp:
        return 40.0
    if "friendly" in comp:
        return 20.0
    return 30.0


def _goal_multiplier(goals_a: int, goals_b: int) -> float:
    """K multiplier based on margin of victory."""
    margin = abs(goals_a - goals_b)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    if margin == 3:
        return 1.75
    return 1.75 + (margin - 3) / 8.0


def compute_elo_update(
    rating_a: float,
    rating_b: float,
    goals_a: int,
    goals_b: int,
    competition: str = "FIFA World Cup",
    team_a: str = "",
    team_b: str = "",
    location: str = "neutral",
) -> tuple[float, float]:
    """Compute ELO rating changes after a match.

    Returns:
        (delta_a, delta_b) — rating changes for team A and team B.
    """
    home = _determine_home(team_a, team_b, location)

    dr = rating_a - rating_b
    if home == "a":
        dr += 100.0
    elif home == "b":
        dr -= 100.0

    we = 1.0 / (10.0 ** (-dr / 400.0) + 1.0)

    if goals_a > goals_b:
        w = 1.0
    elif goals_a < goals_b:
        w = 0.0
    else:
        w = 0.5

    k = _k_base(competition) * _goal_multiplier(goals_a, goals_b)

    delta_a = k * (w - we)
    delta_b = -delta_a  # zero-sum

    return delta_a, delta_b
