"""World Football Elo-style rating updates.

This is designed to match eloratings.net behavior more closely than FIFA ranking.

Formula:
    R_new = R_old + round(K * G * (W - We))

Where:
- K = tournament importance
- G = goal-difference multiplier
- W = actual result
- We = expected result
- home team gets +100 rating points in expected-result calculation
"""

from __future__ import annotations


_MEXICO_VENUES = {"mexico", "mexico city", "guadalajara", "monterrey"}
_USA_VENUES = {
    "united states", "usa", "los angeles", "san francisco", "seattle",
    "dallas", "houston", "kansas city", "miami", "philadelphia",
    "new york", "boston", "nashville", "atlanta", "detroit",
}
_CANADA_VENUES = {"canada", "toronto", "vancouver", "edmonton"}


def _determine_home(team_a: str, team_b: str, location: str) -> str | None:
    """Return 'a', 'b', or None for World Football Elo home advantage."""
    loc = str(location).lower()
    ta = str(team_a).lower()
    tb = str(team_b).lower()

    if any(x in loc for x in _MEXICO_VENUES):
        if ta == "mexico":
            return "a"
        if tb == "mexico":
            return "b"

    if any(x in loc for x in _USA_VENUES):
        if ta in {"usa", "united states"}:
            return "a"
        if tb in {"usa", "united states"}:
            return "b"

    if any(x in loc for x in _CANADA_VENUES):
        if ta == "canada":
            return "a"
        if tb == "canada":
            return "b"

    return None


def _k_factor(competition: str = "FIFA World Cup") -> float:
    """World Football Elo K factor."""
    comp = str(competition).lower()

    if "world cup" in comp and "qualifier" not in comp and "qualification" not in comp:
        return 60.0

    if any(
        x in comp
        for x in [
            "euro",
            "copa america",
            "africa cup",
            "african cup",
            "asian cup",
            "gold cup",
            "concacaf",
            "continental",
            "intercontinental",
            "nations league final",
        ]
    ):
        return 50.0

    if "qualifier" in comp or "qualification" in comp:
        return 40.0

    if "friendly" in comp:
        return 20.0

    return 30.0


def _goal_multiplier(goals_a: int, goals_b: int) -> float:
    """World Football Elo goal difference multiplier."""
    diff = abs(int(goals_a) - int(goals_b))

    if diff <= 1:
        return 1.0

    if diff == 2:
        return 1.5

    return (11.0 + diff) / 8.0


def _actual_result(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def _expected_result(rating_a: float, rating_b: float) -> float:
    return 1.0 / (10.0 ** (-(rating_a - rating_b) / 400.0) + 1.0)


def compute_elo_update(
    rating_a: float,
    rating_b: float,
    goals_a: int,
    goals_b: int,
    competition: str = "FIFA World Cup",
    team_a: str = "",
    team_b: str = "",
    location: str = "neutral",
    stage: str | None = None,
    knockout: bool = False,
) -> tuple[float, float]:
    """Compute World Football Elo-style rating changes.

    Returns:
        (delta_a, delta_b), rounded to whole Elo points.
    """
    rating_a = float(rating_a)
    rating_b = float(rating_b)

    adjusted_a = rating_a
    adjusted_b = rating_b

    home = _determine_home(team_a, team_b, location)

    if home == "a":
        adjusted_a += 100.0
    elif home == "b":
        adjusted_b += 100.0

    expected_a = _expected_result(adjusted_a, adjusted_b)
    actual_a = _actual_result(goals_a, goals_b)

    k = _k_factor(competition)
    g = _goal_multiplier(goals_a, goals_b)

    delta_a = round(k * g * (actual_a - expected_a))
    delta_b = -delta_a

    return float(delta_a), float(delta_b)