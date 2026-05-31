"""Match context features."""

def competition_weight(competition: str) -> float:
    competition = str(competition).lower()

    if competition == "world cup":
        return 5.0

    if (
        "euro" in competition
        or "copa america" in competition
        or "africa cup" in competition
        or "asian cup" in competition
        or "gold cup" in competition
    ):
        return 4.0

    if "nations league" in competition:
        return 2.5

    if "qualifier" in competition or "qualification" in competition:
        return 2.0

    if "friendly" in competition:
        return 0.5

    return 1.5


def compute_match_context(
    competition: str,
    is_home_adv: int = 0,
) -> dict:
    return {
        "competition_weight": competition_weight(competition),
        "is_home_adv": int(is_home_adv),
    }
