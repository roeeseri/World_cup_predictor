"""Resolve knockout bracket placeholders into real teams."""

from __future__ import annotations

import re

import pandas as pd

from src.tournament.bracket import get_round_of_32_template
from src.tournament.group_standings import get_best_third_placed_teams


def _normalize_group(group: str) -> str:
    return str(group).replace("Group ", "").strip()


def build_third_place_slot_map(
    standings: pd.DataFrame,
    n: int = 8,
) -> dict[str, str]:
    """Build mapping for slots like 3ABCDF.

    The current implementation chooses the best available third-placed team
    whose group appears in the allowed slot string.

    Example:
    - slot 3ABCDF can be filled by the best third-place team from groups A/B/C/D/F.
    """
    best_thirds = get_best_third_placed_teams(standings, n=n).copy()
    best_thirds["group_letter"] = best_thirds["group"].apply(_normalize_group)

    used_teams = set()
    third_slot_map = {}

    all_third_slots = [
        "3ABCDF",
        "3CDFGH",
        "3BEFIJ",
        "3AEHIJ",
        "3CEFHI",
        "3EHIJK",
        "3EFGIJ",
        "3DEIJL",
    ]

    for slot in all_third_slots:
        allowed_groups = set(slot.replace("3", ""))

        candidates = best_thirds[
            best_thirds["group_letter"].isin(allowed_groups)
            & ~best_thirds["team"].isin(used_teams)
        ]

        if candidates.empty:
            third_slot_map[slot] = None
            continue

        selected = candidates.iloc[0]
        third_slot_map[slot] = selected["team"]
        used_teams.add(selected["team"])

    return third_slot_map


def resolve_basic_slot(
    slot: str,
    position_map: dict[str, str],
    third_slot_map: dict[str, str],
) -> str | None:
    """Resolve a slot like 1A, 2B, or 3ABCDF into a team."""
    slot = str(slot)

    if re.fullmatch(r"[12][A-L]", slot):
        return position_map.get(slot)

    if re.fullmatch(r"3[A-L]+", slot):
        return third_slot_map.get(slot)

    return None


def build_round_of_32_fixtures(
    standings: pd.DataFrame,
    position_map: dict[str, str],
) -> pd.DataFrame:
    """Create actual Round of 32 fixtures from group standings."""
    template = get_round_of_32_template()
    third_slot_map = build_third_place_slot_map(standings)

    rows = []

    for _, row in template.iterrows():
        team_a = resolve_basic_slot(
            row["team_a_slot"],
            position_map,
            third_slot_map,
        )

        team_b = resolve_basic_slot(
            row["team_b_slot"],
            position_map,
            third_slot_map,
        )

        rows.append({
            "round": row["round"],
            "match_slot": row["match_slot"],
            "team_a_slot": row["team_a_slot"],
            "team_b_slot": row["team_b_slot"],
            "team_a": team_a,
            "team_b": team_b,
        })

    return pd.DataFrame(rows)


def validate_round_of_32(fixtures: pd.DataFrame) -> None:
    """Validate that all R32 fixtures have resolved teams."""
    missing = fixtures[
        fixtures["team_a"].isna()
        | fixtures["team_b"].isna()
    ]

    if not missing.empty:
        raise ValueError(
            "Some Round of 32 slots could not be resolved:\n"
            + missing.to_string(index=False)
        )

    teams = fixtures["team_a"].tolist() + fixtures["team_b"].tolist()

    duplicate_teams = (
        pd.Series(teams)
        .value_counts()
        .loc[lambda s: s > 1]
    )

    if not duplicate_teams.empty:
        raise ValueError(
            "Duplicate teams found in Round of 32 fixtures:\n"
            + duplicate_teams.to_string()
        )
