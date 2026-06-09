"""Resolve knockout bracket placeholders into real teams."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.tournament.bracket import get_round_of_32_template
from src.tournament.group_standings import get_best_third_placed_teams

THIRD_COMBO_CSV = Path("data/processed/round32_third_combinations.csv")

# Each CSV column header is the group-1st-place that the 3rd-place team will face.
# The value maps back to the template's slot string (team_b_slot in the R32 template).
_CSV_COL_TO_THIRD_SLOT: dict[str, str] = {
    "1A": "3CEFHI",
    "1B": "3EFGIJ",
    "1D": "3BEFIJ",
    "1E": "3ABCDF",
    "1G": "3AEHIJ",
    "1I": "3CDFGH",
    "1K": "3DEIJL",
    "1L": "3EHIJK",
}


def _normalize_group(group: str) -> str:
    return str(group).replace("Group ", "").strip()


def _greedy_third_place_slot_map(best_thirds: pd.DataFrame) -> dict[str, str]:
    """Fallback: greedy first-fit assignment if CSV lookup fails."""
    all_third_slots = list(_CSV_COL_TO_THIRD_SLOT.values())
    slot_allowed: dict[str, set[str]] = {
        "3ABCDF": set("ABCDF"),
        "3CDFGH": set("CDFGH"),
        "3BEFIJ": set("BEFIJ"),
        "3AEHIJ": set("AEHIJ"),
        "3CEFHI": set("CEFHI"),
        "3EHIJK": set("EHIJK"),
        "3EFGIJ": set("EFGIJ"),
        "3DEIJL": set("DEIJL"),
    }
    used_teams: set[str] = set()
    result: dict[str, str] = {}
    for slot in all_third_slots:
        allowed = slot_allowed[slot]
        candidates = best_thirds[
            best_thirds["group_letter"].isin(allowed)
            & ~best_thirds["team"].isin(used_teams)
        ]
        if candidates.empty:
            candidates = best_thirds[~best_thirds["team"].isin(used_teams)]
        if candidates.empty:
            continue
        selected = candidates.iloc[0]
        result[slot] = selected["team"]
        used_teams.add(selected["team"])
    return result


def build_third_place_slot_map(
    standings: pd.DataFrame,
    n: int = 8,
    combo_csv: Path = THIRD_COMBO_CSV,
) -> dict[str, str]:
    """Return {slot_string: team_name} for all 8 third-place R32 slots.

    Uses the official combination CSV which maps every C(12,8)=495 possible
    set of qualifying groups to specific slot assignments.
    Falls back to a greedy algorithm if the CSV is unavailable or the
    combination isn't found.
    """
    best_thirds = get_best_third_placed_teams(standings, n=n).copy()
    best_thirds["group_letter"] = best_thirds["group"].apply(_normalize_group)

    letter_to_team: dict[str, str] = {
        row["group_letter"]: row["team"]
        for _, row in best_thirds.iterrows()
    }
    qualifying = frozenset(letter_to_team.keys())

    slot_cols = list(_CSV_COL_TO_THIRD_SLOT.keys())

    if combo_csv.exists():
        combos = pd.read_csv(combo_csv)
        for _, row in combos.iterrows():
            row_groups = frozenset(row[col][1] for col in slot_cols)  # "3E" → "E"
            if row_groups == qualifying:
                result: dict[str, str] = {}
                for csv_col, slot_str in _CSV_COL_TO_THIRD_SLOT.items():
                    grp_letter = row[csv_col][1]
                    team = letter_to_team.get(grp_letter)
                    if team:
                        result[slot_str] = team
                return result

    return _greedy_third_place_slot_map(best_thirds)


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
