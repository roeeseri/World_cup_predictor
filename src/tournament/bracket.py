"""World Cup 2026 knockout bracket template.

The placeholders follow FIFA-style notation:
- 1A = winner of Group A
- 2A = runner-up of Group A
- 3ABCDF = one of the best third-place teams from groups A/B/C/D/F

This module only defines the bracket structure.
Resolving placeholders into real teams happens after group-stage simulation.
"""

from __future__ import annotations

import pandas as pd


ROUND_OF_32_TEMPLATE = [
    {"round": "R32", "match_slot": "R32_01", "team_a_slot": "1E", "team_b_slot": "3ABCDF"},
    {"round": "R32", "match_slot": "R32_02", "team_a_slot": "1I", "team_b_slot": "3CDFGH"},
    {"round": "R32", "match_slot": "R32_03", "team_a_slot": "2A", "team_b_slot": "2B"},
    {"round": "R32", "match_slot": "R32_04", "team_a_slot": "1F", "team_b_slot": "2C"},

    {"round": "R32", "match_slot": "R32_05", "team_a_slot": "2K", "team_b_slot": "2L"},
    {"round": "R32", "match_slot": "R32_06", "team_a_slot": "1H", "team_b_slot": "2J"},
    {"round": "R32", "match_slot": "R32_07", "team_a_slot": "1D", "team_b_slot": "3BEFIJ"},
    {"round": "R32", "match_slot": "R32_08", "team_a_slot": "1G", "team_b_slot": "3AEHIJ"},

    {"round": "R32", "match_slot": "R32_09", "team_a_slot": "1C", "team_b_slot": "2F"},
    {"round": "R32", "match_slot": "R32_10", "team_a_slot": "2E", "team_b_slot": "2I"},
    {"round": "R32", "match_slot": "R32_11", "team_a_slot": "1A", "team_b_slot": "3CEFHI"},
    {"round": "R32", "match_slot": "R32_12", "team_a_slot": "1L", "team_b_slot": "3EHIJK"},

    {"round": "R32", "match_slot": "R32_13", "team_a_slot": "1J", "team_b_slot": "2H"},
    {"round": "R32", "match_slot": "R32_14", "team_a_slot": "2D", "team_b_slot": "2G"},
    {"round": "R32", "match_slot": "R32_15", "team_a_slot": "1B", "team_b_slot": "3EFGIJ"},
    {"round": "R32", "match_slot": "R32_16", "team_a_slot": "1K", "team_b_slot": "3DEIJL"},
]


KNOCKOUT_TEMPLATE = [
    *ROUND_OF_32_TEMPLATE,

    {"round": "R16", "match_slot": "R16_01", "team_a_slot": "W_R32_01", "team_b_slot": "W_R32_02"},
    {"round": "R16", "match_slot": "R16_02", "team_a_slot": "W_R32_03", "team_b_slot": "W_R32_04"},
    {"round": "R16", "match_slot": "R16_03", "team_a_slot": "W_R32_05", "team_b_slot": "W_R32_06"},
    {"round": "R16", "match_slot": "R16_04", "team_a_slot": "W_R32_07", "team_b_slot": "W_R32_08"},
    {"round": "R16", "match_slot": "R16_05", "team_a_slot": "W_R32_09", "team_b_slot": "W_R32_10"},
    {"round": "R16", "match_slot": "R16_06", "team_a_slot": "W_R32_11", "team_b_slot": "W_R32_12"},
    {"round": "R16", "match_slot": "R16_07", "team_a_slot": "W_R32_13", "team_b_slot": "W_R32_14"},
    {"round": "R16", "match_slot": "R16_08", "team_a_slot": "W_R32_15", "team_b_slot": "W_R32_16"},

    {"round": "QF", "match_slot": "QF_01", "team_a_slot": "W_R16_01", "team_b_slot": "W_R16_02"},
    {"round": "QF", "match_slot": "QF_02", "team_a_slot": "W_R16_03", "team_b_slot": "W_R16_04"},
    {"round": "QF", "match_slot": "QF_03", "team_a_slot": "W_R16_05", "team_b_slot": "W_R16_06"},
    {"round": "QF", "match_slot": "QF_04", "team_a_slot": "W_R16_07", "team_b_slot": "W_R16_08"},

    {"round": "SF", "match_slot": "SF_01", "team_a_slot": "W_QF_01", "team_b_slot": "W_QF_02"},
    {"round": "SF", "match_slot": "SF_02", "team_a_slot": "W_QF_03", "team_b_slot": "W_QF_04"},

    {"round": "FINAL", "match_slot": "FINAL", "team_a_slot": "W_SF_01", "team_b_slot": "W_SF_02"},
    {"round": "THIRD_PLACE", "match_slot": "THIRD_PLACE", "team_a_slot": "L_SF_01", "team_b_slot": "L_SF_02"},
]


def get_knockout_template() -> pd.DataFrame:
    """Return the full knockout bracket template as a DataFrame."""
    return pd.DataFrame(KNOCKOUT_TEMPLATE)


def get_round_of_32_template() -> pd.DataFrame:
    """Return only the Round of 32 template."""
    return pd.DataFrame(ROUND_OF_32_TEMPLATE)
