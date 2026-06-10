"""WC 2026 step-by-step tournament simulation page."""

from __future__ import annotations

import copy

import pandas as pd
import streamlit as st

from src.features.build_features import build_pre_match_features
from src.features.team_names import normalize_team_name
from src.models.score_conversion import most_likely_score, most_likely_score_v5, win_draw_loss_probs
from src.state.elo import compute_elo_update
from src.state.live_state import (
    derive_rankings_from_elo,
    initialize_live_state,
    record_match_result,
)
from src.tournament.build_knockout import build_round_of_32_fixtures
from src.tournament.group_standings import (
    build_group_standings,
    get_best_third_placed_teams,
    get_group_position_map,
)

# ---------------------------------------------------------------------------
# Bracket pairings (derived from bracket.py template)
# ---------------------------------------------------------------------------

R16_PAIRINGS = [
    # Ordered chronologically by game date (July 4 → 7)
    ("R16_01", "R32_01", "R32_02"),  # game 89, July 4: W(74) vs W(77)
    ("R16_02", "R32_03", "R32_04"),  # game 90, July 4: W(73) vs W(75)
    ("R16_05", "R32_09", "R32_10"),  # game 91, July 5: W(76) vs W(78)
    ("R16_06", "R32_11", "R32_12"),  # game 92, July 5: W(79) vs W(80)
    ("R16_03", "R32_05", "R32_06"),  # game 93, July 6: W(83) vs W(84)
    ("R16_04", "R32_07", "R32_08"),  # game 94, July 6: W(81) vs W(82)
    ("R16_07", "R32_13", "R32_14"),  # game 95, July 7: W(86) vs W(88)
    ("R16_08", "R32_15", "R32_16"),  # game 96, July 7: W(85) vs W(87)
]

QF_PAIRINGS = [
    ("QF_01", "R16_01", "R16_02"),
    ("QF_02", "R16_03", "R16_04"),
    ("QF_03", "R16_05", "R16_06"),
    ("QF_04", "R16_07", "R16_08"),
]

SF_PAIRINGS = [
    ("SF_01", "QF_01", "QF_02"),
    ("SF_02", "QF_03", "QF_04"),
]

# Exact dates per match slot, derived from the official schedule
# R32 game numbers 73-88, R16 89-96, QF 97-100, SF 101-102, 3rd 103, Final 104
R32_SLOT_DATES: dict[str, pd.Timestamp] = {
    "R32_03": pd.Timestamp("2026-06-28"),  # game 73: 2A vs 2B
    "R32_01": pd.Timestamp("2026-06-29"),  # game 74: 1E vs 3ABCDF
    "R32_04": pd.Timestamp("2026-06-29"),  # game 75: 1F vs 2C
    "R32_09": pd.Timestamp("2026-06-29"),  # game 76: 1C vs 2F
    "R32_02": pd.Timestamp("2026-06-30"),  # game 77: 1I vs 3CDFGH
    "R32_10": pd.Timestamp("2026-06-30"),  # game 78: 2E vs 2I
    "R32_11": pd.Timestamp("2026-06-30"),  # game 79: 1A vs 3CEFHI
    "R32_12": pd.Timestamp("2026-07-01"),  # game 80: 1L vs 3EHIJK
    "R32_07": pd.Timestamp("2026-07-01"),  # game 81: 1D vs 3BEFIJ
    "R32_08": pd.Timestamp("2026-07-01"),  # game 82: 1G vs 3AEHIJ
    "R32_05": pd.Timestamp("2026-07-02"),  # game 83: 2K vs 2L
    "R32_06": pd.Timestamp("2026-07-02"),  # game 84: 1H vs 2J
    "R32_15": pd.Timestamp("2026-07-02"),  # game 85: 1B vs 3EFGIJ
    "R32_13": pd.Timestamp("2026-07-03"),  # game 86: 1J vs 2H
    "R32_16": pd.Timestamp("2026-07-03"),  # game 87: 1K vs 3DEIJL
    "R32_14": pd.Timestamp("2026-07-03"),  # game 88: 2D vs 2G
}

R16_SLOT_DATES: dict[str, pd.Timestamp] = {
    "R16_01": pd.Timestamp("2026-07-04"),  # game 89: W(74) vs W(77)
    "R16_02": pd.Timestamp("2026-07-04"),  # game 90: W(73) vs W(75)
    "R16_05": pd.Timestamp("2026-07-05"),  # game 91: W(76) vs W(78)
    "R16_06": pd.Timestamp("2026-07-05"),  # game 92: W(79) vs W(80)
    "R16_03": pd.Timestamp("2026-07-06"),  # game 93: W(83) vs W(84)
    "R16_04": pd.Timestamp("2026-07-06"),  # game 94: W(81) vs W(82)
    "R16_07": pd.Timestamp("2026-07-07"),  # game 95: W(86) vs W(88)
    "R16_08": pd.Timestamp("2026-07-07"),  # game 96: W(85) vs W(87)
}

QF_SLOT_DATES: dict[str, pd.Timestamp] = {
    "QF_01": pd.Timestamp("2026-07-09"),   # game 97
    "QF_02": pd.Timestamp("2026-07-10"),   # game 98
    "QF_03": pd.Timestamp("2026-07-11"),   # game 99
    "QF_04": pd.Timestamp("2026-07-11"),   # game 100
}

SF_SLOT_DATES: dict[str, pd.Timestamp] = {
    "SF_01": pd.Timestamp("2026-07-14"),   # game 101
    "SF_02": pd.Timestamp("2026-07-15"),   # game 102
}

FINAL_SLOT_DATES: dict[str, pd.Timestamp] = {
    "THIRD_PLACE": pd.Timestamp("2026-07-18"),  # game 103
    "FINAL":       pd.Timestamp("2026-07-19"),  # game 104
}

# All slot → date in one lookup
_ALL_SLOT_DATES: dict[str, pd.Timestamp] = {
    **R32_SLOT_DATES, **R16_SLOT_DATES,
    **QF_SLOT_DATES, **SF_SLOT_DATES, **FINAL_SLOT_DATES,
}

STAGE_LABELS = {
    "R32": "Round of 32",
    "R16": "Round of 16",
    "QF": "Quarter Finals",
    "SF": "Semi Finals",
    "FINAL": "Final + 3rd Place",
}

_FLAGS: dict[str, str] = {
    "Argentina": "🇦🇷", "Brazil": "🇧🇷", "France": "🇫🇷", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Spain": "🇪🇸", "Germany": "🇩🇪", "Portugal": "🇵🇹", "Netherlands": "🇳🇱",
    "Belgium": "🇧🇪", "Croatia": "🇭🇷", "Uruguay": "🇺🇾", "Mexico": "🇲🇽",
    "United States": "🇺🇸", "USA": "🇺🇸", "Canada": "🇨🇦", "Japan": "🇯🇵",
    "South Korea": "🇰🇷", "Korea Republic": "🇰🇷", "Morocco": "🇲🇦",
    "Senegal": "🇸🇳", "Switzerland": "🇨🇭", "Colombia": "🇨🇴",
    "Ecuador": "🇪🇨", "Australia": "🇦🇺", "Iran": "🇮🇷", "Qatar": "🇶🇦",
    "Saudi Arabia": "🇸🇦", "Ghana": "🇬🇭", "Tunisia": "🇹🇳", "Egypt": "🇪🇬",
    "Turkey": "🇹🇷", "Norway": "🇳🇴", "Sweden": "🇸🇪",
    "Czechia": "🇨🇿", "Austria": "🇦🇹", "Algeria": "🇩🇿",
    "Ivory Coast": "🇨🇮", "New Zealand": "🇳🇿", "Panama": "🇵🇦",
    "Paraguay": "🇵🇾", "South Africa": "🇿🇦", "Cape Verde": "🇨🇻",
    "Haiti": "🇭🇹", "Jordan": "🇯🇴", "Iraq": "🇮🇶", "Uzbekistan": "🇺🇿",
    "DR Congo": "🇨🇩", "Bosnia and Herzegovina": "🇧🇦", "Curaçao": "🇨🇼",
    "Venezuela": "🇻🇪", "Chile": "🇨🇱", "Peru": "🇵🇪", "Bolivia": "🇧🇴",
    "Costa Rica": "🇨🇷", "Honduras": "🇭🇳", "Jamaica": "🇯🇲",
    "Nigeria": "🇳🇬", "Cameroon": "🇨🇲", "Mali": "🇲🇱",
    "Serbia": "🇷🇸", "Poland": "🇵🇱", "Ukraine": "🇺🇦", "Romania": "🇷🇴",
    "Hungary": "🇭🇺", "Slovakia": "🇸🇰", "Greece": "🇬🇷", "Denmark": "🇩🇰",
    "Finland": "🇫🇮", "Iceland": "🇮🇸", "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
    "Indonesia": "🇮🇩", "Thailand": "🇹🇭",
}


def _flag(team: str) -> str:
    return _FLAGS.get(team, _FLAGS.get(normalize_team_name(team), "⚽"))


def _team_label(team: str) -> str:
    return f"{_flag(team)} {team}"


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_wc_sim(historical_matches: pd.DataFrame, fixtures: pd.DataFrame) -> None:
    if "wc_sim" in st.session_state:
        return
    state = initialize_live_state(historical_matches, fixtures)
    st.session_state["wc_sim"] = {
        "state": state,
        "stage_complete": {k: False for k in ["R1", "R2", "R3", "R32", "R16", "QF", "SF", "FINAL"]},
        "group_results": {1: [], 2: [], 3: []},
        "r32_fixtures": None,
        "knockout_results": {"R32": [], "R16": [], "QF": [], "SF": [], "FINAL": []},
    }


# ---------------------------------------------------------------------------
# Core prediction helpers
# ---------------------------------------------------------------------------

def _get_last5(historical_matches: pd.DataFrame, team: str) -> pd.DataFrame:
    mask = (historical_matches["team_a"] == team) | (historical_matches["team_b"] == team)
    df = historical_matches[mask].copy()
    df["date"] = pd.to_datetime(df["date"])
    return (
        df.sort_values("date", ascending=False)
        .head(5)
        [["date", "team_a", "team_b", "goals_a", "goals_b"]]
        .reset_index(drop=True)
    )


def _predict_match(
    state: dict,
    model,
    team_a: str,
    team_b: str,
    match_date,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    match_id: int | None = None,
    match_slot: str | None = None,
    score_fn=None,
    feature_fn=None,
) -> dict:
    if score_fn is None:
        score_fn = most_likely_score
    if feature_fn is None:
        feature_fn = build_pre_match_features

    canon_a = normalize_team_name(team_a)
    canon_b = normalize_team_name(team_b)
    error_msg = None

    try:
        feature_row = feature_fn(
            team_a=canon_a,
            team_b=canon_b,
            match_date=match_date,
            team_states=state["team_states"],
            historical_matches=state["historical_matches"],
            market_values=market_values,
            position_values=position_values,
            elo_ratings=state["elo_ratings"],
            rankings=state["rankings"],
        ).fillna(0)

        pred = model.predict(feature_row)
        lambda_a = float(pred[0, 0])
        lambda_b = float(pred[0, 1])
        goals_a, goals_b = score_fn(lambda_a, lambda_b)
        win_a, draw, win_b = win_draw_loss_probs(lambda_a, lambda_b)
    except Exception as exc:
        error_msg = str(exc)
        lambda_a = lambda_b = 1.0
        goals_a = goals_b = 1
        win_a = draw = win_b = 1 / 3
        feature_row = pd.DataFrame()

    is_tie = goals_a == goals_b
    if not is_tie:
        winner = team_a if goals_a > goals_b else team_b
    else:
        winner = team_a if win_a >= win_b else team_b

    last5_a = _get_last5(state["historical_matches"], canon_a)
    last5_b = _get_last5(state["historical_matches"], canon_b)

    return {
        "match_id": match_id,
        "match_slot": match_slot,
        "team_a": team_a,
        "team_b": team_b,
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "goals_a": goals_a,
        "goals_b": goals_b,
        "win_a": win_a,
        "draw": draw,
        "win_b": win_b,
        "winner": winner,
        "is_tie": is_tie,
        "feature_row": feature_row,
        "last5_a": last5_a,
        "last5_b": last5_b,
        "error": error_msg,
    }


# ---------------------------------------------------------------------------
# Simulation runners
# ---------------------------------------------------------------------------

def _simulate_group_round(
    wc_sim: dict,
    model,
    matchday: int,
    fixtures: pd.DataFrame,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    score_fn=None,
    feature_fn=None,
) -> None:
    state = wc_sim["state"]
    matchday_fixtures = (
        fixtures[fixtures["matchday"] == matchday]
        .sort_values("date")
        .reset_index(drop=True)
    )

    results = []
    for _, fix in matchday_fixtures.iterrows():
        result = _predict_match(
            state=state,
            model=model,
            team_a=fix["team_a"],
            team_b=fix["team_b"],
            match_date=pd.to_datetime(fix["date"]),
            market_values=market_values,
            position_values=position_values,
            match_id=int(fix["match_id"]),
            score_fn=score_fn,
            feature_fn=feature_fn,
        )
        state = record_match_result(state, int(fix["match_id"]), result["goals_a"], result["goals_b"])
        results.append(result)

    wc_sim["state"] = state
    wc_sim["group_results"][matchday] = results
    wc_sim["stage_complete"][f"R{matchday}"] = True


def _apply_knockout_result_to_state(
    state: dict,
    team_a: str,
    team_b: str,
    goals_a: int,
    goals_b: int,
    match_date: pd.Timestamp,
) -> dict:
    state = copy.deepcopy(state)
    from src.features.tournament_state_features import update_state_after_match

    canon_a = normalize_team_name(team_a)
    canon_b = normalize_team_name(team_b)

    ra = state["elo_ratings"].get(canon_a, state["elo_ratings"].get(team_a, 1500.0))
    rb = state["elo_ratings"].get(canon_b, state["elo_ratings"].get(team_b, 1500.0))

    delta_a, delta_b = compute_elo_update(
        rating_a=ra,
        rating_b=rb,
        goals_a=goals_a,
        goals_b=goals_b,
        competition="FIFA World Cup",
        team_a=team_a,
        team_b=team_b,
    )

    state["elo_ratings"][canon_a] = ra + delta_a
    state["elo_ratings"][canon_b] = rb + delta_b
    if team_a != canon_a:
        state["elo_ratings"][team_a] = ra + delta_a
    if team_b != canon_b:
        state["elo_ratings"][team_b] = rb + delta_b

    state["rankings"] = derive_rankings_from_elo(state["elo_ratings"])
    state["team_states"] = update_state_after_match(state["team_states"], canon_a, canon_b, goals_a, goals_b)

    new_row = {
        "date": match_date,
        "team_a": canon_a,
        "team_b": canon_b,
        "goals_a": goals_a,
        "goals_b": goals_b,
        "competition": "FIFA World Cup",
        "location": "neutral",
        "rating_change_a": delta_a,
        "rating_change_b": delta_b,
        "rating_a": ra + delta_a,
        "rating_b": rb + delta_b,
        "rating_a_before": ra,
        "rating_b_before": rb,
        "rank_a": state["rankings"].get(canon_a, 0),
        "rank_b": state["rankings"].get(canon_b, 0),
        "rank_a_before": state["rankings"].get(canon_a, 0),
        "rank_b_before": state["rankings"].get(canon_b, 0),
        "rank_change_a": 0,
        "rank_change_b": 0,
        "elo_diff": ra - rb,
        "rank_diff": 0,
        "source_file": "wc_sim_2026",
        "tournament_year": 2026,
        "tournament_key": "FIFA World Cup_2026",
    }
    state["historical_matches"] = pd.concat(
        [state["historical_matches"], pd.DataFrame([new_row])],
        ignore_index=True,
    )
    return state


def _simulate_knockout_stage(
    wc_sim: dict,
    model,
    stage: str,
    fixtures_list: list[tuple[str, str, str]],
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    score_fn=None,
    feature_fn=None,
) -> None:
    state = wc_sim["state"]
    fallback_date = pd.Timestamp("2026-07-10")

    results = []
    for match_slot, team_a, team_b in fixtures_list:
        match_date = _ALL_SLOT_DATES.get(match_slot, fallback_date)
        result = _predict_match(
            state=state,
            model=model,
            team_a=team_a,
            team_b=team_b,
            match_date=match_date,
            market_values=market_values,
            position_values=position_values,
            match_slot=match_slot,
            score_fn=score_fn,
            feature_fn=feature_fn,
        )
        state = _apply_knockout_result_to_state(
            state, team_a, team_b, result["goals_a"], result["goals_b"], match_date
        )
        results.append(result)

    wc_sim["state"] = state
    wc_sim["knockout_results"][stage] = results
    wc_sim["stage_complete"][stage] = True


def _build_r32_fixtures(wc_sim: dict) -> pd.DataFrame:
    state = wc_sim["state"]
    completed = state["fixtures"][state["fixtures"]["is_completed"]].copy()
    completed["goals_a"] = completed["goals_a"].astype(int)
    completed["goals_b"] = completed["goals_b"].astype(int)
    standings = build_group_standings(
        completed[["group", "team_a", "team_b", "goals_a", "goals_b"]]
    )
    position_map = get_group_position_map(standings)
    r32 = build_round_of_32_fixtures(standings, position_map)
    # Add the correct date per slot and sort chronologically
    r32["date"] = r32["match_slot"].map(R32_SLOT_DATES)
    r32 = r32.sort_values("date").reset_index(drop=True)
    wc_sim["r32_fixtures"] = r32
    return r32


# ---------------------------------------------------------------------------
# Bracket fixture builders (from previous stage winners)
# ---------------------------------------------------------------------------

def _get_r16_fixtures(wc_sim: dict) -> list[tuple[str, str, str]]:
    winners = {r["match_slot"]: r["winner"] for r in wc_sim["knockout_results"]["R32"]}
    return [(slot, winners[a], winners[b]) for slot, a, b in R16_PAIRINGS]


def _get_qf_fixtures(wc_sim: dict) -> list[tuple[str, str, str]]:
    winners = {r["match_slot"]: r["winner"] for r in wc_sim["knockout_results"]["R16"]}
    return [(slot, winners[a], winners[b]) for slot, a, b in QF_PAIRINGS]


def _get_sf_fixtures(wc_sim: dict) -> list[tuple[str, str, str]]:
    winners = {r["match_slot"]: r["winner"] for r in wc_sim["knockout_results"]["QF"]}
    return [(slot, winners[a], winners[b]) for slot, a, b in SF_PAIRINGS]


def _get_final_fixtures(wc_sim: dict) -> list[tuple[str, str, str]]:
    sf = {r["match_slot"]: r for r in wc_sim["knockout_results"]["SF"]}
    sf1, sf2 = sf["SF_01"], sf["SF_02"]
    loser_sf1 = sf1["team_b"] if sf1["winner"] == sf1["team_a"] else sf1["team_a"]
    loser_sf2 = sf2["team_b"] if sf2["winner"] == sf2["team_a"] else sf2["team_a"]
    return [
        ("FINAL", sf1["winner"], sf2["winner"]),
        ("THIRD_PLACE", loser_sf1, loser_sf2),
    ]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_game_card(result: dict, is_knockout: bool = False) -> None:
    ta, tb = result["team_a"], result["team_b"]

    with st.container(border=True):
        col_match, col_probs = st.columns([3, 2])

        with col_match:
            slot = result.get("match_slot", "")
            slot_label = f"**{slot}**  " if slot else ""
            st.markdown(
                f"{slot_label}{_team_label(ta)}  **{result['goals_a']} – {result['goals_b']}**  {_team_label(tb)}"
            )
            st.caption(f"xG: {result['lambda_a']:.2f} – {result['lambda_b']:.2f}")

            if result.get("error"):
                st.warning(f"Prediction fallback: {result['error']}")
            elif is_knockout:
                if result["is_tie"]:
                    st.info(f"Draw on score — **{_team_label(result['winner'])} advances** (higher win probability)")
                else:
                    st.success(f"**{_team_label(result['winner'])} advances**")

        with col_probs:
            c1, c2, c3 = st.columns(3)
            c1.metric(_team_label(ta), f"{result['win_a'] * 100:.0f}%")
            c2.metric("Draw", f"{result['draw'] * 100:.0f}%")
            c3.metric(_team_label(tb), f"{result['win_b'] * 100:.0f}%")

        with st.expander("Game details"):
            l, r = st.columns(2)
            with l:
                st.markdown(f"**Last 5 — {ta}**")
                if not result["last5_a"].empty:
                    st.dataframe(result["last5_a"], use_container_width=True, hide_index=True)
                else:
                    st.caption("No history found.")
            with r:
                st.markdown(f"**Last 5 — {tb}**")
                if not result["last5_b"].empty:
                    st.dataframe(result["last5_b"], use_container_width=True, hide_index=True)
                else:
                    st.caption("No history found.")
            if not result["feature_row"].empty:
                st.markdown("**Features used**")
                st.dataframe(
                    result["feature_row"].T.rename(columns={result["feature_row"].index[0]: "value"}),
                    use_container_width=True,
                )


def _render_standings(state: dict, show_thirds: bool = False) -> None:
    completed = state["fixtures"][state["fixtures"]["is_completed"]].copy()
    if completed.empty:
        st.caption("No completed games yet.")
        return

    completed["goals_a"] = completed["goals_a"].astype(int)
    completed["goals_b"] = completed["goals_b"].astype(int)
    standings = build_group_standings(
        completed[["group", "team_a", "team_b", "goals_a", "goals_b"]]
    )

    groups = sorted(standings["group"].unique())
    for i in range(0, len(groups), 3):
        cols = st.columns(3)
        for col, grp in zip(cols, groups[i : i + 3]):
            with col:
                st.markdown(f"**{grp}**")
                gdf = standings[standings["group"] == grp][
                    ["position", "team", "played", "wins", "draws", "losses",
                     "goals_for", "goals_against", "goal_diff", "points"]
                ].copy()
                gdf["position"] = gdf["position"].apply(lambda x: f"🟢 {x}" if x <= 2 else f"⚪ {x}")
                st.dataframe(gdf, use_container_width=True, hide_index=True)

    if show_thirds:
        st.subheader("Best 8 Third-Placed Teams (R32 Qualifiers)")
        thirds = get_best_third_placed_teams(standings, n=8)
        st.dataframe(
            thirds[["group", "team", "points", "goal_diff", "goals_for"]],
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def _render_group_round_tab(
    wc_sim: dict,
    model,
    matchday: int,
    fixtures: pd.DataFrame,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    prerequisite_done: bool,
    score_fn=None,
    feature_fn=None,
) -> None:
    stage_key = f"R{matchday}"
    done = wc_sim["stage_complete"].get(stage_key, False)

    if not prerequisite_done:
        st.info(f"Complete Round {matchday - 1} first to unlock this round.")
        return

    if not done:
        n = int((fixtures["matchday"] == matchday).sum())
        if st.button(f"▶ Predict Round {matchday} ({n} games)", type="primary", key=f"btn_r{matchday}"):
            with st.spinner(f"Simulating {n} Round {matchday} matches…"):
                _simulate_group_round(wc_sim, model, matchday, fixtures, market_values, position_values, score_fn=score_fn, feature_fn=feature_fn)
            st.rerun()
        return

    results = wc_sim["group_results"][matchday]
    # Group by fixture group letter for nicer display
    groups_seen: set[str] = set()
    for result in results:
        # find the group for this fixture
        fix_rows = fixtures[fixtures["match_id"] == result["match_id"]]
        grp = fix_rows["group"].iloc[0] if not fix_rows.empty else ""
        if grp not in groups_seen:
            if groups_seen:
                st.markdown("---")
            st.markdown(f"##### {grp}")
            groups_seen.add(grp)
        _render_game_card(result, is_knockout=False)

    st.divider()
    st.subheader(f"Standings after Round {matchday}")
    _render_standings(wc_sim["state"])


def _render_final_standings_tab(wc_sim: dict) -> None:
    r3_done = wc_sim["stage_complete"].get("R3", False)
    if not r3_done:
        st.info("Complete all 3 group rounds first.")
        return

    st.subheader("Final Group Standings")
    _render_standings(wc_sim["state"], show_thirds=True)

    st.divider()
    if wc_sim.get("r32_fixtures") is None:
        if st.button("Generate Round of 32 Bracket", type="primary", key="btn_gen_r32"):
            with st.spinner("Building bracket…"):
                _build_r32_fixtures(wc_sim)
            st.rerun()
    else:
        st.subheader("Round of 32 Fixtures")
        r32_display = wc_sim["r32_fixtures"][["match_slot", "team_a_slot", "team_b_slot", "team_a", "team_b"]].copy()
        r32_display["team_a"] = r32_display["team_a"].apply(_team_label)
        r32_display["team_b"] = r32_display["team_b"].apply(_team_label)
        st.dataframe(r32_display, use_container_width=True, hide_index=True)


def _render_knockout_tab(
    wc_sim: dict,
    model,
    stage: str,
    get_fixtures_fn,
    prerequisite_done: bool,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    score_fn=None,
    feature_fn=None,
) -> None:
    label = STAGE_LABELS.get(stage, stage)
    done = wc_sim["stage_complete"].get(stage, False)

    if not prerequisite_done:
        st.info(f"Complete the previous stage first to unlock {label}.")
        return

    if not done:
        fixtures_list = get_fixtures_fn(wc_sim)
        n = len(fixtures_list)
        if st.button(f"▶ Predict {label} ({n} games)", type="primary", key=f"btn_{stage}"):
            with st.spinner(f"Simulating {n} {label} matches…"):
                _simulate_knockout_stage(wc_sim, model, stage, fixtures_list, market_values, position_values, score_fn=score_fn, feature_fn=feature_fn)
            st.rerun()

        # Preview who plays who
        if fixtures_list:
            st.markdown("**Upcoming fixtures:**")
            for slot, ta, tb in fixtures_list:
                st.markdown(f"- **{slot}**: {_team_label(ta)} vs {_team_label(tb)}")
        return

    results = wc_sim["knockout_results"][stage]
    for result in results:
        slot = result.get("match_slot", "")
        if slot == "FINAL":
            st.subheader("🏆 Final")
        elif slot == "THIRD_PLACE":
            st.subheader("🥉 Third Place Match")
        _render_game_card(result, is_knockout=True)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def show_wc_simulation(
    model,
    historical_matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    score_fn=None,
    feature_fn=None,
) -> None:
    """Render the step-by-step WC 2026 simulation page."""
    _init_wc_sim(historical_matches, fixtures)
    wc_sim = st.session_state["wc_sim"]

    st.header("🏆 WC 2026 — Tournament Simulator")

    sc = wc_sim["stage_complete"]
    hdr_col, reset_col = st.columns([5, 1])
    with hdr_col:
        stages_done = [k for k, v in sc.items() if v]
        st.caption(
            f"Completed: {', '.join(stages_done) if stages_done else 'none'}"
            f"  |  Uses live feature pipeline (same as Live Tournament page)"
        )
    with reset_col:
        if st.button("🗑️ Reset", key="wc_sim_reset"):
            del st.session_state["wc_sim"]
            st.rerun()

    (
        tab_r1, tab_r2, tab_r3,
        tab_fs,
        tab_r32, tab_r16, tab_qf, tab_sf, tab_final,
    ) = st.tabs([
        "Group Round 1", "Group Round 2", "Group Round 3",
        "Final Standings",
        "Round of 32", "Round of 16", "Quarter Finals", "Semi Finals", "Final + 3rd",
    ])

    with tab_r1:
        _render_group_round_tab(wc_sim, model, 1, fixtures, market_values, position_values, prerequisite_done=True, score_fn=score_fn, feature_fn=feature_fn)

    with tab_r2:
        _render_group_round_tab(wc_sim, model, 2, fixtures, market_values, position_values,
                                prerequisite_done=sc.get("R1", False), score_fn=score_fn, feature_fn=feature_fn)

    with tab_r3:
        _render_group_round_tab(wc_sim, model, 3, fixtures, market_values, position_values,
                                prerequisite_done=sc.get("R2", False), score_fn=score_fn, feature_fn=feature_fn)

    with tab_fs:
        _render_final_standings_tab(wc_sim)

    with tab_r32:
        _render_knockout_tab(
            wc_sim, model, "R32",
            lambda ws: [(row["match_slot"], row["team_a"], row["team_b"])
                        for _, row in ws["r32_fixtures"].iterrows()],
            prerequisite_done=(wc_sim.get("r32_fixtures") is not None),
            market_values=market_values,
            position_values=position_values,
            score_fn=score_fn,
            feature_fn=feature_fn,
        )

    with tab_r16:
        _render_knockout_tab(
            wc_sim, model, "R16", _get_r16_fixtures,
            prerequisite_done=sc.get("R32", False),
            market_values=market_values,
            position_values=position_values,
            score_fn=score_fn,
            feature_fn=feature_fn,
        )

    with tab_qf:
        _render_knockout_tab(
            wc_sim, model, "QF", _get_qf_fixtures,
            prerequisite_done=sc.get("R16", False),
            market_values=market_values,
            position_values=position_values,
            score_fn=score_fn,
            feature_fn=feature_fn,
        )

    with tab_sf:
        _render_knockout_tab(
            wc_sim, model, "SF", _get_sf_fixtures,
            prerequisite_done=sc.get("QF", False),
            market_values=market_values,
            position_values=position_values,
            score_fn=score_fn,
            feature_fn=feature_fn,
        )

    with tab_final:
        _render_knockout_tab(
            wc_sim, model, "FINAL", _get_final_fixtures,
            prerequisite_done=sc.get("SF", False),
            market_values=market_values,
            position_values=position_values,
            score_fn=score_fn,
            feature_fn=feature_fn,
        )
