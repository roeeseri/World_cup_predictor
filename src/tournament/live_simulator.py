from __future__ import annotations

import pandas as pd

from src.features.feature_columns import FEATURE_COLS
from src.models.score_conversion import most_likely_score, win_draw_loss_probs
from src.tournament.build_knockout import build_round_of_32_fixtures, validate_round_of_32
from src.tournament.group_standings import (
    build_group_standings,
    get_group_position_map,
)
from src.tournament.match_simulation import simulate_match


GROUP_PHASE = "GROUPS"
KNOCKOUT_ORDER = ["R32", "R16", "QF", "SF", "FINAL_STAGE"]
DONE_PHASE = "DONE"


def _empty_state() -> dict:
    return {
        "matches": 0,
        "points": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_diff": 0,
    }


def _points_for(gf: int, ga: int) -> int:
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0


def update_team_state(states: dict, team: str, gf: int, ga: int) -> None:
    states.setdefault(team, _empty_state())
    s = states[team]
    s["matches"] += 1
    s["points"] += _points_for(gf, ga)
    s["goals_for"] += gf
    s["goals_against"] += ga
    s["goal_diff"] = s["goals_for"] - s["goals_against"]


def initialize_live_state(group_features: pd.DataFrame) -> dict:
    fixtures = group_features.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"])
    fixtures = fixtures.sort_values(["date", "match_id"]).reset_index(drop=True)

    if "matchday" not in fixtures.columns:
        fixtures["matchday"] = ((fixtures.index // 24) + 1).astype(int)

    return {
        "phase": GROUP_PHASE,
        "current_matchday": 1,
        "current_knockout_round": None,
        "fixtures": fixtures,
        "played_matches": pd.DataFrame(),
        "last_matchday_results": pd.DataFrame(),
        "standings": pd.DataFrame(),
        "team_states": {},
        "r32_fixtures": pd.DataFrame(),
        "knockout_results": pd.DataFrame(),
        "last_knockout_results": pd.DataFrame(),
        "knockout_winners": {},
        "knockout_losers": {},
        "champion": None,
        "runner_up": None,
        "third_place": None,
        "done": False,
    }


def simulate_one_group_match(model, match: pd.Series, team_states: dict) -> dict:
    live_match = match.copy()

    team_a = live_match["team_a"]
    team_b = live_match["team_b"]

    state_a = team_states.setdefault(team_a, _empty_state())
    state_b = team_states.setdefault(team_b, _empty_state())

    live_match["team_a_tournament_matches_played"] = state_a["matches"]
    live_match["team_b_tournament_matches_played"] = state_b["matches"]
    live_match["tournament_points_diff"] = state_a["points"] - state_b["points"]
    live_match["tournament_goal_diff_diff"] = state_a["goal_diff"] - state_b["goal_diff"]

    X = pd.DataFrame([live_match[FEATURE_COLS]]).fillna(0)

    pred = model.predict(X)
    lambda_a = float(pred[0, 0])
    lambda_b = float(pred[0, 1])

    goals_a, goals_b = most_likely_score(lambda_a, lambda_b)
    win_a, draw, win_b = win_draw_loss_probs(lambda_a, lambda_b)

    if goals_a > goals_b:
        winner = team_a
    elif goals_b > goals_a:
        winner = team_b
    else:
        winner = "Draw"

    update_team_state(team_states, team_a, goals_a, goals_b)
    update_team_state(team_states, team_b, goals_b, goals_a)

    return {
        "match_id": live_match.get("match_id"),
        "date": live_match.get("date"),
        "matchday": live_match.get("matchday"),
        "group": live_match.get("group"),
        "team_a": team_a,
        "team_b": team_b,
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "pred_goals_a": goals_a,
        "pred_goals_b": goals_b,
        "pred_score": f"{goals_a}-{goals_b}",
        "team_a_win_prob": win_a,
        "draw_prob": draw,
        "team_b_win_prob": win_b,
        "winner": winner,
    }


def build_live_group_standings(played_matches: pd.DataFrame) -> pd.DataFrame:
    if played_matches.empty:
        return pd.DataFrame()

    matches = played_matches.rename(
        columns={
            "pred_goals_a": "goals_a",
            "pred_goals_b": "goals_b",
        }
    )

    return build_group_standings(
        matches[["group", "team_a", "team_b", "goals_a", "goals_b"]]
    )


def generate_round_of_32(state: dict) -> dict:
    standings = state["standings"]
    position_map = get_group_position_map(standings)
    r32 = build_round_of_32_fixtures(standings, position_map)
    validate_round_of_32(r32)

    state["r32_fixtures"] = r32
    state["phase"] = "R32"
    state["current_knockout_round"] = "R32"
    return state


def simulate_next_matchday(model, state: dict) -> dict:
    if state.get("phase") != GROUP_PHASE:
        return state

    fixtures = state["fixtures"]
    current_matchday = state["current_matchday"]

    matchday_fixtures = fixtures[fixtures["matchday"] == current_matchday].copy()

    if matchday_fixtures.empty:
        state["phase"] = "R32"
        return generate_round_of_32(state)

    rows = []

    for _, match in matchday_fixtures.iterrows():
        rows.append(
            simulate_one_group_match(
                model=model,
                match=match,
                team_states=state["team_states"],
            )
        )

    new_results = pd.DataFrame(rows)

    if state["played_matches"].empty:
        state["played_matches"] = new_results
    else:
        state["played_matches"] = pd.concat(
            [state["played_matches"], new_results],
            ignore_index=True,
        )

    state["current_matchday"] += 1
    state["last_matchday_results"] = new_results
    state["standings"] = build_live_group_standings(state["played_matches"])

    max_matchday = int(fixtures["matchday"].max())
    if state["current_matchday"] > max_matchday:
        state = generate_round_of_32(state)

    return state


def _fixtures_for_round(state: dict, round_name: str) -> list[tuple[str, str, str]]:
    if round_name == "R32":
        r32 = state["r32_fixtures"]
        return [
            (row["match_slot"], row["team_a"], row["team_b"])
            for _, row in r32.iterrows()
        ]

    winners = state["knockout_winners"]
    losers = state["knockout_losers"]

    if round_name == "R16":
        return [
            ("R16_01", winners["R32_01"], winners["R32_02"]),
            ("R16_02", winners["R32_03"], winners["R32_04"]),
            ("R16_03", winners["R32_05"], winners["R32_06"]),
            ("R16_04", winners["R32_07"], winners["R32_08"]),
            ("R16_05", winners["R32_09"], winners["R32_10"]),
            ("R16_06", winners["R32_11"], winners["R32_12"]),
            ("R16_07", winners["R32_13"], winners["R32_14"]),
            ("R16_08", winners["R32_15"], winners["R32_16"]),
        ]

    if round_name == "QF":
        return [
            ("QF_01", winners["R16_01"], winners["R16_02"]),
            ("QF_02", winners["R16_03"], winners["R16_04"]),
            ("QF_03", winners["R16_05"], winners["R16_06"]),
            ("QF_04", winners["R16_07"], winners["R16_08"]),
        ]

    if round_name == "SF":
        return [
            ("SF_01", winners["QF_01"], winners["QF_02"]),
            ("SF_02", winners["QF_03"], winners["QF_04"]),
        ]

    if round_name == "FINAL_STAGE":
        return [
            ("FINAL", winners["SF_01"], winners["SF_02"]),
            ("THIRD_PLACE", losers["SF_01"], losers["SF_02"]),
        ]

    raise ValueError(f"Unknown knockout round: {round_name}")


def simulate_next_knockout_round(model, state: dict) -> dict:
    round_name = state.get("current_knockout_round")

    if round_name is None:
        if state.get("r32_fixtures", pd.DataFrame()).empty:
            state = generate_round_of_32(state)
        round_name = "R32"
        state["current_knockout_round"] = round_name
        state["phase"] = round_name

    fixtures = _fixtures_for_round(state, round_name)

    rows = []
    winners = {}
    losers = {}

    for match_slot, team_a, team_b in fixtures:
        result = simulate_match(
            model=model,
            group_features=state["fixtures"],
            team_states=state["team_states"],
            team_a=team_a,
            team_b=team_b,
            knockout=True,
        )

        display_round = round_name
        if round_name == "FINAL_STAGE":
            display_round = "FINAL" if match_slot == "FINAL" else "THIRD_PLACE"

        result["round"] = display_round
        result["match_slot"] = match_slot

        rows.append(result)
        winners[match_slot] = result["winner"]
        losers[match_slot] = result["loser"]

    new_results = pd.DataFrame(rows)
    state["last_knockout_results"] = new_results

    if state["knockout_results"].empty:
        state["knockout_results"] = new_results
    else:
        state["knockout_results"] = pd.concat(
            [state["knockout_results"], new_results],
            ignore_index=True,
        )

    state["knockout_winners"].update(winners)
    state["knockout_losers"].update(losers)

    idx = KNOCKOUT_ORDER.index(round_name)
    if idx == len(KNOCKOUT_ORDER) - 1:
        final = new_results[new_results["match_slot"] == "FINAL"].iloc[0]
        third = new_results[new_results["match_slot"] == "THIRD_PLACE"].iloc[0]

        state["champion"] = final["winner"]
        state["runner_up"] = final["loser"]
        state["third_place"] = third["winner"]
        state["phase"] = DONE_PHASE
        state["current_knockout_round"] = None
        state["done"] = True
    else:
        next_round = KNOCKOUT_ORDER[idx + 1]
        state["current_knockout_round"] = next_round
        state["phase"] = next_round

    return state


def advance_live_tournament(model, state: dict) -> dict:
    if state.get("phase") == GROUP_PHASE:
        return simulate_next_matchday(model, state)

    if state.get("phase") in ["R32", "R16", "QF", "SF", "FINAL_STAGE"]:
        return simulate_next_knockout_round(model, state)

    return state
