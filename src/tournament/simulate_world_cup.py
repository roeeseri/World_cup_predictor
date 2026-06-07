from __future__ import annotations

import pandas as pd

from src.features.feature_columns import FEATURE_COLS
from src.tournament.bracket import get_knockout_template
from src.tournament.build_knockout import build_round_of_32_fixtures, validate_round_of_32
from src.tournament.group_standings import build_group_standings, get_group_position_map
from src.tournament.match_simulation import simulate_match


def prepare_group_prediction_matches(group_predictions: pd.DataFrame) -> pd.DataFrame:
    df = group_predictions.copy()

    if {"goals_a", "goals_b"}.issubset(df.columns):
        df["goals_a"] = df["goals_a"].astype(int)
        df["goals_b"] = df["goals_b"].astype(int)
    elif {"pred_goals_a", "pred_goals_b"}.issubset(df.columns):
        df["goals_a"] = df["pred_goals_a"].astype(int)
        df["goals_b"] = df["pred_goals_b"].astype(int)
    elif "pred_score" in df.columns:
        scores = df["pred_score"].astype(str).str.extract(r"(\d+)\s*-\s*(\d+)")
        if scores.isna().any().any():
            raise ValueError("Could not parse some pred_score values.")
        df["goals_a"] = scores[0].astype(int)
        df["goals_b"] = scores[1].astype(int)
    else:
        raise ValueError("Missing goals/predicted goals columns.")

    return df[["group", "team_a", "team_b", "goals_a", "goals_b"]].copy()


def _empty_team_state() -> dict:
    return {
        "matches": 0,
        "points": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_diff": 0,
    }


def _points_for(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _update_team_state(
    state: dict,
    goals_for: int,
    goals_against: int,
) -> None:
    state["matches"] += 1
    state["points"] += _points_for(goals_for, goals_against)
    state["goals_for"] += goals_for
    state["goals_against"] += goals_against
    state["goal_diff"] = state["goals_for"] - state["goals_against"]


def _apply_live_tournament_features(
    row: pd.Series,
    team_states: dict[str, dict],
) -> pd.Series:
    row = row.copy()

    team_a = row["team_a"]
    team_b = row["team_b"]

    state_a = team_states.setdefault(team_a, _empty_team_state())
    state_b = team_states.setdefault(team_b, _empty_team_state())

    row["team_a_tournament_matches_played"] = state_a["matches"]
    row["team_b_tournament_matches_played"] = state_b["matches"]
    row["tournament_points_diff"] = state_a["points"] - state_b["points"]
    row["tournament_goal_diff_diff"] = state_a["goal_diff"] - state_b["goal_diff"]

    return row


def _empty_team_state() -> dict:
    return {
        "matches": 0,
        "points": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_diff": 0,
    }


def _points_for(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _update_team_state(
    state: dict,
    goals_for: int,
    goals_against: int,
) -> None:
    state["matches"] += 1
    state["points"] += _points_for(goals_for, goals_against)
    state["goals_for"] += goals_for
    state["goals_against"] += goals_against
    state["goal_diff"] = state["goals_for"] - state["goals_against"]


def _apply_live_tournament_features(
    row: pd.Series,
    team_states: dict[str, dict],
) -> pd.Series:
    row = row.copy()

    team_a = row["team_a"]
    team_b = row["team_b"]

    state_a = team_states.setdefault(team_a, _empty_team_state())
    state_b = team_states.setdefault(team_b, _empty_team_state())

    row["team_a_tournament_matches_played"] = state_a["matches"]
    row["team_b_tournament_matches_played"] = state_b["matches"]
    row["tournament_points_diff"] = state_a["points"] - state_b["points"]
    row["tournament_goal_diff_diff"] = state_a["goal_diff"] - state_b["goal_diff"]

    return row


def _empty_team_state() -> dict:
    return {
        "matches": 0,
        "points": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_diff": 0,
    }


def _points_for(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _update_team_state(
    state: dict,
    goals_for: int,
    goals_against: int,
) -> None:
    state["matches"] += 1
    state["points"] += _points_for(goals_for, goals_against)
    state["goals_for"] += goals_for
    state["goals_against"] += goals_against
    state["goal_diff"] = state["goals_for"] - state["goals_against"]


def _apply_live_tournament_features(
    row: pd.Series,
    team_states: dict[str, dict],
) -> pd.Series:
    row = row.copy()

    team_a = row["team_a"]
    team_b = row["team_b"]

    state_a = team_states.setdefault(team_a, _empty_team_state())
    state_b = team_states.setdefault(team_b, _empty_team_state())

    row["team_a_tournament_matches_played"] = state_a["matches"]
    row["team_b_tournament_matches_played"] = state_b["matches"]
    row["tournament_points_diff"] = state_a["points"] - state_b["points"]
    row["tournament_goal_diff_diff"] = state_a["goal_diff"] - state_b["goal_diff"]

    return row


def _empty_team_state() -> dict:
    return {
        "matches": 0,
        "points": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_diff": 0,
    }


def _points_for(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _update_team_state(
    state: dict,
    goals_for: int,
    goals_against: int,
) -> None:
    state["matches"] += 1
    state["points"] += _points_for(goals_for, goals_against)
    state["goals_for"] += goals_for
    state["goals_against"] += goals_against
    state["goal_diff"] = state["goals_for"] - state["goals_against"]


def _apply_live_tournament_features(
    row: pd.Series,
    team_states: dict[str, dict],
) -> pd.Series:
    row = row.copy()

    team_a = row["team_a"]
    team_b = row["team_b"]

    state_a = team_states.setdefault(team_a, _empty_team_state())
    state_b = team_states.setdefault(team_b, _empty_team_state())

    row["team_a_tournament_matches_played"] = state_a["matches"]
    row["team_b_tournament_matches_played"] = state_b["matches"]
    row["tournament_points_diff"] = state_a["points"] - state_b["points"]
    row["tournament_goal_diff_diff"] = state_a["goal_diff"] - state_b["goal_diff"]

    return row


def predict_group_stage(
    model,
    group_features: pd.DataFrame,
) -> pd.DataFrame:
    """Predict group-stage matches sequentially.

    Important:
    This updates in-tournament features before every match:
    - team_a_tournament_matches_played
    - team_b_tournament_matches_played
    - tournament_points_diff
    - tournament_goal_diff_diff

    That prevents using all-zero static tournament features for matchday 2/3.
    """
    fixtures = group_features.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"])
    fixtures = fixtures.sort_values(["date", "match_id"]).reset_index(drop=True)

    for col in FEATURE_COLS:
        if col not in fixtures.columns:
            fixtures[col] = 0.0

    team_states: dict[str, dict] = {}
    rows = []

    from src.models.score_conversion import most_likely_score, win_draw_loss_probs

    for _, match in fixtures.iterrows():
        live_match = _apply_live_tournament_features(match, team_states)

        X = pd.DataFrame([live_match[FEATURE_COLS]]).fillna(0)
        pred = model.predict(X)

        lambda_a = float(pred[0, 0])
        lambda_b = float(pred[0, 1])

        goals_a, goals_b = most_likely_score(lambda_a, lambda_b)
        win_a, draw, win_b = win_draw_loss_probs(lambda_a, lambda_b)

        rows.append({
            "match_id": live_match.get("match_id"),
            "date": live_match.get("date"),
            "group": live_match["group"],
            "team_a": live_match["team_a"],
            "team_b": live_match["team_b"],
            "lambda_a": lambda_a,
            "lambda_b": lambda_b,
            "pred_goals_a": goals_a,
            "pred_goals_b": goals_b,
            "pred_score": f"{goals_a}-{goals_b}",
            "team_a_win_prob": win_a,
            "draw_prob": draw,
            "team_b_win_prob": win_b,
            "team_a_tournament_matches_played": live_match["team_a_tournament_matches_played"],
            "team_b_tournament_matches_played": live_match["team_b_tournament_matches_played"],
            "tournament_points_diff": live_match["tournament_points_diff"],
            "tournament_goal_diff_diff": live_match["tournament_goal_diff_diff"],
        })

        team_a = live_match["team_a"]
        team_b = live_match["team_b"]

        state_a = team_states.setdefault(team_a, _empty_team_state())
        state_b = team_states.setdefault(team_b, _empty_team_state())

        _update_team_state(state_a, goals_a, goals_b)
        _update_team_state(state_b, goals_b, goals_a)

    return pd.DataFrame(rows)

def build_knockout_from_group_predictions(
    group_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_matches = prepare_group_prediction_matches(group_predictions)
    standings = build_group_standings(group_matches)
    position_map = get_group_position_map(standings)

    r32_fixtures = build_round_of_32_fixtures(
        standings=standings,
        position_map=position_map,
    )

    validate_round_of_32(r32_fixtures)

    return standings, r32_fixtures


def simulate_knockout(
    model,
    model_df: pd.DataFrame,
    r32_fixtures: pd.DataFrame,
) -> pd.DataFrame:
    template = get_knockout_template()

    winner_map = {}
    loser_map = {}
    rows = []

    r32_map = {
        row["match_slot"]: (row["team_a"], row["team_b"])
        for _, row in r32_fixtures.iterrows()
    }

    for _, match in template.iterrows():
        round_name = match["round"]
        slot = match["match_slot"]

        if round_name == "R32":
            team_a, team_b = r32_map[slot]
        else:
            team_a = _resolve_progression_slot(match["team_a_slot"], winner_map, loser_map)
            team_b = _resolve_progression_slot(match["team_b_slot"], winner_map, loser_map)

        result = simulate_match(
            model=model,
            model_df=model_df,
            team_a=team_a,
            team_b=team_b,
            knockout=True,
        )

        winner_map[f"W_{slot}"] = result["winner"]
        loser_map[f"L_{slot}"] = result["loser"]

        rows.append({
            "round": round_name,
            "match_slot": slot,
            "team_a": team_a,
            "team_b": team_b,
            **result,
        })

    return pd.DataFrame(rows)


def _resolve_progression_slot(
    slot: str,
    winner_map: dict[str, str],
    loser_map: dict[str, str],
) -> str:
    slot = str(slot)

    if slot.startswith("W_"):
        return winner_map[slot]

    if slot.startswith("L_"):
        return loser_map[slot]

    raise ValueError(f"Unknown progression slot: {slot}")


def simulate_world_cup_2026(
    model,
    model_df: pd.DataFrame,
    group_features: pd.DataFrame,
) -> dict:
    group_predictions = predict_group_stage(model, group_features)
    standings, r32_fixtures = build_knockout_from_group_predictions(group_predictions)
    knockout_results = simulate_knockout(model, model_df, r32_fixtures)

    final = knockout_results[knockout_results["round"] == "FINAL"].iloc[0]
    third_place = knockout_results[knockout_results["round"] == "THIRD_PLACE"].iloc[0]

    return {
        "group_predictions": group_predictions,
        "standings": standings,
        "r32_fixtures": r32_fixtures,
        "knockout_results": knockout_results,
        "champion": final["winner"],
        "runner_up": final["loser"],
        "third_place": third_place["winner"],
    }
