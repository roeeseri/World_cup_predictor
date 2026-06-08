from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.feature_columns import FEATURE_COLS
from src.tournament.build_knockout import build_round_of_32_fixtures, validate_round_of_32
from src.tournament.group_standings import build_group_standings, get_group_position_map
from src.tournament.match_simulation import simulate_match


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


def predict_group_stage(
    model,
    group_features: pd.DataFrame,
) -> pd.DataFrame:
    """Predict group-stage matches sequentially with live tournament features."""
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

        rows.append(
            {
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
                "team_a_tournament_matches_played": live_match[
                    "team_a_tournament_matches_played"
                ],
                "team_b_tournament_matches_played": live_match[
                    "team_b_tournament_matches_played"
                ],
                "tournament_points_diff": live_match["tournament_points_diff"],
                "tournament_goal_diff_diff": live_match[
                    "tournament_goal_diff_diff"
                ],
            }
        )

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
    matches = prepare_group_prediction_matches(group_predictions)
    standings = build_group_standings(matches)
    position_map = get_group_position_map(standings)
    r32_fixtures = build_round_of_32_fixtures(standings, position_map)
    validate_round_of_32(r32_fixtures)

    return standings, r32_fixtures


def _simulate_round(
    model,
    group_features: pd.DataFrame,
    team_states: dict[str, dict],
    round_name: str,
    fixtures: list[tuple[str, str, str]],
) -> tuple[list[dict], dict[str, str], dict[str, str]]:
    rows = []
    winners = {}
    losers = {}

    for match_slot, team_a, team_b in fixtures:
        result = simulate_match(
            model=model,
            group_features=group_features,
            team_states=team_states,
            team_a=team_a,
            team_b=team_b,
            knockout=True,
        )

        result["round"] = round_name
        result["match_slot"] = match_slot

        rows.append(result)
        winners[match_slot] = result["winner"]
        losers[match_slot] = result["loser"]

    return rows, winners, losers


def simulate_knockout(
    model,
    group_features: pd.DataFrame,
    r32_fixtures: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    team_states: dict[str, dict] = {}

    r32_games = [
        (row["match_slot"], row["team_a"], row["team_b"])
        for _, row in r32_fixtures.iterrows()
    ]

    r32_rows, r32_winners, r32_losers = _simulate_round(
        model=model,
        group_features=group_features,
        team_states=team_states,
        round_name="R32",
        fixtures=r32_games,
    )
    rows.extend(r32_rows)

    r16_games = [
        ("R16_01", r32_winners["R32_01"], r32_winners["R32_02"]),
        ("R16_02", r32_winners["R32_03"], r32_winners["R32_04"]),
        ("R16_03", r32_winners["R32_05"], r32_winners["R32_06"]),
        ("R16_04", r32_winners["R32_07"], r32_winners["R32_08"]),
        ("R16_05", r32_winners["R32_09"], r32_winners["R32_10"]),
        ("R16_06", r32_winners["R32_11"], r32_winners["R32_12"]),
        ("R16_07", r32_winners["R32_13"], r32_winners["R32_14"]),
        ("R16_08", r32_winners["R32_15"], r32_winners["R32_16"]),
    ]

    r16_rows, r16_winners, r16_losers = _simulate_round(
        model=model,
        group_features=group_features,
        team_states=team_states,
        round_name="R16",
        fixtures=r16_games,
    )
    rows.extend(r16_rows)

    qf_games = [
        ("QF_01", r16_winners["R16_01"], r16_winners["R16_02"]),
        ("QF_02", r16_winners["R16_03"], r16_winners["R16_04"]),
        ("QF_03", r16_winners["R16_05"], r16_winners["R16_06"]),
        ("QF_04", r16_winners["R16_07"], r16_winners["R16_08"]),
    ]

    qf_rows, qf_winners, qf_losers = _simulate_round(
        model=model,
        group_features=group_features,
        team_states=team_states,
        round_name="QF",
        fixtures=qf_games,
    )
    rows.extend(qf_rows)

    sf_games = [
        ("SF_01", qf_winners["QF_01"], qf_winners["QF_02"]),
        ("SF_02", qf_winners["QF_03"], qf_winners["QF_04"]),
    ]

    sf_rows, sf_winners, sf_losers = _simulate_round(
        model=model,
        group_features=group_features,
        team_states=team_states,
        round_name="SF",
        fixtures=sf_games,
    )
    rows.extend(sf_rows)

    final_games = [
        ("FINAL", sf_winners["SF_01"], sf_winners["SF_02"]),
        ("THIRD_PLACE", sf_losers["SF_01"], sf_losers["SF_02"]),
    ]

    final_rows, _, _ = _simulate_round(
        model=model,
        group_features=group_features,
        team_states=team_states,
        round_name="FINAL_STAGE",
        fixtures=final_games,
    )

    for row in final_rows:
        if row["match_slot"] == "FINAL":
            row["round"] = "FINAL"
        elif row["match_slot"] == "THIRD_PLACE":
            row["round"] = "THIRD_PLACE"

    rows.extend(final_rows)

    return pd.DataFrame(rows)


def simulate_world_cup_2026(
    model,
    model_df: pd.DataFrame,
    group_features: pd.DataFrame,
    output_dir: str | Path = "outputs/evaluation/world_cup_2026_simulation",
) -> dict:
    group_predictions = predict_group_stage(model, group_features)
    standings, r32_fixtures = build_knockout_from_group_predictions(group_predictions)
    knockout_results = simulate_knockout(model, group_features, r32_fixtures)

    final_row = knockout_results[knockout_results["round"] == "FINAL"].iloc[0]
    third_row = knockout_results[knockout_results["round"] == "THIRD_PLACE"].iloc[0]

    results = {
        "group_predictions": group_predictions,
        "standings": standings,
        "r32_fixtures": r32_fixtures,
        "knockout_results": knockout_results,
        "champion": final_row["winner"],
        "runner_up": final_row["loser"],
        "third_place": third_row["winner"],
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    group_predictions.to_csv(output_path / "group_predictions.csv", index=False)
    standings.to_csv(output_path / "group_standings.csv", index=False)
    r32_fixtures.to_csv(output_path / "round_of_32_fixtures.csv", index=False)
    knockout_results.to_csv(output_path / "knockout_results.csv", index=False)

    return results
