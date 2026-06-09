from __future__ import annotations

import pandas as pd

from src.features.feature_columns import FEATURE_COLS
from src.models.score_conversion import most_likely_score, win_draw_loss_probs
from src.state.elo import compute_elo_update


def empty_state() -> dict:
    return {"matches": 0, "points": 0, "goals_for": 0, "goals_against": 0, "goal_diff": 0}


def points_for(gf: int, ga: int) -> int:
    return 3 if gf > ga else 1 if gf == ga else 0


def update_state(team_states: dict[str, dict], team_a: str, team_b: str, goals_a: int, goals_b: int) -> None:
    for team, gf, ga in [(team_a, goals_a, goals_b), (team_b, goals_b, goals_a)]:
        st = team_states.setdefault(team, empty_state())
        st["matches"] += 1
        st["points"] += points_for(gf, ga)
        st["goals_for"] += gf
        st["goals_against"] += ga
        st["goal_diff"] = st["goals_for"] - st["goals_against"]


def build_rating_profiles(group_features: pd.DataFrame) -> dict[str, dict]:
    profiles = {}

    for _, row in group_features.iterrows():
        profiles.setdefault(row["team_a"], {
            "rating": float(row["rating_a_before"]),
            "matches_before": float(row["team_a_matches_played_before"]),
            "days_since": float(row["team_a_days_since_last_match"]),
        })
        profiles.setdefault(row["team_b"], {
            "rating": float(row["rating_b_before"]),
            "matches_before": float(row["team_b_matches_played_before"]),
            "days_since": float(row["team_b_days_since_last_match"]),
        })

    ratings = {team: p["rating"] for team, p in profiles.items()}
    derived_ranks = {
        team: rank + 1
        for rank, (team, _) in enumerate(sorted(ratings.items(), key=lambda x: x[1], reverse=True))
    }

    for team in profiles:
        profiles[team]["derived_rank"] = derived_ranks[team]

    return profiles


def build_knockout_feature_row(
    group_features: pd.DataFrame,
    team_states: dict[str, dict],
    team_a: str,
    team_b: str,
    current_elo: dict[str, float] | None = None,
) -> pd.DataFrame:
    profiles = build_rating_profiles(group_features)

    if team_a not in profiles:
        raise ValueError(f"Missing profile for {team_a}")
    if team_b not in profiles:
        raise ValueError(f"Missing profile for {team_b}")

    pa = profiles[team_a]
    pb = profiles[team_b]

    # Use live (updated) ELO if available, otherwise fall back to pre-tournament
    if current_elo:
        pa = dict(pa)
        pb = dict(pb)
        pa["rating"] = current_elo.get(team_a, pa["rating"])
        pb["rating"] = current_elo.get(team_b, pb["rating"])
        all_ratings = {t: current_elo.get(t, p["rating"]) for t, p in profiles.items()}
        sorted_teams = sorted(all_ratings.items(), key=lambda x: -x[1])
        live_ranks = {t: r + 1 for r, (t, _) in enumerate(sorted_teams)}
        pa["derived_rank"] = live_ranks.get(team_a, pa["derived_rank"])
        pb["derived_rank"] = live_ranks.get(team_b, pb["derived_rank"])

    sa = team_states.setdefault(team_a, empty_state())
    sb = team_states.setdefault(team_b, empty_state())

    features = {
        "rank_diff": pa["derived_rank"] - pb["derived_rank"],
        "elo_diff": pa["rating"] - pb["rating"],
        "rating_a_before": pa["rating"],
        "rating_b_before": pb["rating"],

        "avg_player_value_diff": 0.0,
        "opponent_strength_diff_last5": 0.0,
        "weighted_goals_for_diff_last5": 0.0,
        "weighted_goals_against_diff_last5": 0.0,
        "market_value_rel_mean_diff": 0.0,
        "rating_change_diff_last5": 0.0,
        "defender_share_diff": 0.0,
        "goalkeeper_share_diff": 0.0,

        "team_a_matches_played_before": pa["matches_before"],
        "team_b_matches_played_before": pb["matches_before"],
        "team_a_days_since_last_match": pa["days_since"],
        "team_b_days_since_last_match": pb["days_since"],

        "tournament_goal_diff_diff": sa["goal_diff"] - sb["goal_diff"],
        "tournament_points_diff": sa["points"] - sb["points"],
        "team_a_tournament_matches_played": sa["matches"],
        "team_b_tournament_matches_played": sb["matches"],
    }

    row = pd.DataFrame([features])
    return row[FEATURE_COLS].fillna(0)


def simulate_match(
    model,
    group_features: pd.DataFrame,
    team_states: dict[str, dict],
    team_a: str,
    team_b: str,
    knockout: bool = True,
    current_elo: dict[str, float] | None = None,
) -> dict:
    X = build_knockout_feature_row(group_features, team_states, team_a, team_b, current_elo)
    pred = model.predict(X)[0]

    lambda_a = float(pred[0])
    lambda_b = float(pred[1])

    goals_a, goals_b = most_likely_score(lambda_a, lambda_b)
    win_a, draw, win_b = win_draw_loss_probs(lambda_a, lambda_b)

    if goals_a > goals_b:
        winner, loser = team_a, team_b
    elif goals_b > goals_a:
        winner, loser = team_b, team_a
    else:
        winner = team_a if win_a >= win_b else team_b
        loser = team_b if winner == team_a else team_a

    update_state(team_states, team_a, team_b, goals_a, goals_b)

    if current_elo is not None:
        ra = current_elo.get(team_a, 1500.0)
        rb = current_elo.get(team_b, 1500.0)
        delta_a, delta_b = compute_elo_update(
            ra, rb, goals_a, goals_b,
            competition="FIFA World Cup", team_a=team_a, team_b=team_b,
        )
        current_elo[team_a] = ra + delta_a
        current_elo[team_b] = rb + delta_b

    return {
        "team_a": team_a,
        "team_b": team_b,
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "goals_a": goals_a,
        "goals_b": goals_b,
        "pred_score": f"{goals_a}-{goals_b}",
        "team_a_win_prob": win_a,
        "draw_prob": draw,
        "team_b_win_prob": win_b,
        "winner": winner,
        "loser": loser,
    }
