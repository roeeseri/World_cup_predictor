"""Live tournament state: initialization, result recording, and simulation."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from src.state.elo import compute_elo_update


# ---------------------------------------------------------------------------
# Rankings helpers
# ---------------------------------------------------------------------------

def derive_rankings_from_elo(elo_ratings: dict[str, float]) -> dict[str, int]:
    """Rank all teams by current ELO (highest = rank 1)."""
    sorted_teams = sorted(elo_ratings.items(), key=lambda x: -x[1])
    return {team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)}


# ---------------------------------------------------------------------------
# State initialization
# ---------------------------------------------------------------------------

def _extract_elo_ratings(historical_matches: pd.DataFrame) -> dict[str, float]:
    """Get the most recent post-match ELO rating for every team."""
    df = historical_matches.sort_values("date")
    ratings: dict[str, float] = {}

    for _, row in df.iterrows():
        # rating_a / rating_b are post-match ratings in the ELO CSV schema
        ratings[row["team_a"]] = float(row["rating_a"])
        ratings[row["team_b"]] = float(row["rating_b"])

    return ratings


def initialize_live_state(
    historical_matches: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> dict:
    """Create a fresh LiveTournamentState from historical data and fixtures.

    Any completion data in *fixtures* is ignored — call
    ``apply_results_from_csv`` afterwards to replay known results.
    """
    from src.features.team_names import TEAM_NAME_MAP

    elo_ratings = _extract_elo_ratings(historical_matches)

    # Fixture files use raw ELO names (e.g. "Korea Republic", "USA").
    # Historical data uses canonical names ("South Korea", "United States").
    # Populate aliases so lookups succeed for both naming conventions.
    for alias, canonical in TEAM_NAME_MAP.items():
        if canonical in elo_ratings and alias not in elo_ratings:
            elo_ratings[alias] = elo_ratings[canonical]

    rankings = derive_rankings_from_elo(elo_ratings)

    from src.features.team_names import normalize_team_name
    from src.features.tournament_state_features import initialize_team_states

    # Use canonical names for team_states so historical lookups work correctly
    wc_teams_canonical = list({
        normalize_team_name(t)
        for t in set(fixtures["team_a"]) | set(fixtures["team_b"])
    })
    team_states = initialize_team_states(wc_teams_canonical)

    # Store fixtures with completion reset — results get applied separately
    clean_fixtures = fixtures.copy()
    clean_fixtures["goals_a"] = float("nan")
    clean_fixtures["goals_b"] = float("nan")
    clean_fixtures["is_completed"] = False

    return {
        "historical_matches": historical_matches.copy(),
        "elo_ratings": elo_ratings,
        "rankings": rankings,
        "team_states": team_states,
        "fixtures": clean_fixtures,
    }


# ---------------------------------------------------------------------------
# Recording a real match result
# ---------------------------------------------------------------------------

def record_match_result(
    state: dict,
    match_id: int,
    goals_a: int,
    goals_b: int,
) -> dict:
    """Return a new state with the match result applied.

    The original *state* dict is deep-copied so the caller's reference is
    never mutated.  This makes simulation branches safe.
    """
    state = copy.deepcopy(state)
    fixtures = state["fixtures"]

    mask = fixtures["match_id"] == int(match_id)
    if not mask.any():
        raise ValueError(f"match_id {match_id} not found in fixtures")

    fixture = fixtures.loc[mask].iloc[0]

    if bool(fixture.get("is_completed", False)):
        return state  # already recorded — idempotent

    from src.features.team_names import normalize_team_name

    team_a: str = fixture["team_a"]           # fixture/display name
    team_b: str = fixture["team_b"]
    team_a_canon: str = normalize_team_name(team_a)   # canonical data name
    team_b_canon: str = normalize_team_name(team_b)
    match_date = pd.to_datetime(fixture["date"]).tz_localize(None)  # naive
    location: str = str(fixture.get("location", "neutral"))
    competition = "FIFA World Cup"

    goals_a = int(goals_a)
    goals_b = int(goals_b)

    # Pre-match ratings (elo_ratings has both alias and canonical keys)
    rating_a_before = state["elo_ratings"].get(team_a_canon,
                       state["elo_ratings"].get(team_a, 1500.0))
    rating_b_before = state["elo_ratings"].get(team_b_canon,
                       state["elo_ratings"].get(team_b, 1500.0))
    rank_a_before = state["rankings"].get(team_a_canon,
                    state["rankings"].get(team_a, 0))
    rank_b_before = state["rankings"].get(team_b_canon,
                    state["rankings"].get(team_b, 0))

    stage = str(fixture.get("stage", "GROUPS"))
    knockout = stage.upper() not in {"GROUPS", "GROUP", "GROUP_STAGE"}

    delta_a, delta_b = compute_elo_update(
        rating_a=rating_a_before,
        rating_b=rating_b_before,
        goals_a=goals_a,
        goals_b=goals_b,
        competition=competition,
        team_a=team_a,
        team_b=team_b,
        location=location,
        stage=stage,
        knockout=knockout,
    )

    rating_a_after = rating_a_before + delta_a
    rating_b_after = rating_b_before + delta_b

    # Build new historical row using canonical names so form features find it
    new_row = {
        "date": match_date,
        "team_a": team_a_canon,
        "team_b": team_b_canon,
        "goals_a": goals_a,
        "goals_b": goals_b,
        "competition": competition,
        "location": location,
        "rating_change_a": delta_a,
        "rating_change_b": delta_b,
        "rating_a": rating_a_after,
        "rating_b": rating_b_after,
        "rating_a_before": rating_a_before,
        "rating_b_before": rating_b_before,
        "rank_a": rank_a_before,
        "rank_b": rank_b_before,
        "rank_a_before": rank_a_before,
        "rank_b_before": rank_b_before,
        "rank_change_a": 0,
        "rank_change_b": 0,
        "elo_diff": rating_a_before - rating_b_before,
        "rank_diff": rank_a_before - rank_b_before,
        "source_file": "live_2026",
        "tournament_year": 2026,
        "tournament_key": "FIFA World Cup_2026",
    }

    state["historical_matches"] = pd.concat(
        [state["historical_matches"], pd.DataFrame([new_row])],
        ignore_index=True,
    )

    # Update live ELO and rankings — update both canonical and alias keys
    state["elo_ratings"][team_a_canon] = rating_a_after
    state["elo_ratings"][team_b_canon] = rating_b_after
    if team_a != team_a_canon:
        state["elo_ratings"][team_a] = rating_a_after
    if team_b != team_b_canon:
        state["elo_ratings"][team_b] = rating_b_after
    state["rankings"] = derive_rankings_from_elo(state["elo_ratings"])

    # Update tournament table using canonical names
    from src.features.tournament_state_features import update_state_after_match
    state["team_states"] = update_state_after_match(
        state["team_states"], team_a_canon, team_b_canon, goals_a, goals_b
    )

    # Mark fixture as completed
    state["fixtures"].loc[mask, "goals_a"] = goals_a
    state["fixtures"].loc[mask, "goals_b"] = goals_b
    state["fixtures"].loc[mask, "is_completed"] = True

    return state


# ---------------------------------------------------------------------------
# Loading real results from the updates CSV
# ---------------------------------------------------------------------------

def apply_results_from_csv(
    state: dict,
    csv_path: str | Path = "data/raw/world_cup_updates/all_world_cup_2026_updates.csv",
) -> dict:
    """Apply any completed-match rows from the updates CSV to *state*.

    Matches are identified by team_a + team_b + date.  Already-recorded
    matches are skipped silently.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return state

    updates = pd.read_csv(csv_path)
    if updates.empty:
        return state

    updates["date"] = pd.to_datetime(updates["date"], errors="coerce")

    fixtures = state["fixtures"].copy()

    for _, row in updates.iterrows():
        if pd.isna(row.get("goals_a")) or pd.isna(row.get("goals_b")):
            continue

        # Match by team names only (each pair plays once in group stage)
        match = fixtures[
            (fixtures["team_a"] == row["team_a"])
            & (fixtures["team_b"] == row["team_b"])
        ]
        if not match.empty:
            state = record_match_result(
                state,
                int(match.iloc[0]["match_id"]),
                int(row["goals_a"]),
                int(row["goals_b"]),
            )
            continue

        # Try reversed order
        match = fixtures[
            (fixtures["team_a"] == row["team_b"])
            & (fixtures["team_b"] == row["team_a"])
        ]
        if not match.empty:
            state = record_match_result(
                state,
                int(match.iloc[0]["match_id"]),
                int(row["goals_b"]),
                int(row["goals_a"]),
            )

    return state


# ---------------------------------------------------------------------------
# Simulate forward from current true state
# ---------------------------------------------------------------------------

def simulate_forward(
    state: dict,
    model,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    feature_fn=None,
    score_fn=None,
) -> dict:
    """Return a deep copy of *state* with all unplayed fixtures simulated.

    The original *state* is never mutated.
    feature_fn: callable with same signature as build_pre_match_features (default: V4)
    score_fn:   callable (lambda_a, lambda_b) -> (goals_a, goals_b) (default: most_likely_score V4)
    """
    from src.features.build_features import build_pre_match_features
    from src.models.score_conversion import most_likely_score

    if feature_fn is None:
        feature_fn = build_pre_match_features
    if score_fn is None:
        score_fn = most_likely_score

    sim = copy.deepcopy(state)

    unplayed = (
        sim["fixtures"][~sim["fixtures"]["is_completed"]]
        .sort_values("date")
        .reset_index(drop=True)
    )

    for _, fixture in unplayed.iterrows():
        team_a: str = fixture["team_a"]
        team_b: str = fixture["team_b"]
        match_date = pd.to_datetime(fixture["date"])
        match_id = int(fixture["match_id"])

        try:
            feature_row = feature_fn(
                team_a=team_a,
                team_b=team_b,
                match_date=match_date,
                team_states=sim["team_states"],
                historical_matches=sim["historical_matches"],
                market_values=market_values,
                position_values=position_values,
                elo_ratings=sim["elo_ratings"],
                rankings=sim["rankings"],
            )
            pred = model.predict(feature_row.fillna(0))
            lambda_a = float(pred[0, 0])
            lambda_b = float(pred[0, 1])
            goals_a, goals_b = score_fn(lambda_a, lambda_b)
        except Exception:
            goals_a, goals_b = 1, 1

        sim = record_match_result(sim, match_id, goals_a, goals_b)

    return sim
