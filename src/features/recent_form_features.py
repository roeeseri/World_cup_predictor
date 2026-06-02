"""Recent-form and match-activity feature helpers."""

from datetime import datetime

import numpy as np
import pandas as pd


WINDOW = 5


def _to_naive_datetime(value):
    """Convert timestamp-like value to timezone-naive pandas Timestamp."""
    ts = pd.to_datetime(value)

    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(None)

    return ts


def _team_matches_before(
    team: str,
    historical_matches: pd.DataFrame,
    cutoff_date,
) -> pd.DataFrame:
    """Return all matches involving team before cutoff_date."""
    cutoff_date = _to_naive_datetime(cutoff_date)

    match_dates = pd.to_datetime(historical_matches["date"], errors="coerce")

    if getattr(match_dates.dt, "tz", None) is not None:
        match_dates = match_dates.dt.tz_convert(None)

    matches = historical_matches[
        (
            (historical_matches["team_a"] == team) |
            (historical_matches["team_b"] == team)
        ) &
        (match_dates < cutoff_date)
    ].copy()

    return matches.sort_values("date")


def _team_match_view(team: str, matches: pd.DataFrame) -> pd.DataFrame:
    """Convert matches into team-centric rows."""
    rows = []

    for _, row in matches.iterrows():
        is_team_a = row["team_a"] == team

        goals_for = row["goals_a"] if is_team_a else row["goals_b"]
        goals_against = row["goals_b"] if is_team_a else row["goals_a"]

        opponent_elo = (
            row["rating_b_before"] if is_team_a else row["rating_a_before"]
        )

        rating_change = (
            row.get("rating_change_a", 0) if is_team_a else row.get("rating_change_b", 0)
        )

        if goals_for > goals_against:
            points = 3
        elif goals_for == goals_against:
            points = 1
        else:
            points = 0

        rows.append({
            "date": row["date"],
            "points": points,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "opponent_elo": opponent_elo,
            "rating_change": rating_change,
        })

    return pd.DataFrame(rows)


def _mean_or_zero(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _recent_stats(team: str, historical_matches: pd.DataFrame, cutoff_date) -> dict:
    matches = _team_matches_before(team, historical_matches, cutoff_date)
    team_view = _team_match_view(team, matches)

    elo_base = float(
        pd.concat([
            historical_matches["rating_a_before"],
            historical_matches["rating_b_before"],
        ]).mean()
    )

    if team_view.empty:
        return {
            "form_last5": 0.0,
            "weighted_goals_for_last5": 0.0,
            "weighted_goals_against_last5": 0.0,
            "opponent_strength_last5": elo_base,
            "rating_change_last5": 0.0,
            "matches_played_before": 0,
            "days_since_last_match": 60,
        }

    recent = team_view.tail(WINDOW)

    weighted_goals_for = [
        row["goals_for"] * (row["opponent_elo"] / elo_base)
        for _, row in recent.iterrows()
    ]

    weighted_goals_against = [
        row["goals_against"] * (elo_base / row["opponent_elo"])
        for _, row in recent.iterrows()
        if row["opponent_elo"] != 0
    ]

    cutoff_date = _to_naive_datetime(cutoff_date)
    last_match_date = _to_naive_datetime(team_view.iloc[-1]["date"])

    days_since_last_match = (cutoff_date - last_match_date).days
    days_since_last_match = max(0, min(days_since_last_match, 60))

    return {
        "form_last5": _mean_or_zero(recent["points"]),
        "weighted_goals_for_last5": _mean_or_zero(weighted_goals_for),
        "weighted_goals_against_last5": _mean_or_zero(weighted_goals_against),
        "opponent_strength_last5": _mean_or_zero(recent["opponent_elo"]),
        "rating_change_last5": _mean_or_zero(recent["rating_change"]),
        "matches_played_before": int(len(team_view)),
        "days_since_last_match": int(days_since_last_match),
    }

def compute_recent_form_features(
    team_a: str,
    team_b: str,
    historical_matches: pd.DataFrame,
    cutoff_date,
) -> dict:
    """Compute compact recent-form and activity features for a match."""
    stats_a = _recent_stats(team_a, historical_matches, cutoff_date)
    stats_b = _recent_stats(team_b, historical_matches, cutoff_date)

    days_diff = (
        stats_a["days_since_last_match"] -
        stats_b["days_since_last_match"]
    )

    days_diff = max(-60, min(days_diff, 60))

    return {
        "form_diff_last5": stats_a["form_last5"] - stats_b["form_last5"],
        "weighted_goals_for_diff_last5": (
            stats_a["weighted_goals_for_last5"] -
            stats_b["weighted_goals_for_last5"]
        ),
        "weighted_goals_against_diff_last5": (
            stats_a["weighted_goals_against_last5"] -
            stats_b["weighted_goals_against_last5"]
        ),
        "opponent_strength_diff_last5": (
            stats_a["opponent_strength_last5"] -
            stats_b["opponent_strength_last5"]
        ),
        "rating_change_diff_last5": (
            stats_a["rating_change_last5"] -
            stats_b["rating_change_last5"]
        ),
        "team_a_matches_played_before": stats_a["matches_played_before"],
        "team_b_matches_played_before": stats_b["matches_played_before"],
        "team_a_days_since_last_match": stats_a["days_since_last_match"],
        "team_b_days_since_last_match": stats_b["days_since_last_match"],
        "days_since_match_diff": days_diff,
    }
