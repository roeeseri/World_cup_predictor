"""Load latest available Elo ratings and rankings from historical ELO files."""

from pathlib import Path

import pandas as pd

from src.data.load_results import load_historical_results


def load_current_elo_ratings(
    raw_dir: str | Path = "data/raw",
    as_of_date: str | None = None,
    start_year: int = 2004,
) -> dict[str, float]:
    """Return latest known Elo rating per team.

    Uses the latest match available before or on as_of_date.
    """
    df = load_historical_results(raw_dir=raw_dir, start_year=start_year)

    if as_of_date is not None:
        as_of_date = pd.to_datetime(as_of_date)
        df = df[df["date"] <= as_of_date].copy()

    if df.empty:
        raise ValueError("No ELO data available for the requested date.")

    events = []

    for _, row in df.iterrows():
        events.append({
            "team": row["team_a"],
            "date": row["date"],
            "rating": row["rating_a"],
        })
        events.append({
            "team": row["team_b"],
            "date": row["date"],
            "rating": row["rating_b"],
        })

    ratings_df = pd.DataFrame(events)
    ratings_df = ratings_df.sort_values("date")

    latest = ratings_df.groupby("team").tail(1)

    return dict(zip(latest["team"], latest["rating"].astype(float)))


def load_current_rankings(
    raw_dir: str | Path = "data/raw",
    as_of_date: str | None = None,
    start_year: int = 2004,
) -> dict[str, int]:
    """Return latest known ranking per team.

    Uses the latest match available before or on as_of_date.
    """
    df = load_historical_results(raw_dir=raw_dir, start_year=start_year)

    if as_of_date is not None:
        as_of_date = pd.to_datetime(as_of_date)
        df = df[df["date"] <= as_of_date].copy()

    if df.empty:
        raise ValueError("No ranking data available for the requested date.")

    events = []

    for _, row in df.iterrows():
        events.append({
            "team": row["team_a"],
            "date": row["date"],
            "rank": row["rank_a"],
        })
        events.append({
            "team": row["team_b"],
            "date": row["date"],
            "rank": row["rank_b"],
        })

    ranks_df = pd.DataFrame(events)
    ranks_df = ranks_df.sort_values("date")

    latest = ranks_df.groupby("team").tail(1)

    return dict(zip(latest["team"], latest["rank"].astype(int)))
