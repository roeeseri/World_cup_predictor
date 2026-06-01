"""Load World Cup 2026 fixtures."""

from pathlib import Path

import pandas as pd


def load_tournament_fixtures(
    path: str | Path = "data/raw/fixtures/world_cup_2026_group_stage.csv",
) -> pd.DataFrame:
    """Load cleaned tournament fixtures."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing fixtures file: {path}")

    df = pd.read_csv(path)

    required_cols = [
        "match_id",
        "date",
        "team_a",
        "team_b",
        "group",
        "stage",
        "matchday",
        "location",
        "goals_a",
        "goals_b",
        "is_completed",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing fixture columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["match_id"] = df["match_id"].astype(int)
    df["matchday"] = df["matchday"].astype(int)

    df["is_completed"] = (
        df["goals_a"].notna() &
        df["goals_b"].notna()
    )

    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

    return df


def get_upcoming_matches(
    fixtures: pd.DataFrame,
    completed_match_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Return fixtures that are not completed yet."""
    completed_match_ids = completed_match_ids or []

    upcoming = fixtures[
        (~fixtures["match_id"].isin(completed_match_ids)) &
        (~fixtures["is_completed"])
    ].copy()

    return upcoming.sort_values(["date", "match_id"]).reset_index(drop=True)
