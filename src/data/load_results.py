"""Load historical ELO match results for feature computation."""

from pathlib import Path

import pandas as pd


def load_historical_results(
    raw_dir: str | Path = "data/raw",
    start_year: int = 2004,
) -> pd.DataFrame:
    """Load yearly ELO match result CSV files into one chronological DataFrame.

    Args:
        raw_dir: Directory containing elo_YEAR_results.csv files.
        start_year: First year to keep.

    Returns:
        Clean historical results DataFrame sorted by date.
    """
    raw_dir = Path(raw_dir)

    elo_files = sorted(raw_dir.glob("elo_*_results.csv"))

    if not elo_files:
        raise FileNotFoundError(f"No ELO files found in {raw_dir}")

    dfs = []

    for file in elo_files:
        temp = pd.read_csv(file)
        temp["source_file"] = file.name
        dfs.append(temp)

    df = pd.concat(dfs, ignore_index=True)

    required_cols = [
        "date",
        "team_a",
        "team_b",
        "goals_a",
        "goals_b",
        "competition",
        "location",
        "rating_a",
        "rating_b",
        "rating_change_a",
        "rating_change_b",
        "rank_a",
        "rank_b",
        "rank_change_a",
        "rank_change_b",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required ELO columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(
        subset=[
            "date",
            "team_a",
            "team_b",
            "goals_a",
            "goals_b",
        ]
    ).copy()

    df = df[df["date"].dt.year >= start_year].copy()

    df["goals_a"] = df["goals_a"].astype(int)
    df["goals_b"] = df["goals_b"].astype(int)

    df["rating_a_before"] = df["rating_a"] - df["rating_change_a"]
    df["rating_b_before"] = df["rating_b"] - df["rating_change_b"]

    df["rank_a_before"] = df["rank_a"] - df["rank_change_a"]
    df["rank_b_before"] = df["rank_b"] - df["rank_change_b"]

    df["elo_diff"] = df["rating_a_before"] - df["rating_b_before"]
    df["rank_diff"] = df["rank_a_before"] - df["rank_b_before"]

    df["tournament_year"] = df["date"].dt.year
    df["tournament_key"] = (
        df["competition"].astype(str)
        + "_"
        + df["tournament_year"].astype(str)
    )

    df = df.sort_values("date").reset_index(drop=True)

    return df
