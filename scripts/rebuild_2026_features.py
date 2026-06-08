from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.load_fixtures import load_tournament_fixtures
from src.features.build_features import build_pre_match_features
from src.features.feature_columns import FEATURE_COLS
from src.features.market_value_features import load_market_values
from src.features.position_value_features import load_position_values
from src.features.team_names import normalize_team_name
from src.features.tournament_state_features import (
    initialize_team_states,
    update_state_after_match,
)


RAW_RESULTS_PATHS = [
    "data/raw/elo_2004_results.csv",
    "data/raw/elo_2005_results.csv",
    "data/raw/elo_2006_results.csv",
    "data/raw/elo_2007_results.csv",
    "data/raw/elo_2008_results.csv",
    "data/raw/elo_2009_results.csv",
    "data/raw/elo_2010_results.csv",
    "data/raw/elo_2011_results.csv",
    "data/raw/elo_2012_results.csv",
    "data/raw/elo_2013_results.csv",
    "data/raw/elo_2014_results.csv",
    "data/raw/elo_2015_results.csv",
    "data/raw/elo_2016_results.csv",
    "data/raw/elo_2017_results.csv",
    "data/raw/elo_2018_results.csv",
    "data/raw/elo_2019_results.csv",
    "data/raw/elo_2020_results.csv",
    "data/raw/elo_2021_results.csv",
    "data/raw/elo_2022_results.csv",
    "data/raw/elo_2023_results.csv",
    "data/raw/elo_2024_results.csv",
    "data/raw/elo_2025_results.csv",
    "data/raw/elo_2026_results.csv",
]

FIXTURES_PATH = "data/raw/fixtures/world_cup_2026_group_stage.csv"
OUTPUT_PATH = "data/processed/world_cup_2026_group_stage_features.csv"


def load_historical_matches() -> pd.DataFrame:
    frames = []

    for path in RAW_RESULTS_PATHS:
        p = Path(path)
        if not p.exists():
            print(f"Skipping missing raw file: {path}")
            continue
        frames.append(pd.read_csv(p))

    if not frames:
        raise FileNotFoundError("No raw ELO result files found.")

    df = pd.concat(frames, ignore_index=True)

    rename_map = {
        "goals_A": "goals_a",
        "goals_B": "goals_b",
        "rating_A": "rating_a",
        "rating_B": "rating_b",
        "rank_A": "rank_a",
        "rank_B": "rank_b",
    }
    df = df.rename(columns=rename_map)

    required = [
        "date",
        "team_a",
        "team_b",
        "goals_a",
        "goals_b",
        "rating_a",
        "rating_b",
        "rank_a",
        "rank_b",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"historical matches missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date", "team_a", "team_b"])

    df["team_a"] = df["team_a"].map(normalize_team_name)
    df["team_b"] = df["team_b"].map(normalize_team_name)

    # In our feature code, rating_a_before/rating_b_before are expected.
    # The raw ELO files store current pre-match rating columns as rating_a/rating_b.
    df["rating_a_before"] = pd.to_numeric(df["rating_a"], errors="coerce")
    df["rating_b_before"] = pd.to_numeric(df["rating_b"], errors="coerce")

    for col in [
        "goals_a",
        "goals_b",
        "rating_a_before",
        "rating_b_before",
        "rank_a",
        "rank_b",
        "rating_change_a",
        "rating_change_b",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df.sort_values(["date", "team_a", "team_b"]).reset_index(drop=True)


def latest_team_ratings_and_ranks(historical: pd.DataFrame) -> tuple[dict[str, float], dict[str, int]]:
    rows = []

    for _, row in historical.iterrows():
        rows.append({
            "team": row["team_a"],
            "date": row["date"],
            "rating": row["rating_a_before"],
            "rank": row["rank_a"],
        })
        rows.append({
            "team": row["team_b"],
            "date": row["date"],
            "rating": row["rating_b_before"],
            "rank": row["rank_b"],
        })

    team_df = pd.DataFrame(rows)
    team_df = team_df.dropna(subset=["team", "date"])
    latest = team_df.sort_values("date").groupby("team", as_index=False).tail(1)

    ratings = dict(zip(latest["team"], latest["rating"]))
    rankings = dict(zip(latest["team"], latest["rank"].astype(int)))

    return ratings, rankings


def completed_fixture_states(fixtures: pd.DataFrame) -> dict:
    teams = sorted(set(fixtures["team_a"]) | set(fixtures["team_b"]))
    states = initialize_team_states(teams)

    completed = fixtures[fixtures["is_completed"]].sort_values(["date", "match_id"])

    for _, row in completed.iterrows():
        update_state_after_match(
            states,
            row["team_a"],
            row["team_b"],
            int(row["goals_a"]),
            int(row["goals_b"]),
        )

    return states


def main() -> None:
    print("Loading historical matches...")
    historical = load_historical_matches()

    print("Loading fixtures...")
    fixtures = load_tournament_fixtures(FIXTURES_PATH)
    fixtures["team_a"] = fixtures["team_a"].map(normalize_team_name)
    fixtures["team_b"] = fixtures["team_b"].map(normalize_team_name)

    print("Loading market/position values...")
    market_values = load_market_values()
    position_values = load_position_values()

    print("Building latest ratings/rankings...")
    elo_ratings, rankings = latest_team_ratings_and_ranks(historical)

    print("Building tournament states from completed fixtures...")
    team_states = completed_fixture_states(fixtures)

    rows = []

    for _, match in fixtures.iterrows():
        feature_row = build_pre_match_features(
            team_a=match["team_a"],
            team_b=match["team_b"],
            match_date=match["date"],
            team_states=team_states,
            historical_matches=historical,
            market_values=market_values,
            position_values=position_values,
            elo_ratings=elo_ratings,
            rankings=rankings,
        )

        feature_row.insert(0, "match_id", int(match["match_id"]))
        feature_row.insert(1, "date", match["date"])
        feature_row.insert(2, "team_a", match["team_a"])
        feature_row.insert(3, "team_b", match["team_b"])
        feature_row.insert(4, "group", match["group"])

        rows.append(feature_row)

    result = pd.concat(rows, ignore_index=True)

    output_cols = ["match_id", "date", "team_a", "team_b", "group"] + FEATURE_COLS
    result = result[output_cols].copy()

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Shape:", result.shape)
    print("Features:", len(FEATURE_COLS))
    print("Missing:", [c for c in FEATURE_COLS if c not in result.columns])
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()
