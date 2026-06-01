"""Build 2026 World Cup group-stage feature rows."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_fixtures import load_tournament_fixtures
from src.data.load_results import load_historical_results
from src.data.load_elo import load_current_elo_ratings, load_current_rankings
from src.features.market_value_features import load_market_values
from src.features.tournament_state_features import initialize_team_states
from src.features.build_fixture_features import build_features_for_fixtures


OUTPUT_PATH = Path("data/processed/world_cup_2026_group_stage_features.csv")


def main() -> None:
    fixtures = load_tournament_fixtures()
    historical_matches = load_historical_results()
    market_values = load_market_values()

    elo_ratings = load_current_elo_ratings(as_of_date="2026-05-16")
    rankings = load_current_rankings(as_of_date="2026-05-16")

    teams = sorted(set(fixtures["team_a"]) | set(fixtures["team_b"]))
    team_states = initialize_team_states(teams)

    features_df = build_features_for_fixtures(
        fixtures=fixtures,
        team_states=team_states,
        historical_matches=historical_matches,
        market_values=market_values,
        elo_ratings=elo_ratings,
        rankings=rankings,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Shape:", features_df.shape)
    print(features_df.head())


if __name__ == "__main__":
    main()
