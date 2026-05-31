"""Official production feature columns for model_dataset.csv."""

FEATURE_COLS = [
    "rating_a_before",
    "rating_b_before",
    "elo_diff",
    "rank_diff",

    "log_market_value_a",
    "log_market_value_b",
    "market_value_rel_mean_diff",
    "avg_player_value_diff",

    "form_diff_last5",
    "weighted_goals_for_diff_last5",
    "weighted_goals_against_diff_last5",
    "opponent_strength_diff_last5",
    "rating_change_diff_last5",

    "team_a_matches_played_before",
    "team_b_matches_played_before",
    "team_a_days_since_last_match",
    "team_b_days_since_last_match",
    "days_since_match_diff",

    "team_a_tournament_matches_played",
    "team_b_tournament_matches_played",
    "tournament_points_diff",
    "tournament_goal_diff_diff",

    "competition_weight",
    "is_home_adv",
]

TARGET_COLS = ["target_goals_a", "target_goals_b"]

METADATA_COLS = [
    "date",
    "team_a",
    "team_b",
    "competition",
    "location",
    "season_id",
    "tournament_year",
    "tournament_key",
]