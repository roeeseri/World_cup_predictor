"""Production feature columns used by the model."""

BASE_FEATURE_COLS = [
    # Team strength
    "rating_a_before",
    "rating_b_before",
    "elo_diff",
    "rank_diff",

    # Compact market features
    "log_market_value_a",
    "log_market_value_b",
    "market_value_rel_mean_diff",
    "avg_player_value_diff",

    # Position-level market value features
    "goalkeeper_value_diff",
    "defender_value_diff",
    "midfield_value_diff",
    "attack_value_diff",
    "goalkeeper_share_diff",
    "defender_share_diff",
    "midfield_share_diff",
    "attack_share_diff",
    "goalkeeper_age_diff",
    "defender_age_diff",
    "midfield_age_diff",
    "attack_age_diff",
    "attack_defense_ratio_diff",
    "midfield_attack_ratio_diff",
    "total_age_diff",

    # Recent performance
    "form_diff_last5",
    "weighted_goals_for_diff_last5",
    "weighted_goals_against_diff_last5",
    "opponent_strength_diff_last5",
    "rating_change_diff_last5",

    # Experience
    "team_a_matches_played_before",
    "team_b_matches_played_before",

    # Schedule
    "team_a_days_since_last_match",
    "team_b_days_since_last_match",
    "days_since_match_diff",

    # Tournament state
    "team_a_tournament_matches_played",
    "team_b_tournament_matches_played",
    "tournament_points_diff",
    "tournament_goal_diff_diff",

    # Context
    "competition_weight",
    "is_home_adv",
]

FEATURE_COLS = BASE_FEATURE_COLS
