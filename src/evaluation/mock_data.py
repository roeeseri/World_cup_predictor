from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ============================================================================
# Feature Set: Extracted from partner's 05_build_model_dataset.ipynb
# Total: 49 columns (metadata + targets + features)
# ============================================================================

# Metadata columns (informational, not used as model features)
METADATA_COLUMNS = [
    "date",
    "team_A",
    "team_B",
    "competition",
    "location",
]

# Target columns (what we predict)
TARGET_COLUMNS = [
    "goals_A",
    "goals_B",
]

# Model feature columns (49 total: derived from Elo data, rolling stats, and match context)
# Organized by category for clarity:
DEFAULT_FEATURE_COLUMNS = [
    # Pre-match strength ratings (continuous, Elo-based)
    "rating_A_before",
    "rating_B_before",
    "elo_diff",  # rating_A_before - rating_B_before
    
    # Ranking positions
    "rank_A_before",
    "rank_B_before",
    "rank_diff",
    
    # Recent form (last 5 matches)
    "team_A_goals_for_last5",
    "team_B_goals_for_last5",
    "team_A_goals_against_last5",
    "team_B_goals_against_last5",
    "team_A_goal_diff_last5",
    "team_B_goal_diff_last5",
    "team_A_win_rate_last5",
    "team_B_win_rate_last5",
    
    # Recent form (last 10 matches)
    "team_A_goals_for_last10",
    "team_B_goals_for_last10",
    "team_A_goals_against_last10",
    "team_B_goals_against_last10",
    "team_A_goal_diff_last10",
    "team_B_goal_diff_last10",
    "team_A_win_rate_last10",
    "team_B_win_rate_last10",
    
    # Rating momentum (recent rating changes)
    "team_A_rating_change_avg_last5",
    "team_B_rating_change_avg_last5",
    "team_A_rating_change_avg_last10",
    "team_B_rating_change_avg_last10",
    
    # Rank momentum
    "team_A_rank_change_avg_last5",
    "team_B_rank_change_avg_last5",
    "team_A_rank_change_avg_last10",
    "team_B_rank_change_avg_last10",
    
    # Match history & rest
    "team_A_matches_played_before",
    "team_B_matches_played_before",
    "team_A_days_since_last_match",
    "team_B_days_since_last_match",
    "rest_days_diff",  # team_A rest - team_B rest
    
    # Form streaks
    "team_A_win_streak",
    "team_B_win_streak",
    
    # Opponent effect features (enhance dependency modeling)
    "team_A_avg_opponent_rating_last10",
    "team_B_avg_opponent_rating_last10",
    "team_A_strength_vs_recent_opponents",  # How A performed vs opponent strength
    "team_B_strength_vs_recent_opponents",
    
    # Match context
    "is_home_advantage",  # 1 if team_A is at home, 0 otherwise
]


def generate_mock_feature_table(n_matches: int = 600, random_state: int = 42) -> pd.DataFrame:
    """
    Generate realistic mock feature table matching partner's 05_build_model_dataset.ipynb schema.
    
    This synthetic data mirrors the actual feature generation pipeline:
    - Continuous Elo ratings (team strength)
    - Rolling statistics (last 5 & 10 matches)
    - Recent form and momentum
    - Rest days and match history
    - Home advantage
    
    The targets (goals_A, goals_B) are generated with realistic distributions:
    - Base rate ~2.6 total goals per match (typical soccer)
    - Correlation through Elo difference (stronger team scores more, concedes less)
    - Rest days effect (well-rested teams score slightly more)
    - Home advantage effect (+10-15% offensive boost)
    
    Args:
        n_matches: Number of synthetic matches to generate
        random_state: Seed for reproducibility
    
    Returns:
        pd.DataFrame with all 49 feature columns + 2 target columns
    """
    rng = np.random.default_rng(random_state)
    
    # ========== Generate basic match info ==========
    team_ids = [f"Team_{i:02d}" for i in range(1, 33)]  # 32 teams
    teams_A = rng.choice(team_ids, size=n_matches)
    teams_B = rng.choice(team_ids, size=n_matches)
    
    base_date = datetime(2018, 6, 1)
    dates = [base_date + timedelta(days=int(x)) for x in rng.integers(0, 2000, size=n_matches)]
    competitions = rng.choice(["Friendly", "Qualifier", "Tournament"], size=n_matches, p=[0.4, 0.4, 0.2])
    locations = rng.choice(["Home", "Away", "Neutral"], size=n_matches, p=[0.5, 0.3, 0.2])
    
    # ========== Generate Elo ratings (continuous, realistic scale 1200-1900) ==========
    rating_A = rng.normal(1500, 140, size=n_matches).clip(1200, 1900)
    rating_B = rng.normal(1500, 140, size=n_matches).clip(1200, 1900)
    elo_diff = rating_A - rating_B
    
    # Rank positions inversely correlated with rating
    rank_A = (2000 - rating_A) / 5 + rng.normal(0, 10, size=n_matches)
    rank_B = (2000 - rating_B) / 5 + rng.normal(0, 10, size=n_matches)
    rank_A = np.clip(rank_A.astype(int), 1, 200)
    rank_B = np.clip(rank_B.astype(int), 1, 200)
    rank_diff = rank_A - rank_B
    
    # ========== Recent form (rolling statistics) ==========
    # Goals scored in recent matches (correlate with rating, add noise)
    strength_factor_A = (rating_A - 1500) / 400  # Normalize strength to [-1, 1] roughly
    strength_factor_B = (rating_B - 1500) / 400
    
    team_A_goals_for_last5 = np.clip(1.2 + 0.4 * strength_factor_A + rng.normal(0, 0.4, size=n_matches), 0.2, 3.0)
    team_B_goals_for_last5 = np.clip(1.2 + 0.4 * strength_factor_B + rng.normal(0, 0.4, size=n_matches), 0.2, 3.0)
    team_A_goals_against_last5 = np.clip(1.2 - 0.3 * strength_factor_A + rng.normal(0, 0.4, size=n_matches), 0.2, 3.0)
    team_B_goals_against_last5 = np.clip(1.2 - 0.3 * strength_factor_B + rng.normal(0, 0.4, size=n_matches), 0.2, 3.0)
    team_A_goal_diff_last5 = team_A_goals_for_last5 - team_A_goals_against_last5
    team_B_goal_diff_last5 = team_B_goals_for_last5 - team_B_goals_against_last5
    
    # Win rates (stronger teams win more often)
    team_A_win_rate_last5 = np.clip(0.35 + 0.3 * strength_factor_A + rng.normal(0, 0.1, size=n_matches), 0, 1)
    team_B_win_rate_last5 = np.clip(0.35 + 0.3 * strength_factor_B + rng.normal(0, 0.1, size=n_matches), 0, 1)
    
    # Last 10 matches (slightly different trend)
    team_A_goals_for_last10 = team_A_goals_for_last5 + rng.normal(0, 0.2, size=n_matches)
    team_B_goals_for_last10 = team_B_goals_for_last5 + rng.normal(0, 0.2, size=n_matches)
    team_A_goals_against_last10 = team_A_goals_against_last5 + rng.normal(0, 0.2, size=n_matches)
    team_B_goals_against_last10 = team_B_goals_against_last5 + rng.normal(0, 0.2, size=n_matches)
    team_A_goal_diff_last10 = team_A_goals_for_last10 - team_A_goals_against_last10
    team_B_goal_diff_last10 = team_B_goals_for_last10 - team_B_goals_against_last10
    team_A_win_rate_last10 = np.clip(team_A_win_rate_last5 + rng.normal(0, 0.1, size=n_matches), 0, 1)
    team_B_win_rate_last10 = np.clip(team_B_win_rate_last5 + rng.normal(0, 0.1, size=n_matches), 0, 1)
    
    # ========== Rating momentum ==========
    team_A_rating_change_avg_last5 = rng.normal(0, 15, size=n_matches)
    team_B_rating_change_avg_last5 = rng.normal(0, 15, size=n_matches)
    team_A_rating_change_avg_last10 = team_A_rating_change_avg_last5 * 0.6 + rng.normal(0, 10, size=n_matches)
    team_B_rating_change_avg_last10 = team_B_rating_change_avg_last5 * 0.6 + rng.normal(0, 10, size=n_matches)
    
    # Rank momentum (inverse)
    team_A_rank_change_avg_last5 = -team_A_rating_change_avg_last5 / 50
    team_B_rank_change_avg_last5 = -team_B_rating_change_avg_last5 / 50
    team_A_rank_change_avg_last10 = -team_A_rating_change_avg_last10 / 50
    team_B_rank_change_avg_last10 = -team_B_rating_change_avg_last10 / 50
    
    # ========== Match history ==========
    team_A_matches_played = rng.integers(0, 80, size=n_matches)
    team_B_matches_played = rng.integers(0, 80, size=n_matches)
    team_A_days_since_last = rng.integers(1, 25, size=n_matches)
    team_B_days_since_last = rng.integers(1, 25, size=n_matches)
    rest_days_diff = team_A_days_since_last - team_B_days_since_last
    
    # ========== Win streaks ==========
    team_A_win_streak = rng.integers(0, 6, size=n_matches)
    team_B_win_streak = rng.integers(0, 6, size=n_matches)
    
    # ========== Opponent strength features (for dependency modeling) ==========
    team_A_avg_opponent_rating = 1500 + rng.normal(0, 100, size=n_matches)
    team_B_avg_opponent_rating = 1500 + rng.normal(0, 100, size=n_matches)
    # How well A performs vs opponent strength (if A is 1800 and played 1600-rated teams, positive)
    team_A_strength_vs_opponents = (rating_A - team_A_avg_opponent_rating) / 100
    team_B_strength_vs_opponents = (rating_B - team_B_avg_opponent_rating) / 100
    
    # ========== Home advantage ==========
    is_home_adv = (locations == "Home").astype(int)
    
    # ========== Generate targets with realistic soccer distributions ==========
    # Expected goals based on Elo difference + form + rest + home advantage
    base_goals = 1.3
    home_boost = 0.2 if is_home_adv.any() else 0
    rest_boost = (team_A_days_since_last - 7) / 20  # 7 days is "normal"
    
    lambda_A = np.clip(
        base_goals 
        + 0.5 * np.tanh(elo_diff / 400)  # Elo difference predicts goal difference
        + 0.3 * team_A_goals_for_last5  # Form matters
        + 0.2 * is_home_adv  # Home advantage
        + 0.1 * rest_boost,  # Rest matters slightly
        0.3, 4.0
    )
    
    lambda_B = np.clip(
        base_goals 
        - 0.5 * np.tanh(elo_diff / 400)  # Opposite to A (dependency!)
        + 0.3 * team_B_goals_for_last5,
        0.3, 4.0
    )
    
    # Sample actual goals from Poisson with these expectations
    goals_A = rng.poisson(lambda_A)
    goals_B = rng.poisson(lambda_B)
    
    # Cap extreme outliers to rare events (9-0 scores exist but are very rare)
    # Keep them in data but they'll be marked as anomalies during evaluation
    goals_A = np.minimum(goals_A, 10)
    goals_B = np.minimum(goals_B, 10)
    
    # ========== Assemble DataFrame ==========
    df = pd.DataFrame(
        {
            # Metadata
            "date": dates,
            "team_A": teams_A,
            "team_B": teams_B,
            "competition": competitions,
            "location": locations,
            
            # Targets
            "goals_A": goals_A,
            "goals_B": goals_B,
            
            # Features: Elo ratings
            "rating_A_before": rating_A,
            "rating_B_before": rating_B,
            "elo_diff": elo_diff,
            "rank_A_before": rank_A,
            "rank_B_before": rank_B,
            "rank_diff": rank_diff,
            
            # Features: Recent form (last 5)
            "team_A_goals_for_last5": team_A_goals_for_last5,
            "team_B_goals_for_last5": team_B_goals_for_last5,
            "team_A_goals_against_last5": team_A_goals_against_last5,
            "team_B_goals_against_last5": team_B_goals_against_last5,
            "team_A_goal_diff_last5": team_A_goal_diff_last5,
            "team_B_goal_diff_last5": team_B_goal_diff_last5,
            "team_A_win_rate_last5": team_A_win_rate_last5,
            "team_B_win_rate_last5": team_B_win_rate_last5,
            
            # Features: Recent form (last 10)
            "team_A_goals_for_last10": team_A_goals_for_last10,
            "team_B_goals_for_last10": team_B_goals_for_last10,
            "team_A_goals_against_last10": team_A_goals_against_last10,
            "team_B_goals_against_last10": team_B_goals_against_last10,
            "team_A_goal_diff_last10": team_A_goal_diff_last10,
            "team_B_goal_diff_last10": team_B_goal_diff_last10,
            "team_A_win_rate_last10": team_A_win_rate_last10,
            "team_B_win_rate_last10": team_B_win_rate_last10,
            
            # Features: Momentum
            "team_A_rating_change_avg_last5": team_A_rating_change_avg_last5,
            "team_B_rating_change_avg_last5": team_B_rating_change_avg_last5,
            "team_A_rating_change_avg_last10": team_A_rating_change_avg_last10,
            "team_B_rating_change_avg_last10": team_B_rating_change_avg_last10,
            "team_A_rank_change_avg_last5": team_A_rank_change_avg_last5,
            "team_B_rank_change_avg_last5": team_B_rank_change_avg_last5,
            "team_A_rank_change_avg_last10": team_A_rank_change_avg_last10,
            "team_B_rank_change_avg_last10": team_B_rank_change_avg_last10,
            
            # Features: Match history
            "team_A_matches_played_before": team_A_matches_played,
            "team_B_matches_played_before": team_B_matches_played,
            "team_A_days_since_last_match": team_A_days_since_last,
            "team_B_days_since_last_match": team_B_days_since_last,
            "rest_days_diff": rest_days_diff,
            
            # Features: Streaks
            "team_A_win_streak": team_A_win_streak,
            "team_B_win_streak": team_B_win_streak,
            
            # Features: Opponent strength (dependency)
            "team_A_avg_opponent_rating_last10": team_A_avg_opponent_rating,
            "team_B_avg_opponent_rating_last10": team_B_avg_opponent_rating,
            "team_A_strength_vs_recent_opponents": team_A_strength_vs_opponents,
            "team_B_strength_vs_recent_opponents": team_B_strength_vs_opponents,
            
            # Features: Context
            "is_home_advantage": is_home_adv,
        }
    )
    
    return df
