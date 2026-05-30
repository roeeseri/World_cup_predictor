"""Utilities for World Cup-focused model optimization and evaluation."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import warnings


# ============================================================================
# DATA QUALITY FUNCTIONS
# ============================================================================

def validate_dataset_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate dataset quality against feature dictionary.

    Returns dictionary with validation results and anomalies found.
    """
    issues = {}

    # Check required columns (handle both naming conventions)
    required_cols_map = {
        'date': ['date'],
        'team_a': ['team_a'],
        'team_b': ['team_b'],
        'goals': ['target_goals_a', 'goals_a', 'target_goals_b', 'goals_b']
    }

    missing_cols = []
    for col_group in required_cols_map.values():
        if not any(c in df.columns for c in col_group):
            missing_cols.extend(col_group)

    if missing_cols:
        issues['missing_required_columns'] = missing_cols

    # Check days_since_match_diff consistency
    if 'team_a_days_since_last_match' in df.columns and 'team_b_days_since_last_match' in df.columns:
        if 'days_since_match_diff' in df.columns:
            calculated_diff = df['team_a_days_since_last_match'] - df['team_b_days_since_last_match']
            mismatches = (df['days_since_match_diff'] != calculated_diff).sum()
            if mismatches > 0:
                issues['days_since_match_diff_mismatches'] = {
                    'count': mismatches,
                    'percentage': 100 * mismatches / len(df)
                }

    # Check for negative goals (handle both naming conventions)
    goal_a_col = 'target_goals_a' if 'target_goals_a' in df.columns else 'goals_a' if 'goals_a' in df.columns else None
    goal_b_col = 'target_goals_b' if 'target_goals_b' in df.columns else 'goals_b' if 'goals_b' in df.columns else None

    if goal_a_col and goal_b_col:
        negative_goals_a = (df[goal_a_col] < 0).sum()
        negative_goals_b = (df[goal_b_col] < 0).sum()
        if negative_goals_a > 0 or negative_goals_b > 0:
            issues['negative_goals'] = {
                'team_a': negative_goals_a,
                'team_b': negative_goals_b
            }

    # Check for missing values in key features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing_by_col = df[numeric_cols].isnull().sum()
    missing_cols_dict = missing_by_col[missing_by_col > 0].to_dict()
    if missing_cols_dict:
        issues['missing_values_in_features'] = missing_cols_dict

    # Check zero-value features
    zero_features = {}
    for col in numeric_cols:
        zero_count = (df[col] == 0).sum()
        zero_pct = 100 * zero_count / len(df)
        if zero_pct > 50:  # More than 50% zeros
            zero_features[col] = {'count': zero_count, 'percentage': zero_pct}
    if zero_features:
        issues['high_zero_value_features'] = zero_features

    # Check tournament representation
    if 'competition' in df.columns:
        competition_counts = df['competition'].value_counts().to_dict()
        issues['competition_distribution'] = competition_counts

    # Check feature value ranges
    feature_ranges = {}
    for col in numeric_cols:
        if df[col].dtype in [np.float64, np.int64]:
            feature_ranges[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
    issues['feature_ranges'] = feature_ranges

    return issues


def fix_days_since_match_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate days_since_match_diff from individual team values.

    Returns modified dataframe with corrected column.
    """
    df = df.copy()
    if 'team_a_days_since_last_match' in df.columns and 'team_b_days_since_last_match' in df.columns:
        original = df['days_since_match_diff'].copy() if 'days_since_match_diff' in df.columns else None
        df['days_since_match_diff'] = df['team_a_days_since_last_match'] - df['team_b_days_since_last_match']

        if original is not None:
            mismatches = (original != df['days_since_match_diff']).sum()
            if mismatches > 0:
                print(f"Fixed {mismatches} mismatched days_since_match_diff values")

    return df


# ============================================================================
# TOURNAMENT FILTERING & SPLITTING FUNCTIONS
# ============================================================================

def filter_world_cup_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Extract World Cup matches only."""
    if 'tournament_key' in df.columns:
        wc_matches = df[df['tournament_key'].str.contains('World Cup', case=False, na=False)]
    elif 'competition' in df.columns:
        wc_matches = df[df['competition'].str == 'World Cup']
    else:
        raise ValueError("Cannot identify World Cup matches - no 'tournament_key' or 'competition' column")

    return wc_matches.reset_index(drop=True)


def filter_major_tournaments(df: pd.DataFrame) -> pd.DataFrame:
    """Extract major tournament matches (Euro, Copa América, African Cup, Asian Cup)."""
    if 'tournament_key' in df.columns:
        major = df[df['tournament_key'].str.contains(
            'Euro|Copa|Africa|African|Asian Cup', case=False, na=False
        )]
    elif 'competition' in df.columns:
        major = df[df['competition'].str.contains(
            'Euro|Copa|Africa|African|Asian Cup', case=False, na=False
        ) & df['competition'].str.contains('qualifier', case=False, na=False) == False]
    else:
        raise ValueError("Cannot identify major tournaments - no tournament column")

    return major.reset_index(drop=True)


def get_unique_tournaments(df: pd.DataFrame) -> Dict[str, int]:
    """Get count of matches per tournament."""
    if 'tournament_key' in df.columns:
        return df['tournament_key'].value_counts().to_dict()
    elif 'competition' in df.columns:
        return df['competition'].value_counts().to_dict()
    return {}


def get_world_cup_years(df: pd.DataFrame) -> List[int]:
    """Extract unique World Cup years from dataset."""
    wc_df = filter_world_cup_matches(df)
    if 'tournament_year' in wc_df.columns:
        years = sorted(wc_df['tournament_year'].unique())
    else:
        years = []
    return years


def create_expanding_window_splits(
    df: pd.DataFrame,
    tournament_years: List[int],
    tournament_filter_func=None
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create expanding-window train/test splits for each tournament year.

    For each year, train on all matches before that tournament,
    test on matches during that tournament.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    tournament_years : List[int]
        Years to create splits for (e.g., [2006, 2010, 2014, 2018, 2022])
    tournament_filter_func : callable, optional
        Function to filter to tournament matches. If None, assumes 'tournament_year' column

    Returns
    -------
    Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]
        Mapping of year -> (train_df, test_df)
    """
    splits = {}

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    for year in sorted(tournament_years):
        # Get tournament matches for this year
        if tournament_filter_func:
            all_tournament_matches = tournament_filter_func(df)
        else:
            if 'tournament_year' not in df.columns:
                raise ValueError("Need 'tournament_year' column or provide tournament_filter_func")
            all_tournament_matches = df.copy()

        test_matches = all_tournament_matches[all_tournament_matches['tournament_year'] == year].copy()

        if len(test_matches) == 0:
            print(f"Warning: No matches found for year {year}")
            continue

        # Get cutoff date as earliest test match date
        cutoff_date = test_matches['date'].min()

        # Train on all matches before cutoff, excluding this tournament
        train_matches = df[
            (df['date'] < cutoff_date) &
            (df['tournament_year'] != year)
        ].copy()

        if len(train_matches) == 0:
            print(f"Warning: Skipping year {year} because no prior training matches were found")
            continue

        splits[year] = (train_matches, test_matches)
        print(f"Year {year}: train={len(train_matches)}, test={len(test_matches)}")

    return splits


def prepare_feature_sets(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Prepare feature sets A (current) and B (with state features).

    Returns
    -------
    Tuple[List[str], List[str]]
        (feature_set_a, feature_set_b)
    """
    # Identify all numeric columns except targets and metadata
    exclude_cols = {
        'date', 'team_a', 'team_b', 'competition', 'location', 'season_id',
        'tournament_year', 'tournament_key',
        # target columns under both naming conventions (standardize_goal_columns renames target_goals_* → goals_*)
        'target_goals_a', 'target_goals_b', 'goals_a', 'goals_b',
        'target_goal_diff', 'target_total_goals',
    }

    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    # State features
    state_features = {
        'team_a_tournament_matches_played',
        'team_b_tournament_matches_played',
        'tournament_points_diff',
        'tournament_goal_diff_diff'
    }

    feature_set_a = [c for c in numeric_cols if c not in state_features]
    feature_set_b = numeric_cols

    # Verify features exist
    feature_set_a = [c for c in feature_set_a if c in df.columns]
    feature_set_b = [c for c in feature_set_b if c in df.columns]

    print(f"Feature Set A (no state): {len(feature_set_a)} features")
    print(f"Feature Set B (with state): {len(feature_set_b)} features")
    print(f"State-only features: {set(feature_set_b) - set(feature_set_a)}")

    return feature_set_a, feature_set_b


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def evaluate_world_cup_only(
    y_true: np.ndarray,
    y_pred_scores: np.ndarray,
    y_pred_expected: Optional[np.ndarray] = None,
    match_info: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Calculate evaluation metrics on World Cup matches.

    Parameters
    ----------
    y_true : np.ndarray
        True goals, shape (n_matches, 2)
    y_pred_scores : np.ndarray
        Predicted scores (rounded), shape (n_matches, 2)
    y_pred_expected : np.ndarray, optional
        Predicted expected goals, shape (n_matches, 2)
    match_info : pd.DataFrame, optional
        Match metadata for breakdown analysis

    Returns
    -------
    Dict[str, float]
        Metrics including exact_score_accuracy, result_accuracy, goal_mae, goal_rmse
    """
    # Ensure correct shapes
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 2)
    if y_pred_scores.ndim == 1:
        y_pred_scores = y_pred_scores.reshape(-1, 2)

    metrics = {}

    # Exact score accuracy
    exact_matches = np.all(y_true == y_pred_scores, axis=1).sum()
    metrics['exact_score_accuracy'] = 100 * exact_matches / len(y_true)

    # Result accuracy (win/draw/loss)
    y_true_result = np.sign(y_true[:, 0] - y_true[:, 1])
    y_pred_result = np.sign(y_pred_scores[:, 0] - y_pred_scores[:, 1])
    result_matches = (y_true_result == y_pred_result).sum()
    metrics['result_accuracy'] = 100 * result_matches / len(y_true)

    # Goal MAE/RMSE
    if y_pred_expected is not None:
        if y_pred_expected.ndim == 1:
            y_pred_expected = y_pred_expected.reshape(-1, 2)
        goal_diff = np.abs(y_true - y_pred_expected)
        metrics['goal_mae'] = goal_diff.mean()
        metrics['goal_rmse'] = np.sqrt((goal_diff ** 2).mean())

    # Rounded score MAE
    rounded_pred = np.round(y_pred_expected) if y_pred_expected is not None else y_pred_scores
    score_diff = np.abs(y_true - rounded_pred)
    metrics['rounded_score_mae'] = score_diff.mean()

    return metrics


def evaluate_major_tournaments(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred_scores: np.ndarray,
    y_pred_expected: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, float]]:
    """
    Breakdown evaluation by major tournament type.

    Returns metrics per tournament.
    """
    results_by_tournament = {}

    if 'competition' not in df.columns:
        return results_by_tournament

    for comp in df['competition'].unique():
        mask = df['competition'] == comp
        if mask.sum() == 0:
            continue

        y_true_comp = y_true[mask]
        y_pred_comp = y_pred_scores[mask]
        y_pred_exp_comp = y_pred_expected[mask] if y_pred_expected is not None else None

        metrics = evaluate_world_cup_only(y_true_comp, y_pred_comp, y_pred_exp_comp)
        results_by_tournament[comp] = metrics

    return results_by_tournament


def compare_feature_sets(
    model_a_scores: Dict[str, float],
    model_b_scores: Dict[str, float],
    feature_set_names: Tuple[str, str] = ("Feature Set A", "Feature Set B")
) -> pd.DataFrame:
    """Compare performance of models trained on different feature sets."""
    comparison = pd.DataFrame({
        feature_set_names[0]: model_a_scores,
        feature_set_names[1]: model_b_scores
    })
    comparison['Difference (B - A)'] = comparison[feature_set_names[1]] - comparison[feature_set_names[0]]
    comparison['Improvement %'] = 100 * comparison['Difference (B - A)'] / comparison[feature_set_names[0]].abs()

    return comparison


def rank_models_by_accuracy(
    results: Dict[str, Dict[str, float]],
    metric: str = 'exact_score_accuracy'
) -> pd.DataFrame:
    """
    Rank models by a specific metric.

    Parameters
    ----------
    results : Dict[str, Dict[str, float]]
        Mapping of model_name -> metrics_dict
    metric : str
        Which metric to rank by (default: 'exact_score_accuracy')

    Returns
    -------
    pd.DataFrame
        Sorted ranking table
    """
    data = []
    for model_name, metrics in results.items():
        row = {'Model': model_name}
        row.update(metrics)
        data.append(row)

    df = pd.DataFrame(data)
    if metric in df.columns:
        df = df.sort_values(metric, ascending=False)

    return df.reset_index(drop=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def standardize_goal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize goal column names (team_a -> goals_a, target_goals_a -> goals_a)."""
    df = df.copy()
    rename_map = {}

    # Handle various naming conventions
    for old_name in df.columns:
        if old_name in ['target_goals_a', 'goals_a', 'team_a_goals']:
            if 'goals_a' not in df.columns:
                rename_map[old_name] = 'goals_a'
        elif old_name in ['target_goals_b', 'goals_b', 'team_b_goals']:
            if 'goals_b' not in df.columns:
                rename_map[old_name] = 'goals_b'

    return df.rename(columns=rename_map)


def apply_team_rating_filter(df: pd.DataFrame, min_rating: float = 1420) -> pd.DataFrame:
    """Filter to matches where both teams have ELO >= min_rating."""
    if 'rating_a_before' in df.columns and 'rating_b_before' in df.columns:
        filtered = df[
            (df['rating_a_before'] >= min_rating) &
            (df['rating_b_before'] >= min_rating)
        ].copy()
        print(f"Filtered to matches with both teams >= {min_rating}: {len(filtered)} / {len(df)}")
        return filtered

    print("Warning: Cannot apply team rating filter - columns not found")
    return df
