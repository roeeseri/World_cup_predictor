"""
Feature analysis and selection utilities for goal prediction models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def get_tree_feature_importance(X, y, feature_names=None, n_estimators=200, max_depth=15):
    """Extract feature importance from tree-based models."""

    # Handle both goals_A and goals_B
    if isinstance(y, pd.DataFrame):
        y_values = y[['goals_A', 'goals_B']].values
    elif isinstance(y, np.ndarray):
        if y.ndim == 2 and y.shape[1] >= 2:
            y_values = y[:, :2]
        else:
            raise ValueError("y must be 2D with 2 goal columns")
    else:
        raise ValueError("y must be DataFrame or 2D array")

    # Train model for goals_A
    rf_a = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    rf_a.fit(X, y_values[:, 0])

    # Train model for goals_B
    rf_b = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    rf_b.fit(X, y_values[:, 1])

    # Get feature names
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Combine importances
    importance_a = rf_a.feature_importances_
    importance_b = rf_b.feature_importances_
    importance_avg = (importance_a + importance_b) / 2

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_a': importance_a,
        'importance_b': importance_b,
        'importance_avg': importance_avg,
        'importance_sum': importance_a + importance_b,
    })

    return importance_df.sort_values('importance_avg', ascending=False).reset_index(drop=True)


def select_top_features(importance_df, n_features=20):
    """Select top N features by importance."""
    return importance_df.head(n_features)['feature'].tolist()


def analyze_feature_distributions(df, feature_cols, by_column='competition', top_n=5):
    """Analyze feature distributions across different competitions/tournaments."""
    results = []

    for comp in df[by_column].unique():
        comp_mask = df[by_column] == comp
        comp_data = df.loc[comp_mask, feature_cols]

        results.append({
            'competition': comp,
            'matches': comp_mask.sum(),
            'feature_mean': comp_data.mean().mean(),
            'feature_std': comp_data.std().mean(),
        })

    results_df = pd.DataFrame(results)
    return results_df.sort_values('matches', ascending=False).head(top_n)


def correlation_with_exact_score(df, feature_cols, y_true):
    """Calculate correlation of features with exact score prediction accuracy."""

    # For each feature, check if higher values correlate with better outcomes
    correlations = []

    for col in feature_cols:
        try:
            corr_a = df[col].corr(pd.Series(y_true[:, 0]))
            corr_b = df[col].corr(pd.Series(y_true[:, 1]))
            avg_corr = (abs(corr_a) + abs(corr_b)) / 2

            correlations.append({
                'feature': col,
                'corr_goals_a': corr_a,
                'corr_goals_b': corr_b,
                'avg_abs_corr': avg_corr,
            })
        except Exception:
            pass

    corr_df = pd.DataFrame(correlations)
    return corr_df.sort_values('avg_abs_corr', ascending=False).reset_index(drop=True)
