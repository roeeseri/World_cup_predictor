"""
Sample weighting strategies for different tournaments and competitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Competition weight mappings
COMPETITION_WEIGHTS = {
    'FIFA World Cup': 5.0,
    'World Cup qualifier': 3.0,
    'European Championship': 4.0,
    'European Championship qualifier': 2.0,
    'Copa America': 3.5,
    'African Nations Cup': 3.0,
    'African Nations Cup qualifier': 1.5,
    'CONCACAF Gold Cup': 2.5,
    'Friendly': 1.0,
}


def get_competition_weight(competition: str, weights_dict: dict | None = None) -> float:
    """Get weight for a single competition."""
    if weights_dict is None:
        weights_dict = COMPETITION_WEIGHTS

    # Exact match
    if competition in weights_dict:
        return weights_dict[competition]

    # Partial match (check if it contains important keywords)
    competition_lower = competition.lower()
    for pattern, weight in weights_dict.items():
        if pattern.lower() in competition_lower:
            return weight

    # Default weight for unknown competitions
    return 1.0


def apply_competition_weights(df: pd.DataFrame, weights_dict: dict | None = None) -> np.ndarray:
    """
    Create sample weights based on competition importance.

    Higher weights for major tournaments, lower for friendlies.
    """
    if weights_dict is None:
        weights_dict = COMPETITION_WEIGHTS

    weights = df['competition'].apply(lambda x: get_competition_weight(x, weights_dict)).values
    weights = weights.astype(float)

    # Normalize to avoid training issues
    weights = weights / np.mean(weights)

    return weights


def apply_temporal_decay(df: pd.DataFrame, decay_rate: float = 0.95, reference_year: int = 2024) -> np.ndarray:
    """
    Apply temporal decay: older matches have lower weight.

    Formula: weight = decay_rate ** (reference_year - match_year)
    """
    df_work = df.copy()
    df_work['date'] = pd.to_datetime(df_work['date'])
    years = df_work['date'].dt.year.values

    weights = np.array([decay_rate ** (reference_year - year) for year in years])
    weights = weights / np.mean(weights)

    return weights


def apply_combined_weighting(
    df: pd.DataFrame,
    competition_weights: dict | None = None,
    apply_decay: bool = False,
    decay_rate: float = 0.95,
    reference_year: int = 2024,
    competition_weight: float = 0.7,
    temporal_weight: float = 0.3,
) -> np.ndarray:
    """
    Combine competition weighting and temporal decay.

    Args:
        df: DataFrame with 'competition' and 'date' columns
        competition_weights: Dict mapping competition names to weights
        apply_decay: Whether to apply temporal decay
        decay_rate: Exponential decay rate (0.95 = 5% decay per year)
        reference_year: Year to decay from
        competition_weight: Weight of competition importance (0-1)
        temporal_weight: Weight of recency (0-1)

    Returns:
        Combined weight array, normalized to mean 1.0
    """
    comp_weights = apply_competition_weights(df, competition_weights)

    if apply_decay:
        temporal_weights = apply_temporal_decay(df, decay_rate, reference_year)
    else:
        temporal_weights = np.ones(len(df))

    # Combine weights
    # Normalize each component to mean 1.0 first
    comp_weights = comp_weights / np.mean(comp_weights)
    temporal_weights = temporal_weights / np.mean(temporal_weights)

    combined = (competition_weight * comp_weights) + (temporal_weight * temporal_weights)
    combined = combined / np.mean(combined)

    return combined


def get_competition_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Get statistics about competition distribution in dataset."""
    return (
        df.groupby('competition')
        .agg({
            'competition': 'count',
            'date': ['min', 'max'],
        })
        .rename(columns={'competition': 'count'})
        .sort_values(('competition', 'count'), ascending=False)
    )
