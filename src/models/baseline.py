from __future__ import annotations

import numpy as np
import pandas as pd

from .base import coerce_goal_array, ensure_non_negative


class ConstantScoreBaseline:
    """
    Predicts constant expected goals for all matches.
    
    Useful as lowest-skill baseline. If your model can't beat this, it's broken.
    """
    def __init__(self, goals_a: float | None = None, goals_b: float | None = None) -> None:
        self.goals_a = goals_a
        self.goals_b = goals_b

    def fit(self, X, y, sample_weight=None):
        """Learn average goals from training data."""
        y_arr = coerce_goal_array(y)
        weights = _coerce_sample_weight(sample_weight, len(y_arr))
        if self.goals_a is None:
            self.goals_a = float(np.average(y_arr[:, 0], weights=weights))
        if self.goals_b is None:
            self.goals_b = float(np.average(y_arr[:, 1], weights=weights))
        return self

    def predict(self, X):
        """Predict constant goals for any number of matches."""
        n_rows = len(X) if hasattr(X, "__len__") else 1
        return ensure_non_negative(np.tile([self.goals_a, self.goals_b], (n_rows, 1)))


class AverageGoalsBaseline:
    """
    Predicts average goals per match (fitted each time).
    
    Quick sanity check baseline.
    """
    def __init__(self) -> None:
        self.avg_goals_a = 1.2
        self.avg_goals_b = 1.0

    def fit(self, X, y, sample_weight=None):
        """Learn average goals from training data."""
        y_arr = coerce_goal_array(y)
        weights = _coerce_sample_weight(sample_weight, len(y_arr))
        self.avg_goals_a = float(np.average(y_arr[:, 0], weights=weights))
        self.avg_goals_b = float(np.average(y_arr[:, 1], weights=weights))
        return self

    def predict(self, X):
        """Predict average goals for any match."""
        n_rows = len(X) if hasattr(X, "__len__") else 1
        return ensure_non_negative(np.tile([self.avg_goals_a, self.avg_goals_b], (n_rows, 1)))

class EloHeuristicBaseline:
    """
    Predicts goals based on Elo rating difference and recent form.
    
    Formula:
        goals_A = avg_A + scale * tanh(elo_diff / 400)
        goals_B = avg_B - scale * tanh(elo_diff / 400)
    
    Key features:
    - Uses continuous Elo ratings (no team memorization)
    - tanh() keeps predictions bounded
    - Elo difference is the PRIMARY dependency mechanism:
      When A is much stronger than B, A scores more AND B scores less (natural coupling)
    """
    def __init__(self, scale: float = 0.35) -> None:
        self.scale = scale
        self.avg_goals_a = 1.2
        self.avg_goals_b = 1.0

    def fit(self, X, y, sample_weight=None):
        """Learn average goals from training data."""
        y_arr = coerce_goal_array(y)
        weights = _coerce_sample_weight(sample_weight, len(y_arr))
        self.avg_goals_a = float(np.average(y_arr[:, 0], weights=weights))
        self.avg_goals_b = float(np.average(y_arr[:, 1], weights=weights))
        return self

    def predict(self, X):
        """
        Predict goals using Elo difference.
        
        Dependency mechanism: elo_diff couples A's and B's predictions.
        """
        if not isinstance(X, pd.DataFrame):
            n_rows = len(X) if hasattr(X, "__len__") else 1
            return ensure_non_negative(np.tile([self.avg_goals_a, self.avg_goals_b], (n_rows, 1)))

        # Extract Elo difference (primary feature for this baseline)
        if "elo_diff" in X.columns:
            diff = X["elo_diff"].to_numpy(dtype=float)
        elif {"rating_a_before", "rating_b_before"}.issubset(X.columns):
            diff = (X["rating_a_before"] - X["rating_b_before"]).to_numpy(dtype=float)
        elif {"rating_A_before", "rating_B_before"}.issubset(X.columns):
            diff = (X["rating_A_before"] - X["rating_B_before"]).to_numpy(dtype=float)
        else:
            # Fallback: no Elo difference available
            diff = np.zeros(len(X), dtype=float)

        # Tanh transformation: maps [-inf, inf] to [-1, 1]
        # Divide by 400 because Elo convention: 400-point gap ≈ 92% win probability
        shift = np.tanh(diff / 400.0) * self.scale
        goals_a = np.clip(self.avg_goals_a + shift, 0.1, None)
        goals_b = np.clip(self.avg_goals_b - shift, 0.1, None)
        return ensure_non_negative(np.column_stack([goals_a, goals_b]))


class EloBaseline:
    """
    Legacy Elo-based model (for reference, not recommended for feature-based prediction).
    
    Updates ratings dynamically during training (only works with match DataFrames).
    """
    def __init__(self, k_factor: float = 20.0, base_rating: float = 1500.0, home_advantage: float = 50.0) -> None:
        self.k_factor = k_factor
        self.base_rating = base_rating
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = {}
        self.avg_home_goals = 1.2
        self.avg_away_goals = 1.0

    def fit(self, matches_df: pd.DataFrame):
        if not isinstance(matches_df, pd.DataFrame):
            raise ValueError("EloBaseline.fit expects a pandas DataFrame of matches.")

        self.avg_home_goals = float(matches_df["goals_A"].mean())
        self.avg_away_goals = float(matches_df["goals_B"].mean())

        if "date" in matches_df.columns:
            matches_df = matches_df.sort_values("date")

        for _, row in matches_df.iterrows():
            self._update_ratings(
                row["team_A"],
                row["team_B"],
                int(row["goals_A"]),
                int(row["goals_B"]),
            )
        return self

    def predict(self, matches_df: pd.DataFrame):
        if not isinstance(matches_df, pd.DataFrame):
            raise ValueError("EloBaseline.predict expects a pandas DataFrame of matches.")

        preds = []
        for _, row in matches_df.iterrows():
            home_team = row["team_A"]
            away_team = row["team_B"]
            home_rating = self._rating(home_team) + self.home_advantage
            away_rating = self._rating(away_team)
            p_home = self._expected_score(home_rating, away_rating)

            shift = (p_home - 0.5) * 0.8
            home_lambda = max(0.1, self.avg_home_goals + shift)
            away_lambda = max(0.1, self.avg_away_goals - shift)
            preds.append([home_lambda, away_lambda])

        return np.array(preds)

    def _rating(self, team: str) -> float:
        if team not in self.ratings:
            self.ratings[team] = self.base_rating
        return self.ratings[team]

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def _update_ratings(self, home_team: str, away_team: str, home_goals: int, away_goals: int) -> None:
        home_rating = self._rating(home_team)
        away_rating = self._rating(away_team)

        expected_home = self._expected_score(home_rating + self.home_advantage, away_rating)
        if home_goals > away_goals:
            actual_home = 1.0
        elif home_goals < away_goals:
            actual_home = 0.0
        else:
            actual_home = 0.5

        delta = self.k_factor * (actual_home - expected_home)
        self.ratings[home_team] = home_rating + delta
        self.ratings[away_team] = away_rating - delta


def _coerce_sample_weight(sample_weight, n_rows: int) -> np.ndarray | None:
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape[0] != n_rows:
        raise ValueError("sample_weight must have one value per training row.")
    weights = np.clip(weights, 0.0, None)
    if np.all(weights == 0):
        return None
    return weights
