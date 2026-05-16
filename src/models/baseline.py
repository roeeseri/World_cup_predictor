from __future__ import annotations

import numpy as np
import pandas as pd


class ConstantScoreBaseline:
    def __init__(self, home_goals: float | None = None, away_goals: float | None = None) -> None:
        self.home_goals = home_goals
        self.away_goals = away_goals

    def fit(self, X, y):
        y_arr = _coerce_goal_array(y)
        if self.home_goals is None:
            self.home_goals = float(np.mean(y_arr[:, 0]))
        if self.away_goals is None:
            self.away_goals = float(np.mean(y_arr[:, 1]))
        return self

    def predict(self, X):
        n_rows = len(X) if hasattr(X, "__len__") else 1
        return np.tile([self.home_goals, self.away_goals], (n_rows, 1))


class EloBaseline:
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

        self.avg_home_goals = float(matches_df["home_goals"].mean())
        self.avg_away_goals = float(matches_df["away_goals"].mean())

        if "date" in matches_df.columns:
            matches_df = matches_df.sort_values("date")

        for _, row in matches_df.iterrows():
            self._update_ratings(
                row["home_team"],
                row["away_team"],
                int(row["home_goals"]),
                int(row["away_goals"]),
            )
        return self

    def predict(self, matches_df: pd.DataFrame):
        if not isinstance(matches_df, pd.DataFrame):
            raise ValueError("EloBaseline.predict expects a pandas DataFrame of matches.")

        preds = []
        for _, row in matches_df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]
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


def _coerce_goal_array(y) -> np.ndarray:
    if isinstance(y, pd.DataFrame):
        if {"home_goals", "away_goals"}.issubset(y.columns):
            return y[["home_goals", "away_goals"]].to_numpy()
        return y.iloc[:, :2].to_numpy()
    y_arr = np.asarray(y)
    if y_arr.ndim != 2 or y_arr.shape[1] < 2:
        raise ValueError("Expected y with shape (n_samples, 2) for home/away goals.")
    return y_arr[:, :2]
