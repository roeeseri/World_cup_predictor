from __future__ import annotations

from pathlib import Path

from joblib import dump, load

from .baseline import AverageGoalsBaseline, ConstantScoreBaseline, EloBaseline, EloHeuristicBaseline
from .ensemble import EnsembleGoalModel
from .poisson_model import PoissonGoalModel
from .tree_model import TreeGoalModel


def train_model(X_train, y_train, model_type: str = "poisson"):
    model_type = model_type.lower()

    if model_type == "poisson":
        model = PoissonGoalModel()
        model.fit(X_train, y_train)
        return model

    if model_type == "tree":
        model = TreeGoalModel()
        model.fit(X_train, y_train)
        return model

    if model_type == "ensemble":
        model = EnsembleGoalModel([PoissonGoalModel(), TreeGoalModel()])
        model.fit(X_train, y_train)
        return model

    if model_type == "constant":
        model = ConstantScoreBaseline()
        model.fit(X_train, y_train)
        return model

    if model_type == "average":
        model = AverageGoalsBaseline()
        model.fit(X_train, y_train)
        return model

    if model_type == "elo":
        model = EloHeuristicBaseline()
        model.fit(X_train, y_train)
        return model

    if model_type == "elo_legacy":
        model = EloBaseline()
        model.fit(X_train)
        return model

    raise ValueError(f"Unknown model_type: {model_type}")


def save_model(model, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    dump(model, target)


def load_model(path):
    return load(Path(path))
