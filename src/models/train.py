from __future__ import annotations

from pathlib import Path

from joblib import dump, load

from .baseline import AverageGoalsBaseline, ConstantScoreBaseline, EloBaseline, EloHeuristicBaseline
from .ensemble import EnsembleGoalModel
from .lgbm_model import LGBMGoalModel
from .poisson_model import PoissonGoalModel
from .tree_model import TreeGoalModel
from .xgb_model import XGBGoalModel


def train_model(X_train, y_train, model_type: str = "poisson"):
    factories = {
        "poisson": PoissonGoalModel,
        "lgbm": LGBMGoalModel,
        "tree": TreeGoalModel,
        "xgboost": XGBGoalModel,
        "ensemble": lambda: EnsembleGoalModel([PoissonGoalModel(), TreeGoalModel()]),
        "constant": ConstantScoreBaseline,
        "average": AverageGoalsBaseline,
        "elo": EloHeuristicBaseline,
        "elo_legacy": EloBaseline,
    }
    model_type = model_type.lower()
    if model_type not in factories:
        raise ValueError(f"Unknown model_type: {model_type}")
    model = factories[model_type]()
    if model_type == "elo_legacy":
        model.fit(X_train)
    else:
        model.fit(X_train, y_train)
    return model


def save_model(model, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    dump(model, target)


def load_model(path):
    return load(Path(path))
