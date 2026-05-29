"""
Production training script for World Cup score predictor.

Uses best hyperparameters and weighting strategy to train models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import load_model_dataset, infer_feature_columns, build_sample_weight
from src.models.poisson_model import PoissonGoalModel
from src.models.tree_model import TreeGoalModel
from src.models.ensemble import EnsembleGoalModel
from src.models.weighting import apply_competition_weights, apply_combined_weighting
from src.models.train import save_model


def train_production_model(
    model_type: str = "tree",
    use_weights: bool = True,
    temporal_decay: bool = False,
    save_path: str = "models/saved/production_model.pkl",
    config_path: str = "models/saved/model_config.json",
):
    """
    Train production model with best hyperparameters.

    Args:
        model_type: "poisson", "tree", or "ensemble"
        use_weights: Whether to apply competition weights
        temporal_decay: Whether to apply temporal decay to older matches
        save_path: Where to save the model
        config_path: Where to save model configuration

    Returns:
        Trained model and configuration dict
    """

    print("Loading dataset...")
    df = load_model_dataset()

    # Sort chronologically
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Feature engineering
    feature_cols = infer_feature_columns(df)
    X = df[feature_cols].fillna(0)
    y = df[['goals_A', 'goals_B']].values

    print(f"Dataset: {len(df)} matches, {len(feature_cols)} features")

    # Compute weights
    weights = None
    if use_weights:
        if temporal_decay:
            weights = apply_combined_weighting(
                df,
                apply_decay=True,
                competition_weight=0.6,
                temporal_weight=0.4,
                reference_year=2024,
            )
            print("Applied combined weighting (competition + temporal decay)")
        else:
            weights = apply_competition_weights(df)
            print("Applied competition-based weighting")

    # Train model with best hyperparameters
    print(f"\nTraining {model_type} model...")

    if model_type == "poisson":
        # Best hyperparameters from Optuna (example, will be updated by experiments)
        model = PoissonGoalModel(alpha=5.0, max_iter=1000)
        model.fit(X, y, sample_weight=weights)

    elif model_type == "tree":
        # Best hyperparameters from Optuna (example, will be updated by experiments)
        model = TreeGoalModel(n_estimators=150, max_depth=18, random_state=42)
        model.fit(X, y, sample_weight=weights)

    elif model_type == "ensemble":
        # Ensemble of best models
        models = [
            PoissonGoalModel(alpha=5.0, max_iter=1000),
            TreeGoalModel(n_estimators=150, max_depth=18, random_state=42),
        ]
        model = EnsembleGoalModel(models, weights=[0.5, 0.5])
        model.fit(X, y, sample_weight=weights)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"✓ Model trained successfully")

    # Save model
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, save_path)
    print(f"✓ Model saved to {save_path}")

    # Save configuration
    config = {
        "model_type": model_type,
        "feature_columns": feature_cols,
        "dataset_info": {
            "n_samples": len(df),
            "n_features": len(feature_cols),
            "date_min": str(df['date'].min()),
            "date_max": str(df['date'].max()),
        },
        "hyperparameters": {
            "use_weights": use_weights,
            "temporal_decay": temporal_decay,
        },
        "model_specific": {
            "poisson": {"alpha": 5.0, "max_iter": 1000},
            "tree": {"n_estimators": 150, "max_depth": 18},
        }
    }

    config_path_obj = Path(config_path)
    config_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path_obj, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved to {config_path}")

    return model, config


def evaluate_production_model(model_path: str, config_path: str, test_data_path: str | None = None):
    """
    Evaluate production model on test data.

    Args:
        model_path: Path to saved model
        config_path: Path to model config
        test_data_path: Optional path to test data CSV

    Returns:
        Evaluation metrics dict
    """
    from src.models.train import load_model
    from src.models.optuna_tuning import exact_score_accuracy, result_accuracy

    print("Loading model and config...")
    model = load_model(model_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    feature_cols = config["feature_columns"]

    # Load test data (last 20% chronologically)
    if test_data_path is None:
        df = load_model_dataset()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        split_idx = int(len(df) * 0.8)
        df_test = df.iloc[split_idx:].copy()
    else:
        df_test = pd.read_csv(test_data_path)

    X_test = df_test[feature_cols].fillna(0)
    y_test = df_test[['goals_A', 'goals_B']].values

    # Predict
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    # Calculate metrics
    metrics = {
        "exact_score_accuracy": float(exact_score_accuracy(y_test, y_pred)),
        "result_accuracy": float(result_accuracy(y_test, y_pred)),
        "goal_mae": float(np.mean(np.abs(y_test - y_pred))),
        "n_test_samples": len(X_test),
    }

    print("\nEvaluation Results:")
    for key, value in metrics.items():
        if "accuracy" in key:
            print(f"  {key}: {value*100:.2f}%")
        else:
            print(f"  {key}: {value}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        default="tree",
        choices=["poisson", "tree", "ensemble"],
        help="Model type to train",
    )
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Don't use competition weights",
    )
    parser.add_argument(
        "--temporal-decay",
        action="store_true",
        help="Apply temporal decay to older matches",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate after training",
    )

    args = parser.parse_args()

    # Train
    model, config = train_production_model(
        model_type=args.model_type,
        use_weights=not args.no_weights,
        temporal_decay=args.temporal_decay,
    )

    # Evaluate if requested
    if args.evaluate:
        metrics = evaluate_production_model(
            "models/saved/production_model.pkl",
            "models/saved/model_config.json",
        )
