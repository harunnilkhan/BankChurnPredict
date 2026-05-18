"""
Model registry module for BankChurnPredict.

Provides candidate model definitions and artifact saving/loading utilities.
"""

import json
import os
from datetime import datetime

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_model_candidates() -> dict:
    """
    Return a dictionary of candidate models to train and compare.

    Returns:
        Dictionary of {model_name: model_instance}.
    """
    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
            class_weight="balanced",
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        ),
    }

    print(f"[INFO] Registered {len(candidates)} candidate models:")
    for name in candidates:
        print(f"       - {name}")

    return candidates


def save_artifacts(
    model,
    preprocessor,
    metadata: dict,
    output_dir: str = "models",
) -> None:
    """
    Save trained model, preprocessor, and metadata to disk.

    Args:
        model: Trained best model.
        preprocessor: Fitted preprocessing pipeline.
        metadata: Dictionary of model metadata.
        output_dir: Directory to save artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "best_model.joblib")
    preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")
    metadata_path = os.path.join(output_dir, "model_metadata.json")

    # Save model
    joblib.dump(model, model_path)
    print(f"[INFO] Model saved to: {model_path}")

    # Save preprocessor
    joblib.dump(preprocessor, preprocessor_path)
    print(f"[INFO] Preprocessor saved to: {preprocessor_path}")

    # Save metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Metadata saved to: {metadata_path}")


def load_artifacts(model_dir: str = "models") -> tuple:
    """
    Load model, preprocessor, and metadata from disk.

    Args:
        model_dir: Directory containing saved artifacts.

    Returns:
        Tuple of (model, preprocessor, metadata).
    """
    model_path = os.path.join(model_dir, "best_model.joblib")
    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    metadata_path = os.path.join(model_dir, "model_metadata.json")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"[INFO] Loaded model: {metadata.get('best_model', 'unknown')}")
    print(f"[INFO] Model version: {metadata.get('model_version', 'unknown')}")

    return model, preprocessor, metadata


def build_metadata(
    best_model_name: str,
    metrics: dict,
    target_column: str = "Exited",
    model_version: str = "v1.0.0",
) -> dict:
    """
    Build a metadata dictionary for the trained model.

    Args:
        best_model_name: Name of the best model.
        metrics: Evaluation metrics dictionary.
        target_column: Name of the target column.
        model_version: Version string.

    Returns:
        Metadata dictionary.
    """
    metadata = {
        "project_name": "BankChurnPredict",
        "model_version": model_version,
        "problem_type": "binary_classification",
        "target_column": target_column,
        "best_model": best_model_name,
        "optimal_threshold": metrics.get("optimal_threshold", 0.5),
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "confusion_matrix": metrics["confusion_matrix"],
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return metadata
