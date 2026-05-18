"""
Model evaluation module for BankChurnPredict.

Provides functions to evaluate classification models and compare
multiple model results to select the best performer.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a trained classification model on test data.

    Args:
        model: Trained scikit-learn model.
        X_test: Preprocessed test features.
        y_test: True test labels.

    Returns:
        Dictionary containing all evaluation metrics.
    """
    y_pred = model.predict(X_test)

    # Get probability scores for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        roc_auc = None

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return metrics


def compare_models(results: dict, primary_metric: str = "roc_auc") -> str:
    """
    Compare multiple model results and return the name of the best model.

    Args:
        results: Dictionary of {model_name: metrics_dict}.
        primary_metric: Metric to use for comparison (default: roc_auc).

    Returns:
        Name of the best-performing model.
    """
    fallback_metric = "f1_score"

    best_model = None
    best_score = -1.0

    print("\n" + "=" * 70)
    print(f"{'Model':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10}")
    print("=" * 70)

    for model_name, metrics in results.items():
        # Determine which metric to compare
        score = metrics.get(primary_metric)
        if score is None:
            score = metrics.get(fallback_metric, 0.0)

        print(
            f"{model_name:<30} "
            f"{metrics['accuracy']:<10.4f} "
            f"{metrics['precision']:<10.4f} "
            f"{metrics['recall']:<10.4f} "
            f"{metrics['f1_score']:<10.4f} "
            f"{str(metrics.get('roc_auc', 'N/A')):<10}"
        )

        if score > best_score:
            best_score = score
            best_model = model_name

    print("=" * 70)
    print(f"\n[BEST] {best_model} (based on {primary_metric} = {best_score:.4f})")

    return best_model


def print_classification_report(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Print a detailed classification report for a model.

    Args:
        model: Trained scikit-learn model.
        X_test: Preprocessed test features.
        y_test: True test labels.
    """
    y_pred = model.predict(X_test)
    target_names = ["No Churn (0)", "Churn (1)"]
    report = classification_report(y_test, y_pred, target_names=target_names)
    print("\nClassification Report:")
    print(report)
