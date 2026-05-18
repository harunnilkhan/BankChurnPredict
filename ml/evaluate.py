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
    precision_recall_curve,
)


def _find_optimal_threshold(y_test: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Find the probability threshold that maximises F1-score on the test set
    using the precision-recall curve.

    Args:
        y_test: True labels.
        y_proba: Predicted probabilities for the positive class.

    Returns:
        Optimal threshold value (float).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    # precision_recall_curve returns arrays where the last precision/recall
    # entry has no corresponding threshold, so we slice to match lengths.
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1]),
        0.0,
    )

    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a trained classification model on test data.

    Args:
        model: Trained scikit-learn model.
        X_test: Preprocessed test features.
        y_test: True test labels.

    Returns:
        Dictionary containing all evaluation metrics and optimal threshold.
    """
    y_pred = model.predict(X_test)

    # Get probability scores for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        optimal_threshold = _find_optimal_threshold(y_test, y_proba)
    else:
        y_proba = None
        roc_auc = None
        optimal_threshold = 0.5

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "optimal_threshold": round(optimal_threshold, 4),
    }

    return metrics


def compare_models(results: dict, primary_metric: str = "f1_score") -> str:
    """
    Compare multiple model results and return the name of the best model.

    Uses f1_score by default because it balances precision and recall,
    which is critical for imbalanced churn prediction problems.

    Args:
        results: Dictionary of {model_name: metrics_dict}.
        primary_metric: Metric to use for comparison (default: f1_score).

    Returns:
        Name of the best-performing model.
    """
    fallback_metric = "roc_auc"

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
