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


def _find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Find the probability threshold that maximises F1-score using the
    precision-recall curve.

    Should be called with a **validation** set so that threshold selection
    remains independent of the held-out test set used for final reporting.

    Args:
        y_true: True labels (validation set).
        y_proba: Predicted probabilities for the positive class (validation set).

    Returns:
        Optimal threshold value (float).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # precision_recall_curve returns arrays where the last precision/recall
    # entry has no corresponding threshold, so we slice to match lengths.
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1]),
        0.0,
    )

    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
) -> dict:
    """
    Evaluate a trained classification model on test data.

    Threshold selection strategy
    ----------------------------
    When ``X_val`` and ``y_val`` are provided the optimal decision threshold
    is determined on the **validation set** and only *applied* to the test
    set.  This prevents the test set from being used for both model selection
    and performance reporting, which would produce overly optimistic metrics.

    When no validation set is supplied (legacy / quick-eval mode) the
    threshold is found on the test set itself — acceptable for exploration
    but not recommended for final reporting.

    All classification metrics (accuracy, precision, recall, F1, confusion
    matrix) are computed from the **threshold-adjusted** predictions so that
    the reported figures match the actual inference behaviour of the API.

    Args:
        model: Trained scikit-learn model.
        X_test: Preprocessed test features.
        y_test: True test labels.
        X_val: Preprocessed validation features (keyword-only, optional).
        y_val: True validation labels (keyword-only, optional).

    Returns:
        Dictionary containing all evaluation metrics and the optimal threshold.
    """
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)

        if X_val is not None and y_val is not None:
            # Preferred path: threshold selected on validation set
            y_val_proba = model.predict_proba(X_val)[:, 1]
            optimal_threshold = _find_optimal_threshold(
                np.asarray(y_val), y_val_proba
            )
            print(
                f"[INFO] Threshold tuned on validation set: {optimal_threshold:.4f}"
            )
        else:
            # Fallback: threshold selected on test set (exploration mode)
            optimal_threshold = _find_optimal_threshold(
                np.asarray(y_test), y_proba
            )
            print(
                "[WARN] No validation set provided — threshold tuned on test set. "
                "Reported metrics may be slightly optimistic."
            )

        # Apply threshold to produce final predictions
        y_pred = (y_proba >= optimal_threshold).astype(int)
    else:
        # Model without probability support — fall back to hard predictions
        y_pred = model.predict(X_test)
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
