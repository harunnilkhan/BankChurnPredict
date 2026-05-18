"""
Prediction service for BankChurnPredict.

Orchestrates the prediction workflow: preprocessing, inference, and logging.
"""

import json

import pandas as pd
from sqlalchemy.orm import Session

from app.services.model_loader import model_loader
from app.services.history_service import save_prediction_log, save_prediction_logs_bulk
from app.core.logging import logger


def _get_threshold() -> float:
    """Return the optimal threshold from model metadata, defaulting to 0.5."""
    return model_loader.metadata.get("optimal_threshold", 0.5)


def make_single_prediction(input_data: dict, db: Session) -> dict:
    """
    Make a single prediction and log it to the database.

    Args:
        input_data: Dictionary of input features.
        db: Database session for logging.

    Returns:
        Prediction result dictionary.
    """
    try:
        # Convert to DataFrame for preprocessing
        df = pd.DataFrame([input_data])

        # Preprocess
        X_processed = model_loader.preprocessor.transform(df)

        # Get probability and apply optimal threshold
        threshold = _get_threshold()
        if hasattr(model_loader.model, "predict_proba"):
            probability = float(model_loader.model.predict_proba(X_processed)[0][1])
            prediction = int(probability >= threshold)
        else:
            prediction = int(model_loader.model.predict(X_processed)[0])
            probability = float(prediction)

        # Build result
        result = {
            "prediction": prediction,
            "label": "Churn Risk" if prediction == 1 else "No Churn Risk",
            "probability": round(probability, 4),
            "model_version": model_loader.metadata.get("model_version", "unknown"),
        }

        # Log to database
        save_prediction_log(db, input_data, result)

        logger.info(
            f"Prediction made: {result['label']} "
            f"(prob={result['probability']:.4f}, threshold={threshold})"
        )

        return result

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def make_batch_prediction(instances: list, db: Session) -> list:
    """
    Make batch predictions and log them to the database.

    Args:
        instances: List of input feature dictionaries.
        db: Database session for logging.

    Returns:
        List of prediction result dictionaries.
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(instances)

        # Preprocess
        X_processed = model_loader.preprocessor.transform(df)

        # Get probabilities and apply optimal threshold
        threshold = _get_threshold()
        if hasattr(model_loader.model, "predict_proba"):
            probabilities = model_loader.model.predict_proba(X_processed)[:, 1]
            predictions = (probabilities >= threshold).astype(int)
        else:
            predictions = model_loader.model.predict(X_processed)
            probabilities = predictions.astype(float)

        model_version = model_loader.metadata.get("model_version", "unknown")

        # Build results
        results = []
        for pred, prob in zip(predictions, probabilities):
            pred_int = int(pred)
            result = {
                "prediction": pred_int,
                "label": "Churn Risk" if pred_int == 1 else "No Churn Risk",
                "probability": round(float(prob), 4),
                "model_version": model_version,
            }
            results.append(result)

        # Bulk log all predictions in a single commit
        save_prediction_logs_bulk(db, instances, results)

        logger.info(f"Batch prediction completed: {len(results)} predictions made.")

        return results

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise
