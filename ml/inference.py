"""
Inference module for BankChurnPredict.

Provides reusable prediction functions that are independent of FastAPI.
Used by the API layer to make single and batch predictions.
"""

import os
import json

import pandas as pd
import joblib

from ml.preprocess import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, VALID_CATEGORIES


# Module-level cache for loaded artifacts
_model = None
_preprocessor = None
_metadata = None
EXPECTED_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def _validate_records(records: list[dict]) -> None:
    """Validate inference input records before passing them to sklearn."""
    if not records:
        raise ValueError("At least one input record is required for prediction.")

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record at index {index} must be a dictionary.")

        missing = [field for field in EXPECTED_FEATURES if field not in record]
        if missing:
            raise ValueError(
                f"Record at index {index} is missing required fields: {missing}"
            )

        for field, allowed_values in VALID_CATEGORIES.items():
            value = record.get(field)
            if value not in allowed_values:
                raise ValueError(
                    f"Record at index {index} has invalid {field}: {value!r}. "
                    f"Expected one of: {allowed_values}"
                )


def load_artifacts(
    model_path: str = None,
    preprocessor_path: str = None,
    metadata_path: str = None,
) -> tuple:
    """
    Load model artifacts from disk. Uses module-level caching to avoid
    reloading on every prediction.

    Args:
        model_path: Path to the saved model file.
        preprocessor_path: Path to the saved preprocessor file.
        metadata_path: Path to the model metadata JSON file.

    Returns:
        Tuple of (model, preprocessor, metadata).
    """
    global _model, _preprocessor, _metadata

    if _model is not None and _preprocessor is not None and _metadata is not None:
        return _model, _preprocessor, _metadata

    if model_path is None:
        model_path = os.environ.get("MODEL_PATH", "models/best_model.joblib")
    if preprocessor_path is None:
        preprocessor_path = os.environ.get("PREPROCESSOR_PATH", "models/preprocessor.joblib")
    if metadata_path is None:
        metadata_path = os.environ.get("MODEL_METADATA_PATH", "models/model_metadata.json")

    _model = joblib.load(model_path)
    _preprocessor = joblib.load(preprocessor_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        _metadata = json.load(f)

    print(f"[INFO] Inference artifacts loaded successfully.")
    print(f"       Model: {_metadata.get('best_model', 'unknown')}")
    print(f"       Version: {_metadata.get('model_version', 'unknown')}")

    return _model, _preprocessor, _metadata


def predict_single(input_data: dict) -> dict:
    """
    Make a single prediction from input data.

    Args:
        input_data: Dictionary containing feature values.
            Expected keys: CreditScore, Geography, Gender, Age, Tenure,
                           Balance, NumOfProducts, HasCrCard, IsActiveMember,
                           EstimatedSalary

    Returns:
        Dictionary with prediction, label, probability, and model_version.
    """
    _validate_records([input_data])
    model, preprocessor, metadata = load_artifacts()

    # Convert input to DataFrame for preprocessing
    df = pd.DataFrame([input_data])

    # Preprocess
    X_processed = preprocessor.transform(df)

    # Predict
    prediction = int(model.predict(X_processed)[0])

    # Get probability
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X_processed)[0][1])
    else:
        probability = float(prediction)

    # Build response
    result = {
        "prediction": prediction,
        "label": "Churn Risk" if prediction == 1 else "No Churn Risk",
        "probability": round(probability, 4),
        "model_version": metadata.get("model_version", "unknown"),
    }

    return result


def predict_batch(input_data: list) -> list:
    """
    Make batch predictions from a list of input data dictionaries.

    Args:
        input_data: List of dictionaries, each containing feature values.

    Returns:
        List of prediction result dictionaries.
    """
    _validate_records(input_data)
    model, preprocessor, metadata = load_artifacts()

    # Convert to DataFrame
    df = pd.DataFrame(input_data)

    # Preprocess
    X_processed = preprocessor.transform(df)

    # Predict
    predictions = model.predict(X_processed)

    # Get probabilities
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_processed)[:, 1]
    else:
        probabilities = predictions.astype(float)

    # Build results
    results = []
    model_version = metadata.get("model_version", "unknown")

    for pred, prob in zip(predictions, probabilities):
        pred_int = int(pred)
        results.append({
            "prediction": pred_int,
            "label": "Churn Risk" if pred_int == 1 else "No Churn Risk",
            "probability": round(float(prob), 4),
            "model_version": model_version,
        })

    return results


def get_model_metadata() -> dict:
    """
    Get the loaded model metadata.

    Returns:
        Model metadata dictionary.
    """
    _, _, metadata = load_artifacts()
    return metadata


def reset_artifacts() -> None:
    """
    Reset cached artifacts. Useful for testing or reloading models.
    """
    global _model, _preprocessor, _metadata
    _model = None
    _preprocessor = None
    _metadata = None


if __name__ == "__main__":
    # Quick test
    sample_input = {
        "CreditScore": 619,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 101348.88,
    }

    print("[TEST] Single prediction:")
    result = predict_single(sample_input)
    print(json.dumps(result, indent=2))

    print("\n[TEST] Batch prediction:")
    batch_results = predict_batch([sample_input, sample_input])
    print(json.dumps(batch_results, indent=2))
