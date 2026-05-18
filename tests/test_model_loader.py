"""
Tests for model loader service.
"""

import os
import json


def test_model_file_exists():
    """Test that the model artifact file exists."""
    assert os.path.exists("models/best_model.joblib"), \
        "Model file not found. Run 'python -m ml.train' first."


def test_preprocessor_file_exists():
    """Test that the preprocessor artifact file exists."""
    assert os.path.exists("models/preprocessor.joblib"), \
        "Preprocessor file not found. Run 'python -m ml.train' first."


def test_metadata_file_exists():
    """Test that the metadata file exists."""
    assert os.path.exists("models/model_metadata.json"), \
        "Metadata file not found. Run 'python -m ml.train' first."


def test_metadata_is_valid_json():
    """Test that model_metadata.json contains valid JSON."""
    with open("models/model_metadata.json", "r") as f:
        metadata = json.load(f)

    assert isinstance(metadata, dict)


def test_metadata_has_required_fields():
    """Test that metadata contains all required fields."""
    with open("models/model_metadata.json", "r") as f:
        metadata = json.load(f)

    required_fields = [
        "project_name",
        "model_version",
        "problem_type",
        "target_column",
        "best_model",
        "metrics",
        "created_at",
    ]

    for field in required_fields:
        assert field in metadata, f"Missing required field: {field}"


def test_metadata_metrics_fields():
    """Test that metrics contain all required evaluation metrics."""
    with open("models/model_metadata.json", "r") as f:
        metadata = json.load(f)

    metrics = metadata["metrics"]
    required_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "confusion_matrix",
    ]

    for metric in required_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
        if metric == "confusion_matrix":
            assert isinstance(metrics[metric], list)
        elif metrics[metric] is not None:
            assert 0.0 <= metrics[metric] <= 1.0, \
                f"Metric {metric} out of range: {metrics[metric]}"


def test_model_loader_singleton():
    """Test that model loader works as a singleton."""
    from app.services.model_loader import ModelLoader

    loader1 = ModelLoader()
    loader2 = ModelLoader()
    assert loader1 is loader2
