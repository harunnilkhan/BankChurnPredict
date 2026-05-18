"""
Tests for health check endpoint.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check_returns_200():
    """Test that /health returns status code 200."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200


def test_health_check_response_body():
    """Test that /health returns expected JSON body."""
    with TestClient(app) as client:
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "BankChurnPredict API"
        assert data["model_loaded"] is True
        assert data["database_connected"] is True


def test_root_endpoint():
    """Test that root / returns welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data


def test_model_info_returns_metadata():
    """Test that /model/info returns model metadata."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"]
    assert data["best_model"]
    assert "metrics" in data
    assert "roc_auc" in data["metrics"]
