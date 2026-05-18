"""
Tests for prediction endpoints.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# Sample valid input matching the bank churn dataset schema
VALID_INPUT = {
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

# Invalid input (missing required fields)
INVALID_INPUT = {
    "CreditScore": 619,
    "Geography": "France",
    # Missing Gender, Age, etc.
}


def test_single_prediction_valid():
    """Test POST /predict with valid input returns 200."""
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "label" in data
    assert "probability" in data
    assert "model_version" in data
    assert data["prediction"] in [0, 1]
    assert data["label"] in ["Churn Risk", "No Churn Risk"]
    assert 0.0 <= data["probability"] <= 1.0


def test_single_prediction_invalid():
    """Test POST /predict with invalid input returns 422."""
    response = client.post("/predict", json=INVALID_INPUT)
    assert response.status_code == 422


def test_single_prediction_invalid_category():
    """Test POST /predict with unsupported categorical values returns 422."""
    payload = {
        **VALID_INPUT,
        "Geography": "Italy",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_single_prediction_empty_body():
    """Test POST /predict with empty body returns 422."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_batch_prediction_valid():
    """Test POST /predict/batch with valid input returns 200."""
    batch_request = {
        "instances": [VALID_INPUT, VALID_INPUT]
    }
    response = client.post("/predict/batch", json=batch_request)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "count" in data
    assert data["count"] == 2
    assert len(data["results"]) == 2

    for result in data["results"]:
        assert "prediction" in result
        assert "label" in result
        assert "probability" in result
        assert "model_version" in result


def test_batch_prediction_empty_list():
    """Test POST /predict/batch with empty list returns 422."""
    response = client.post("/predict/batch", json={"instances": []})
    assert response.status_code == 422


def test_prediction_history():
    """Test GET /predictions/history returns 200."""
    # First, make a prediction to ensure there's data
    client.post("/predict", json=VALID_INPUT)

    response = client.get("/predictions/history")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "limit" in data
    assert "offset" in data
    assert isinstance(data["items"], list)


def test_prediction_history_with_pagination():
    """Test GET /predictions/history with limit and offset."""
    response = client.get("/predictions/history?limit=5&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert data["limit"] == 5
    assert data["offset"] == 0
