"""
Pydantic schemas for API responses.

Defines the response models for all API endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = {"protected_namespaces": ()}

    status: str = Field(..., examples=["ok"])
    service: str = Field(..., examples=["BankChurnPredict API"])
    model_loaded: bool = Field(..., examples=[True])
    database_connected: bool = Field(..., examples=[True])


class MetricsDetail(BaseModel):
    """Model evaluation metrics."""

    accuracy: float = Field(..., examples=[0.81])
    precision: float = Field(..., examples=[0.76])
    recall: float = Field(..., examples=[0.69])
    f1_score: float = Field(..., examples=[0.72])
    roc_auc: Optional[float] = Field(None, examples=[0.84])
    confusion_matrix: Optional[List[List[int]]] = Field(
        None,
        examples=[[[1557, 36], [229, 178]]],
    )


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_config = {"protected_namespaces": ()}

    project_name: str = Field(..., examples=["BankChurnPredict"])
    model_version: str = Field(..., examples=["v1.0.0"])
    problem_type: str = Field(..., examples=["binary_classification"])
    target_column: str = Field(..., examples=["Exited"])
    best_model: str = Field(..., examples=["RandomForestClassifier"])
    metrics: MetricsDetail
    created_at: str = Field(..., examples=["2026-05-13 12:00:00"])


class PredictionResponse(BaseModel):
    """Single prediction response."""

    model_config = {"protected_namespaces": ()}

    prediction: int = Field(..., examples=[1])
    label: str = Field(..., examples=["Churn Risk"])
    probability: float = Field(..., examples=[0.82])
    model_version: str = Field(..., examples=["v1.0.0"])


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    results: List[PredictionResponse]
    count: int = Field(..., examples=[2])


class PredictionHistoryItem(BaseModel):
    """Single prediction history item."""

    model_config = {"protected_namespaces": ()}

    id: int
    input_data: Any
    prediction: int
    label: str
    probability: float
    model_version: str
    created_at: str


class PredictionHistoryResponse(BaseModel):
    """Prediction history response."""

    items: List[PredictionHistoryItem]
    total: int = Field(..., examples=[10])
    limit: int = Field(..., examples=[20])
    offset: int = Field(..., examples=[0])


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str = Field(..., examples=["An error occurred"])
    status_code: int = Field(..., examples=[400])
