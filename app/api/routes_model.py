"""
Model metadata endpoint for BankChurnPredict.
"""

from fastapi import APIRouter

from app.services.model_loader import model_loader
from app.schemas.response_schema import ModelInfoResponse, MetricsDetail

router = APIRouter(prefix="/model", tags=["Model"])


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Model Information",
    description="Get metadata about the currently loaded ML model.",
)
async def get_model_info() -> ModelInfoResponse:
    """Return model metadata including version, type, and evaluation metrics."""
    metadata = model_loader.metadata

    metrics = MetricsDetail(
        accuracy=metadata["metrics"]["accuracy"],
        precision=metadata["metrics"]["precision"],
        recall=metadata["metrics"]["recall"],
        f1_score=metadata["metrics"]["f1_score"],
        roc_auc=metadata["metrics"].get("roc_auc"),
        confusion_matrix=metadata["metrics"].get("confusion_matrix"),
    )

    return ModelInfoResponse(
        project_name=metadata.get("project_name", "BankChurnPredict"),
        model_version=metadata.get("model_version", "unknown"),
        problem_type=metadata.get("problem_type", "binary_classification"),
        target_column=metadata.get("target_column", "Exited"),
        best_model=metadata.get("best_model", "unknown"),
        metrics=metrics,
        created_at=metadata.get("created_at", "unknown"),
    )
