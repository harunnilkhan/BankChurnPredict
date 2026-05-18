"""
Health check endpoint for BankChurnPredict.
"""

from fastapi import APIRouter, Depends, Response
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.schemas.response_schema import HealthResponse
from app.services.model_loader import model_loader
from app.db.database import get_db

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API service and its dependencies are running.",
)
async def health_check(
    response: Response,
    db: Session = Depends(get_db),
) -> HealthResponse:
    """Return the health status of the API including model and database checks."""
    # Check model artifacts
    is_model_loaded = model_loader.is_loaded

    # Check database connectivity
    db_connected = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_connected = False

    # Determine overall status
    if is_model_loaded and db_connected:
        status = "ok"
    else:
        status = "degraded"
        response.status_code = 503

    return HealthResponse(
        status=status,
        service="BankChurnPredict API",
        model_loaded=is_model_loaded,
        database_connected=db_connected,
    )
