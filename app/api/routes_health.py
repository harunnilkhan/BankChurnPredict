"""
Health check endpoint for BankChurnPredict.
"""

from fastapi import APIRouter

from app.schemas.response_schema import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API service is running.",
)
async def health_check() -> HealthResponse:
    """Return the health status of the API."""
    return HealthResponse(
        status="ok",
        service="BankChurnPredict API",
    )
