"""
Prediction endpoints for BankChurnPredict.

Includes single prediction, batch prediction, and prediction history.
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.prediction_schema import ChurnPredictionRequest, BatchPredictionRequest
from app.schemas.response_schema import (
    PredictionResponse,
    BatchPredictionResponse,
    PredictionHistoryResponse,
)
from app.services.prediction_service import make_single_prediction, make_batch_prediction
from app.services.history_service import get_prediction_history
from app.core.logging import logger

router = APIRouter(tags=["Predictions"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Single Prediction",
    description="Make a single churn prediction for one customer.",
)
async def predict_single(
    request: ChurnPredictionRequest,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """
    Predict whether a customer will churn based on their features.

    Returns prediction (0 or 1), human-readable label, probability score,
    and model version.
    """
    try:
        input_data = request.model_dump()
        result = make_single_prediction(input_data, db)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Prediction",
    description="Make churn predictions for multiple customers at once.",
)
async def predict_batch(
    request: BatchPredictionRequest,
    db: Session = Depends(get_db),
) -> BatchPredictionResponse:
    """
    Predict churn for a batch of customers (max 1000).

    Returns a list of predictions with labels and probabilities.
    """
    try:
        instances = [inst.model_dump() for inst in request.instances]
        results = make_batch_prediction(instances, db)
        return BatchPredictionResponse(results=results, count=len(results))
    except Exception as e:
        logger.error(f"Batch prediction endpoint error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@router.get(
    "/predictions/history",
    response_model=PredictionHistoryResponse,
    summary="Prediction History",
    description="Retrieve the history of past predictions.",
)
async def prediction_history(
    limit: int = Query(default=20, ge=1, le=100, description="Max records to return"),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
    db: Session = Depends(get_db),
) -> PredictionHistoryResponse:
    """
    Get paginated prediction history from the database.

    Supports limit and offset query parameters for pagination.
    """
    try:
        history = get_prediction_history(db, limit=limit, offset=offset)
        return PredictionHistoryResponse(**history)
    except Exception as e:
        logger.error(f"History endpoint error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve history: {str(e)}"
        )
