"""
History service for BankChurnPredict.

Handles saving and retrieving prediction logs from the database.
"""

import json
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.models import PredictionLog
from app.core.logging import logger


def save_prediction_log(
    db: Session,
    input_data: dict,
    result: dict,
) -> PredictionLog:
    """
    Save a prediction request and result to the database.

    Args:
        db: Database session.
        input_data: Input features dictionary.
        result: Prediction result dictionary.

    Returns:
        The created PredictionLog record.
    """
    try:
        log_entry = PredictionLog(
            input_data=json.dumps(input_data, default=str),
            prediction=result["prediction"],
            label=result["label"],
            probability=result["probability"],
            model_version=result["model_version"],
            created_at=datetime.utcnow(),
        )

        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)

        logger.debug(f"Prediction logged: id={log_entry.id}")
        return log_entry

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save prediction log: {e}")
        raise


def save_prediction_logs_bulk(
    db: Session,
    inputs: list[dict],
    results: list[dict],
) -> None:
    """
    Save multiple prediction logs in a single database transaction.

    More efficient than calling save_prediction_log() in a loop because
    it performs a single commit for all records.

    Args:
        db: Database session.
        inputs: List of input feature dictionaries.
        results: List of prediction result dictionaries.
    """
    try:
        entries = []
        now = datetime.utcnow()

        for input_data, result in zip(inputs, results):
            entries.append(
                PredictionLog(
                    input_data=json.dumps(input_data, default=str),
                    prediction=result["prediction"],
                    label=result["label"],
                    probability=result["probability"],
                    model_version=result["model_version"],
                    created_at=now,
                )
            )

        db.add_all(entries)
        db.commit()

        logger.debug(f"Bulk logged {len(entries)} predictions.")

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to bulk save prediction logs: {e}")
        raise


def get_prediction_history(
    db: Session,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """
    Retrieve prediction history from the database.

    Args:
        db: Database session.
        limit: Maximum number of records to return.
        offset: Number of records to skip.

    Returns:
        Dictionary containing items list and pagination info.
    """
    try:
        # Get total count
        total = db.query(func.count(PredictionLog.id)).scalar()

        # Query with pagination, ordered by most recent first
        logs = (
            db.query(PredictionLog)
            .order_by(PredictionLog.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        items = [log.to_dict() for log in logs]

        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        logger.error(f"Failed to retrieve prediction history: {e}")
        raise
