"""
SQLAlchemy database models for BankChurnPredict.

Defines the PredictionLog table for storing prediction history.
"""

import json
from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, Text, DateTime

from app.db.database import Base


class PredictionLog(Base):
    """
    Table for storing prediction requests and results.
    Each row represents one prediction made by the API.
    """

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    input_data = Column(Text, nullable=False)  # JSON string of input features
    prediction = Column(Integer, nullable=False)
    label = Column(String(50), nullable=False)
    probability = Column(Float, nullable=False)
    model_version = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict:
        """Convert the record to a dictionary for API responses."""
        return {
            "id": self.id,
            "input_data": json.loads(self.input_data) if self.input_data else {},
            "prediction": self.prediction,
            "label": self.label,
            "probability": self.probability,
            "model_version": self.model_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return (
            f"<PredictionLog(id={self.id}, prediction={self.prediction}, "
            f"label='{self.label}', probability={self.probability})>"
        )
