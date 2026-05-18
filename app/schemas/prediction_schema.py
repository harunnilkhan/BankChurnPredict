"""
Pydantic schemas for prediction requests.

Defines the input validation schemas for single and batch predictions
based on the bank customer churn dataset features.
"""

from pydantic import BaseModel, Field
from typing import List, Literal


class ChurnPredictionRequest(BaseModel):
    """Schema for a single churn prediction request."""

    CreditScore: int = Field(
        ...,
        ge=300,
        le=900,
        description="Customer's credit score (300-900)",
        examples=[619],
    )
    Geography: Literal["France", "Germany", "Spain"] = Field(
        ...,
        description="Customer's country (France, Spain, Germany)",
        examples=["France"],
    )
    Gender: Literal["Female", "Male"] = Field(
        ...,
        description="Customer's gender (Male, Female)",
        examples=["Female"],
    )
    Age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Customer's age",
        examples=[42],
    )
    Tenure: int = Field(
        ...,
        ge=0,
        le=10,
        description="Number of years as bank customer",
        examples=[2],
    )
    Balance: float = Field(
        ...,
        ge=0,
        description="Customer's account balance",
        examples=[0.0],
    )
    NumOfProducts: int = Field(
        ...,
        ge=1,
        le=4,
        description="Number of bank products used",
        examples=[1],
    )
    HasCrCard: int = Field(
        ...,
        ge=0,
        le=1,
        description="Has credit card (0=No, 1=Yes)",
        examples=[1],
    )
    IsActiveMember: int = Field(
        ...,
        ge=0,
        le=1,
        description="Is active member (0=No, 1=Yes)",
        examples=[1],
    )
    EstimatedSalary: float = Field(
        ...,
        ge=0,
        description="Customer's estimated salary",
        examples=[101348.88],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request."""

    instances: List[ChurnPredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of prediction requests (max 1000)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "instances": [
                        {
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
                        },
                        {
                            "CreditScore": 850,
                            "Geography": "Spain",
                            "Gender": "Male",
                            "Age": 33,
                            "Tenure": 7,
                            "Balance": 76548.6,
                            "NumOfProducts": 1,
                            "HasCrCard": 0,
                            "IsActiveMember": 1,
                            "EstimatedSalary": 98453.45,
                        },
                    ]
                }
            ]
        }
    }
