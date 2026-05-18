"""
Application configuration module for BankChurnPredict.

Loads settings from environment variables using pydantic-settings.
"""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = Field(default="BankChurnPredict", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    model_path: str = Field(default="models/best_model.joblib", alias="MODEL_PATH")
    preprocessor_path: str = Field(
        default="models/preprocessor.joblib", alias="PREPROCESSOR_PATH"
    )
    model_metadata_path: str = Field(
        default="models/model_metadata.json", alias="MODEL_METADATA_PATH"
    )
    database_url: str = Field(
        default="sqlite:///./bankchurnpredict.db", alias="DATABASE_URL"
    )

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        protected_namespaces=(),
    )


# Global settings instance
settings = Settings()
