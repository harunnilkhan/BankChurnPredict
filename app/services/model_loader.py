"""
Model loader service for BankChurnPredict.

Provides a singleton-style model loading mechanism
to avoid reloading artifacts on every request.
"""

import os
import json

import joblib

from app.core.config import settings
from app.core.logging import logger


class ModelLoader:
    """
    Singleton model loader that caches model artifacts in memory.
    """

    _instance = None
    _model = None
    _preprocessor = None
    _metadata = None
    _is_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> None:
        """Load model artifacts from disk."""
        if self._is_loaded:
            logger.info("Model artifacts already loaded, skipping.")
            return

        try:
            self._model = joblib.load(settings.model_path)
            logger.info(f"Model loaded from: {settings.model_path}")

            self._preprocessor = joblib.load(settings.preprocessor_path)
            logger.info(f"Preprocessor loaded from: {settings.preprocessor_path}")

            with open(settings.model_metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
            logger.info(f"Metadata loaded from: {settings.model_metadata_path}")

            self._is_loaded = True
            logger.info(
                f"All artifacts loaded. Model: {self._metadata.get('best_model')}, "
                f"Version: {self._metadata.get('model_version')}"
            )
        except FileNotFoundError as e:
            logger.error(f"Model artifacts not found: {e}")
            logger.error("Please run 'python -m ml.train' first to generate model artifacts.")
            raise
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}")
            raise

    @property
    def model(self):
        if not self._is_loaded:
            self.load()
        return self._model

    @property
    def preprocessor(self):
        if not self._is_loaded:
            self.load()
        return self._preprocessor

    @property
    def metadata(self) -> dict:
        if not self._is_loaded:
            self.load()
        return self._metadata

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def reload(self) -> None:
        """Force reload of all artifacts."""
        self._is_loaded = False
        self._model = None
        self._preprocessor = None
        self._metadata = None
        self.load()


# Global singleton instance
model_loader = ModelLoader()
