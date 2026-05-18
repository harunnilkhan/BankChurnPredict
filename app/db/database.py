"""Database configuration and session management for BankChurnPredict."""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings
from app.core.logging import logger


def _engine_options(database_url: str) -> dict:
    """Return SQLAlchemy engine options for the configured database."""
    options = {"echo": False}
    if database_url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
    return options


# Create engine
engine = create_engine(settings.database_url, **_engine_options(settings.database_url))

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()
_db_initialized = False


def init_db() -> None:
    """
    Initialize the database by creating all tables.
    Called on application startup.
    """
    global _db_initialized

    if _db_initialized:
        return

    from app.db.models import PredictionLog  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _db_initialized = True
    logger.info("Database initialized successfully.")


def get_db():
    """
    FastAPI dependency that provides a database session.
    Yields a session and ensures it is closed after use.
    """
    init_db()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
