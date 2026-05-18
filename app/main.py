"""
BankChurnPredict — FastAPI Application Entry Point.

This is the main application module that:
- Creates the FastAPI app instance
- Registers API routers
- Initializes the database on startup
- Loads ML model artifacts on startup
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_health import router as health_router
from app.api.routes_model import router as model_router
from app.api.routes_prediction import router as prediction_router
from app.db.database import init_db
from app.services.model_loader import model_loader
from app.core.config import settings
from app.core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Runs startup logic before the app begins serving requests,
    and cleanup logic when the app shuts down.
    """
    # --- Startup ---
    logger.info(f"Starting {settings.app_name}...")
    logger.info(f"Environment: {settings.app_env}")

    # Initialize database (creates tables if they don't exist)
    init_db()

    # Load ML model artifacts
    try:
        model_loader.load()
        logger.info("Model artifacts loaded successfully.")
    except FileNotFoundError:
        logger.warning(
            "Model artifacts not found. "
            "Run 'python -m ml.train' to generate them. "
            "Prediction endpoints will fail until artifacts are available."
        )

    logger.info(f"{settings.app_name} is ready to serve requests.")

    yield

    # --- Shutdown ---
    logger.info(f"{settings.app_name} shutting down.")


# --- Create FastAPI Application ---
app = FastAPI(
    title="BankChurnPredict API",
    description=(
        "A production-style Machine Learning deployment platform for "
        "customer churn prediction. Provides REST endpoints for model "
        "inference, batch predictions, and prediction history logging."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Register Routers ---
app.include_router(health_router)
app.include_router(model_router)
app.include_router(prediction_router)


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint that redirects to API documentation."""
    return {
        "message": "Welcome to BankChurnPredict API",
        "docs": "/docs",
        "health": "/health",
    }
