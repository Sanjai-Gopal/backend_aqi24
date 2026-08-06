"""Shared FastAPI dependencies for API v1."""

from fastapi import Request

from app.ml.model_registry import model_registry
from app.services.prediction_service import PredictionService


def get_prediction_service(request: Request) -> PredictionService:
    """
    Dependency-injects a `PredictionService` bound to the app's model
    registry.

    Reading the registry off `app.state` (rather than importing the global
    singleton directly) makes it trivial to override in tests via
    `app.dependency_overrides[get_prediction_service]`.
    """
    registry = getattr(request.app.state, "model_registry", model_registry)
    return PredictionService(registry)


def get_request_id(request: Request) -> str:
    """Return the correlation/request ID attached by RequestLoggingMiddleware."""
    return getattr(request.state, "request_id", "unknown")
