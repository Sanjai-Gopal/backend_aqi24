"""Health check and model-status endpoints."""

from fastapi import APIRouter, Request

from app.core.config import settings
from app.ml.model_registry import model_registry
from app.schemas.common import HealthResponse, ModelHealth

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness / readiness health check",
    description="Reports overall API health and the load status of every ML model.",
)
def health_check(request: Request) -> HealthResponse:
    registry = getattr(request.app.state, "model_registry", model_registry)

    models = [
        ModelHealth(
            name=model.name,
            loaded=model.is_loaded,
            file=model.path.name,
            expected_features=model.expected_features,
        )
        for model in registry.all()
    ]
    all_loaded = all(m.loaded for m in models)

    return HealthResponse(
        status="healthy" if all_loaded else "degraded",
        api="running",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        models=models,
    )


@router.get(
    "/models/status",
    response_model=HealthResponse,
    summary="Detailed model status",
    description="Alias of /health focused on model-loading diagnostics.",
)
def model_status(request: Request) -> HealthResponse:
    return health_check(request)
