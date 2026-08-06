"""Fire ML insight endpoints: regional predictions and model metadata."""

from fastapi import APIRouter, Depends, Request

from app.ml.model_registry import model_registry
from app.schemas.ml import MLPredictionsResponse, ModelMetaResponse
from app.services.ml_insights_service import MLInsightsService

router = APIRouter(tags=["Machine Learning Insights"])


def get_ml_insights_service(request: Request) -> MLInsightsService:
    registry = getattr(request.app.state, "model_registry", model_registry)
    return MLInsightsService(registry)


@router.get(
    "/ml/predictions",
    response_model=MLPredictionsResponse,
    summary="Regional fire radiative power predictions",
    description=(
        "Computes live fire radiative power (FRP) predictions and rule-based "
        "severity classification for a fixed set of Indian fire-prone regions, "
        "across seasons and brightness levels, using the deployed fire "
        "prediction model."
    ),
)
def get_ml_predictions(
    service: MLInsightsService = Depends(get_ml_insights_service),
) -> MLPredictionsResponse:
    return service.build_predictions()


@router.get(
    "/ml/model-meta",
    response_model=ModelMetaResponse,
    summary="Fire model metadata",
    description="Returns metadata and feature importance for the deployed fire prediction model.",
)
def get_model_meta(
    service: MLInsightsService = Depends(get_ml_insights_service),
) -> ModelMetaResponse:
    return service.build_model_meta()
