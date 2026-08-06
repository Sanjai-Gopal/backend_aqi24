"""Dew point prediction endpoint."""

from fastapi import APIRouter, Depends

from app.api.v1.deps import get_prediction_service, get_request_id
from app.schemas.common import PredictionResponse
from app.schemas.dewpoint import DewPointRequest
from app.services.prediction_service import PredictionService
from app.utils.constants import ModelDisplayName

router = APIRouter(tags=["Dew Point"])


@router.post(
    "/predict/dewpoint",
    response_model=PredictionResponse,
    summary="Predict dew point",
    description=(
        "Predicts dew point (°C) for a given calendar date using a "
        "LightGBM regression model trained on year, month, day, and "
        "day-of-week."
    ),
)
def predict_dewpoint(
    payload: DewPointRequest,
    service: PredictionService = Depends(get_prediction_service),
    request_id: str = Depends(get_request_id),
) -> PredictionResponse:
    prediction = service.predict_dewpoint(payload)
    return PredictionResponse(
        model=ModelDisplayName.DEWPOINT.value,
        prediction=prediction,
        unit="celsius",
        request_id=request_id,
    )
