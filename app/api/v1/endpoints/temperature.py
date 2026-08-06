"""Temperature prediction endpoint."""

from fastapi import APIRouter, Depends

from app.api.v1.deps import get_prediction_service, get_request_id
from app.schemas.common import PredictionResponse
from app.schemas.temperature import TemperatureRequest
from app.services.prediction_service import PredictionService
from app.utils.constants import ModelDisplayName

router = APIRouter(tags=["Temperature"])


@router.post(
    "/predict/temperature",
    response_model=PredictionResponse,
    summary="Predict temperature",
    description=(
        "Predicts temperature (°C) for a given calendar date using a "
        "RandomForest regression model trained on year, month, day, and "
        "day-of-week."
    ),
)
def predict_temperature(
    payload: TemperatureRequest,
    service: PredictionService = Depends(get_prediction_service),
    request_id: str = Depends(get_request_id),
) -> PredictionResponse:
    prediction = service.predict_temperature(payload)
    return PredictionResponse(
        model=ModelDisplayName.TEMPERATURE.value,
        prediction=prediction,
        unit="celsius",
        request_id=request_id,
    )
