"""Fire archive (MODIS) prediction endpoint."""

from fastapi import APIRouter, Depends

from app.api.v1.deps import get_prediction_service, get_request_id
from app.schemas.common import PredictionResponse
from app.schemas.fire import FireRequest
from app.services.prediction_service import PredictionService
from app.utils.constants import ModelDisplayName

router = APIRouter(tags=["Fire"])


@router.post(
    "/predict/fire",
    response_model=PredictionResponse,
    summary="Predict fire archive value",
    description=(
        "Predicts a fire-related target from archived satellite fire-pixel "
        "data using a LightGBM regression model trained on latitude, "
        "longitude, brightness, scan, track, year, month, and day."
    ),
)
def predict_fire(
    payload: FireRequest,
    service: PredictionService = Depends(get_prediction_service),
    request_id: str = Depends(get_request_id),
) -> PredictionResponse:
    prediction = service.predict_fire(payload)
    return PredictionResponse(
        model=ModelDisplayName.FIRE.value,
        prediction=prediction,
        request_id=request_id,
    )
