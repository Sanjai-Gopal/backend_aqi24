"""Fire NRT (near-real-time) prediction endpoint."""

from fastapi import APIRouter, Depends

from app.api.v1.deps import get_prediction_service, get_request_id
from app.schemas.common import PredictionResponse
from app.schemas.fire_nrt import FireNRTRequest
from app.services.prediction_service import PredictionService
from app.utils.constants import ModelDisplayName

router = APIRouter(tags=["Fire NRT"])


@router.post(
    "/predict/fire-nrt",
    response_model=PredictionResponse,
    summary="Predict fire NRT value",
    description=(
        "Predicts a fire-related target from near-real-time satellite "
        "fire-pixel data. IMPORTANT: this model was trained with exactly "
        "8 features: latitude, longitude, brightness, scan, track, year, "
        "month, and day."
    ),
)
def predict_fire_nrt(
    payload: FireNRTRequest,
    service: PredictionService = Depends(get_prediction_service),
    request_id: str = Depends(get_request_id),
) -> PredictionResponse:
    prediction = service.predict_fire_nrt(payload)
    return PredictionResponse(
        model=ModelDisplayName.FIRE_NRT.value,
        prediction=prediction,
        request_id=request_id,
    )
