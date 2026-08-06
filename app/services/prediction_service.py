"""
Prediction service layer.

Sits between the API routers and the model registry, keeping routers thin
(HTTP concerns only) and business/ML logic reusable, unit-testable, and
free of any FastAPI dependency.
"""

from app.ml.model_registry import ModelRegistry
from app.schemas.dewpoint import DewPointRequest
from app.schemas.fire import FireRequest
from app.schemas.fire_nrt import FireNRTRequest
from app.schemas.temperature import TemperatureRequest
from app.utils.constants import ModelName


class PredictionService:
    """Provides one method per prediction use case, backed by the registry."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def predict_temperature(self, request: TemperatureRequest) -> float:
        model = self._registry.get(ModelName.TEMPERATURE.value)
        return model.predict(request.to_feature_vector())

    def predict_dewpoint(self, request: DewPointRequest) -> float:
        model = self._registry.get(ModelName.DEWPOINT.value)
        return model.predict(request.to_feature_vector())

    def predict_fire(self, request: FireRequest) -> float:
        model = self._registry.get(ModelName.FIRE.value)
        return model.predict(request.to_feature_vector())

    def predict_fire_nrt(self, request: FireNRTRequest) -> float:
        model = self._registry.get(ModelName.FIRE_NRT.value)
        return model.predict(request.to_feature_vector())
