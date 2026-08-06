"""
Model registry.

A single, process-wide cache of every ML model the API serves. Models are
loaded once at application startup (see `app.main`'s lifespan handler) and
reused for the lifetime of the process -- avoiding repeated disk I/O and
deserialization cost on every request.
"""

from typing import Dict, Iterable

from app.core.config import settings
from app.core.exceptions import ModelNotFoundError
from app.core.logging_config import get_logger
from app.ml.base import LoadedModel
from app.utils.constants import (
    DEWPOINT_FEATURES,
    FIRE_FEATURES,
    FIRE_NRT_FEATURES,
    ModelDisplayName,
    ModelName,
    TEMPERATURE_FEATURES,
)

logger = get_logger("ml.registry")


class ModelRegistry:
    """Owns and manages the lifecycle of every loaded ML model."""

    def __init__(self) -> None:
        self._models: Dict[str, LoadedModel] = {
            ModelName.TEMPERATURE.value: LoadedModel(
                name=ModelName.TEMPERATURE.value,
                display_name=ModelDisplayName.TEMPERATURE.value,
                path=settings.ML_MODELS_DIR / settings.TEMPERATURE_MODEL_FILENAME,
                expected_features=TEMPERATURE_FEATURES,
            ),
            ModelName.DEWPOINT.value: LoadedModel(
                name=ModelName.DEWPOINT.value,
                display_name=ModelDisplayName.DEWPOINT.value,
                path=settings.ML_MODELS_DIR / settings.DEWPOINT_MODEL_FILENAME,
                expected_features=DEWPOINT_FEATURES,
            ),
            ModelName.FIRE.value: LoadedModel(
                name=ModelName.FIRE.value,
                display_name=ModelDisplayName.FIRE.value,
                path=settings.ML_MODELS_DIR / settings.FIRE_MODEL_FILENAME,
                expected_features=FIRE_FEATURES,
            ),
            ModelName.FIRE_NRT.value: LoadedModel(
                name=ModelName.FIRE_NRT.value,
                display_name=ModelDisplayName.FIRE_NRT.value,
                path=settings.ML_MODELS_DIR / settings.FIRE_NRT_MODEL_FILENAME,
                expected_features=FIRE_NRT_FEATURES,
            ),
        }

    def load_all(self) -> None:
        """Load every registered model, logging successes and failures."""
        for model in self._models.values():
            model.load()

        failed = [m.name for m in self._models.values() if not m.is_loaded]
        if failed:
            logger.warning("models_failed_to_load models=%s", failed)
            if settings.MODEL_LOAD_STRICT:
                raise RuntimeError(
                    f"Failed to load required model(s): {failed}. "
                    "Set MODEL_LOAD_STRICT=false to allow degraded startup."
                )
        else:
            logger.info("all_models_loaded count=%d", len(self._models))

    def get(self, name: str) -> LoadedModel:
        """Fetch a model wrapper by its registry name."""
        model = self._models.get(name)
        if model is None:
            raise ModelNotFoundError(f"Unknown model: '{name}'", details={"available": list(self._models)})
        return model

    def all(self) -> Iterable[LoadedModel]:
        return self._models.values()


# Process-wide singleton, populated during the FastAPI lifespan startup hook.
model_registry = ModelRegistry()
