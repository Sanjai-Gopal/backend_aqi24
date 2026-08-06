"""Base wrapper around a single loaded scikit-learn / LightGBM estimator."""

import warnings
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from app.core.exceptions import ModelNotLoadedError, PredictionError
from app.core.logging_config import get_logger

logger = get_logger("ml.base")


class LoadedModel:
    """
    Wraps a single joblib-serialized estimator with:
      * lazy, cached loading from disk
      * feature-count validation before inference
      * consistent error handling / logging
      * prediction timing
    """

    def __init__(self, name: str, display_name: str, path: Path, expected_features: Sequence[str]):
        self.name = name
        self.display_name = display_name
        self.path = path
        self.expected_features: List[str] = list(expected_features)
        self._estimator = None
        self._load_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._estimator is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def estimator(self):
        return self._estimator

    def load(self) -> None:
        """Load the estimator from disk into memory (idempotent)."""
        if self._estimator is not None:
            return

        if not self.path.exists():
            self._load_error = f"Model file not found: {self.path}"
            logger.error("model_load_failed name=%s reason=%s", self.name, self._load_error)
            return

        try:
            start = perf_counter()
            self._estimator = joblib.load(self.path)
            duration_ms = (perf_counter() - start) * 1000
            logger.info(
                "model_loaded name=%s file=%s duration_ms=%.2f",
                self.name,
                self.path.name,
                duration_ms,
            )
        except Exception as exc:  # noqa: BLE001 - we want to capture and report any load failure
            self._load_error = str(exc)
            logger.exception("model_load_failed name=%s file=%s", self.name, self.path.name)

    def predict(self, features: Sequence[float]) -> float:
        """
        Run inference on a single ordered feature vector.

        Raises:
            ModelNotLoadedError: the estimator failed to load at startup.
            PredictionError: the underlying `.predict()` call raised.
        """
        if not self.is_loaded:
            raise ModelNotLoadedError(
                f"'{self.display_name}' is not available (failed to load).",
                details={"model": self.name, "reason": self._load_error},
            )

        if len(features) != len(self.expected_features):
            raise PredictionError(
                f"'{self.display_name}' expects {len(self.expected_features)} features, "
                f"received {len(features)}.",
                details={"expected_features": self.expected_features},
            )

        try:
            start = perf_counter()
            # Some estimators (e.g. RandomForestRegressor) were fitted on a
            # DataFrame and remember `feature_names_in_`; passing a DataFrame
            # with matching column names avoids a spurious sklearn warning
            # and keeps column-order intent explicit.
            if getattr(self._estimator, "feature_names_in_", None) is not None:
                vector = pd.DataFrame([features], columns=self.expected_features)
            else:
                vector = np.asarray([features], dtype=float)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                raw_prediction = self._estimator.predict(vector)
            duration_ms = (perf_counter() - start) * 1000
            logger.debug(
                "prediction_ok name=%s duration_ms=%.3f",
                self.name,
                duration_ms,
            )
            return float(raw_prediction[0])
        except PredictionError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface any estimator failure uniformly
            logger.exception("prediction_failed name=%s", self.name)
            raise PredictionError(
                f"Prediction failed for '{self.display_name}'.",
                details={"model": self.name, "reason": str(exc)},
            ) from exc
