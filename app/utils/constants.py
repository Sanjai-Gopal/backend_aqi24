"""Shared constants used across the application."""

from enum import Enum


class ModelName(str, Enum):
    """Canonical identifiers for every ML model served by this API."""

    TEMPERATURE = "temperature_model"
    DEWPOINT = "dewpoint_model"
    FIRE = "fire_prediction_model"
    FIRE_NRT = "fire_nrt_prediction_model"


class ModelDisplayName(str, Enum):
    """Human-readable model names returned in API responses."""

    TEMPERATURE = "Temperature Model"
    DEWPOINT = "Dew Point Model"
    FIRE = "Fire Archive Model"
    FIRE_NRT = "Fire NRT Model"


# Ordered feature vectors expected by each underlying estimator.
# These MUST match the exact order the models were trained with.
TEMPERATURE_FEATURES = ["year", "month", "day", "dayofweek"]
DEWPOINT_FEATURES = ["year", "month", "day", "dayofweek"]
FIRE_FEATURES = ["latitude", "longitude", "brightness", "scan", "track", "year", "month", "day"]
FIRE_NRT_FEATURES = ["latitude", "longitude", "brightness", "scan", "track", "year", "month", "day"]

# Reasonable physical / domain bounds used for request validation.
MIN_YEAR = 1900
MAX_YEAR = 2100
MIN_LATITUDE = -90.0
MAX_LATITUDE = 90.0
MIN_LONGITUDE = -180.0
MAX_LONGITUDE = 180.0

REQUEST_ID_HEADER = "X-Request-ID"
PROCESS_TIME_HEADER = "X-Process-Time-Ms"
