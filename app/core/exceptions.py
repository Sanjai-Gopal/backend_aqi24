"""
Custom application exceptions.

Every exception carries a machine-readable `error_code` and an HTTP
`status_code` so the global exception handlers in `app.main` can translate
them into consistent, well-structured JSON responses.
"""

from typing import Any, Dict, Optional


class AppException(Exception):
    """Base class for all application-specific exceptions."""

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
    ) -> None:
        self.message = message
        self.details = details or {}
        if status_code is not None:
            self.status_code = status_code
        if error_code is not None:
            self.error_code = error_code
        super().__init__(message)


class ModelNotLoadedError(AppException):
    """Raised when a prediction is requested but the model failed to load."""

    status_code = 503
    error_code = "MODEL_NOT_LOADED"


class ModelNotFoundError(AppException):
    """Raised when a requested model identifier does not exist."""

    status_code = 404
    error_code = "MODEL_NOT_FOUND"


class PredictionError(AppException):
    """Raised when the underlying estimator fails to produce a prediction."""

    status_code = 422
    error_code = "PREDICTION_FAILED"


class InvalidFeatureError(AppException):
    """Raised when request features fail domain-level validation."""

    status_code = 422
    error_code = "INVALID_FEATURES"


class RateLimitExceededError(AppException):
    """Raised when a client exceeds the configured rate limit."""

    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"


class UnauthorizedError(AppException):
    """Raised when API-key authentication fails."""

    status_code = 401
    error_code = "UNAUTHORIZED"
