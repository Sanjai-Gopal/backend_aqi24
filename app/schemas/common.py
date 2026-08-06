"""Common response schemas shared across all endpoints."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Standard envelope returned by every prediction endpoint."""

    status: str = Field(default="success", examples=["success"])
    model: str = Field(..., description="Human-readable model name")
    prediction: float = Field(..., description="The model's predicted value")
    unit: Optional[str] = Field(default=None, description="Unit of the predicted value")
    request_id: Optional[str] = Field(default=None, description="Correlation ID for this request")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "model": "Temperature Model",
                "prediction": 27.43,
                "unit": "celsius",
                "request_id": "8f14e45f-ceea-4c9c-8f3c-0d1f7a5f2e1a",
                "timestamp": "2026-08-06T10:00:00Z",
            }
        }
    }


class ErrorResponse(BaseModel):
    """Standard error envelope returned by the global exception handlers."""

    status: str = Field(default="error")
    error_code: str = Field(..., description="Machine-readable error identifier")
    message: str = Field(..., description="Human-readable error message")
    details: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ModelHealth(BaseModel):
    """Health/status of a single loaded model."""

    name: str
    loaded: bool
    file: str
    expected_features: List[str]


class HealthResponse(BaseModel):
    """Response schema for the liveness/readiness health check endpoint."""

    status: str
    api: str
    version: str
    environment: str
    models: List[ModelHealth]


class RootResponse(BaseModel):
    """Response schema for the API root endpoint."""

    status: str
    message: str
    version: str
    docs_url: str
