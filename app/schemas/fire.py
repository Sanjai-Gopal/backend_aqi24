"""Request schema for the fire (archive/MODIS) prediction endpoint."""

from datetime import date as date_type

from pydantic import BaseModel, Field, model_validator

from app.utils.constants import (
    MAX_LATITUDE,
    MAX_LONGITUDE,
    MAX_YEAR,
    MIN_LATITUDE,
    MIN_LONGITUDE,
    MIN_YEAR,
)


class FireRequest(BaseModel):
    """
    Input features for the fire archive model.

    Model expects, in order:
    [latitude, longitude, brightness, scan, track, year, month, day].
    """

    latitude: float = Field(
        ..., ge=MIN_LATITUDE, le=MAX_LATITUDE, description="Latitude in decimal degrees", examples=[34.05]
    )
    longitude: float = Field(
        ..., ge=MIN_LONGITUDE, le=MAX_LONGITUDE, description="Longitude in decimal degrees", examples=[-118.24]
    )
    brightness: float = Field(
        ..., gt=0, description="Brightness temperature (Kelvin) of the fire pixel", examples=[320.5]
    )
    scan: float = Field(..., gt=0, description="Along-scan pixel size (km)", examples=[1.2])
    track: float = Field(..., gt=0, description="Along-track pixel size (km)", examples=[1.1])
    year: int = Field(..., ge=MIN_YEAR, le=MAX_YEAR, description="Calendar year", examples=[2026])
    month: int = Field(..., ge=1, le=12, description="Month of year (1-12)", examples=[8])
    day: int = Field(..., ge=1, le=31, description="Day of month (1-31)", examples=[6])

    model_config = {
        "json_schema_extra": {
            "example": {
                "latitude": 34.05,
                "longitude": -118.24,
                "brightness": 320.5,
                "scan": 1.2,
                "track": 1.1,
                "year": 2026,
                "month": 8,
                "day": 6,
            }
        }
    }

    @model_validator(mode="after")
    def validate_calendar_date(self) -> "FireRequest":
        try:
            date_type(self.year, self.month, self.day)
        except ValueError as exc:
            raise ValueError(f"Invalid calendar date: {exc}") from exc
        return self

    def to_feature_vector(self) -> list[float]:
        """Return features in the exact order the model was trained on."""
        return [
            self.latitude,
            self.longitude,
            self.brightness,
            self.scan,
            self.track,
            self.year,
            self.month,
            self.day,
        ]
