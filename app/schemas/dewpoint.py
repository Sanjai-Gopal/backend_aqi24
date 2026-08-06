"""Request schema for the dew point prediction endpoint."""

from datetime import date as date_type

from pydantic import BaseModel, Field, model_validator

from app.utils.constants import MAX_YEAR, MIN_YEAR


class DewPointRequest(BaseModel):
    """
    Input features for the dew point model.

    Model expects, in order: [year, month, day, dayofweek].
    """

    year: int = Field(..., ge=MIN_YEAR, le=MAX_YEAR, description="Calendar year", examples=[2026])
    month: int = Field(..., ge=1, le=12, description="Month of year (1-12)", examples=[8])
    day: int = Field(..., ge=1, le=31, description="Day of month (1-31)", examples=[6])
    dayofweek: int = Field(
        ..., ge=0, le=6, description="Day of week, 0=Monday ... 6=Sunday", examples=[3]
    )

    model_config = {
        "json_schema_extra": {
            "example": {"year": 2026, "month": 8, "day": 6, "dayofweek": 3}
        }
    }

    @model_validator(mode="after")
    def validate_calendar_date(self) -> "DewPointRequest":
        try:
            date_type(self.year, self.month, self.day)
        except ValueError as exc:
            raise ValueError(f"Invalid calendar date: {exc}") from exc
        return self

    def to_feature_vector(self) -> list[float]:
        """Return features in the exact order the model was trained on."""
        return [self.year, self.month, self.day, self.dayofweek]
