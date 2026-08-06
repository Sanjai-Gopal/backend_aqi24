"""Static domain reference data for fire ML insight endpoints."""

from typing import Dict, List, TypedDict


class FireRegion(TypedDict):
    name: str
    lat: float
    lon: float


FIRE_REGIONS: List[FireRegion] = [
    {"name": "Punjab Crop Belt", "lat": 31.0, "lon": 75.5},
    {"name": "Haryana Crop Belt", "lat": 29.5, "lon": 76.5},
    {"name": "Uttar Pradesh", "lat": 27.0, "lon": 81.0},
    {"name": "Madhya Pradesh Forests", "lat": 22.5, "lon": 78.5},
    {"name": "Odisha Forests", "lat": 20.5, "lon": 84.5},
    {"name": "Assam / NE India", "lat": 26.0, "lon": 93.0},
    {"name": "West Bengal", "lat": 23.5, "lon": 87.5},
    {"name": "Rajasthan", "lat": 26.0, "lon": 73.5},
    {"name": "Andhra Pradesh", "lat": 16.0, "lon": 80.0},
    {"name": "Karnataka", "lat": 14.0, "lon": 76.0},
    {"name": "Chhattisgarh", "lat": 21.0, "lon": 82.5},
    {"name": "Bihar", "lat": 25.5, "lon": 85.5},
    {"name": "Delhi NCR", "lat": 28.7, "lon": 77.2},
    {"name": "Uttarakhand Forests", "lat": 30.0, "lon": 79.5},
    {"name": "Kerala Forests", "lat": 10.5, "lon": 76.5},
]

SEASONS: List[str] = ["winter", "summer", "post_monsoon", "monsoon"]

SEASON_MONTH: Dict[str, int] = {
    "winter": 1,
    "summer": 5,
    "post_monsoon": 11,
    "monsoon": 8,
}

SEASON_DAY: int = 15

BRIGHTNESS_LEVELS: Dict[str, float] = {
    "low": 320.0,
    "medium": 340.0,
    "high": 355.0,
    "extreme": 365.0,
}

SEVERITY_CLASSES: List[str] = ["Low", "Medium", "High", "Extreme"]

SEVERITY_THRESHOLDS_MW: List[float] = [10.0, 50.0, 200.0]

DEFAULT_SCAN: float = 1.0
DEFAULT_TRACK: float = 1.0
