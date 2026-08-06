"""Response schemas for fire ML insight endpoints."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RegressorInfo(BaseModel):
    algorithm: str
    target: str
    rmse_mw: Optional[float] = None
    mae_mw: Optional[float] = None
    r2: Optional[float] = None
    r2_log: Optional[float] = None
    feature_importance: Dict[str, float] = Field(default_factory=dict)


class ClassifierInfo(BaseModel):
    algorithm: str
    target: str
    accuracy: Optional[float] = None
    classes: List[str]
    feature_importance: Dict[str, float] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    trained_at: str
    data_source: str
    satellite: str
    n_train: Optional[int] = None
    n_test: Optional[int] = None
    total_records_in_dataset: Optional[int] = None
    date_range: Optional[List[str]] = None
    frp_regressor: RegressorInfo
    severity_classifier: ClassifierInfo


class Coverage(BaseModel):
    lat_range: List[float]
    lon_range: List[float]
    n_years: Optional[int] = None


class ModelMetaResponse(ModelInfo):
    coverage: Coverage


class RegionalPrediction(BaseModel):
    region: str
    lat: float
    lon: float
    season: str
    brightness_level: str
    brightness: float
    frp_predicted: float
    severity: int
    severity_label: str
    severity_prob: List[float]


class MLPredictionsResponse(BaseModel):
    model_info: ModelInfo
    regional_predictions: List[RegionalPrediction]
    feature_columns: List[str]
    severity_classes: List[str]
