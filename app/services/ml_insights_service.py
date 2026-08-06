"""Fire ML insights service: live regional predictions and model introspection."""

import math
from datetime import datetime, timezone
from typing import Dict, List

from app.core.logging_config import get_logger
from app.ml.model_registry import ModelRegistry
from app.utils.constants import FIRE_FEATURES, ModelName
from app.utils.fire_regions import (
    BRIGHTNESS_LEVELS,
    DEFAULT_SCAN,
    DEFAULT_TRACK,
    FIRE_REGIONS,
    SEASON_DAY,
    SEASON_MONTH,
    SEASONS,
    SEVERITY_CLASSES,
    SEVERITY_THRESHOLDS_MW,
)

logger = get_logger("ml_insights")

_SEVERITY_PROB_CENTERS = [2.0, 25.0, 100.0, 250.0]
_SEVERITY_PROB_SCALE = 40.0


def _severity_from_frp(frp: float) -> int:
    if frp < SEVERITY_THRESHOLDS_MW[0]:
        return 0
    if frp < SEVERITY_THRESHOLDS_MW[1]:
        return 1
    if frp < SEVERITY_THRESHOLDS_MW[2]:
        return 2
    return 3


def _severity_probabilities(frp: float) -> List[float]:
    scores = [-abs(frp - center) / _SEVERITY_PROB_SCALE for center in _SEVERITY_PROB_CENTERS]
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps)
    return [round(value / total, 3) for value in exps]


def _normalized_feature_importance(raw: Dict[str, float]) -> Dict[str, float]:
    total = sum(raw.values()) or 1.0
    return {name: round(value / total, 5) for name, value in raw.items()}


class MLInsightsService:
    """Computes regional fire predictions live from the deployed fire model."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def _fire_feature_importance(self) -> Dict[str, float]:
        model = self._registry.get(ModelName.FIRE.value)
        estimator = model.estimator
        if estimator is None:
            return {}
        try:
            booster = estimator.booster_
            names = booster.feature_name()
            gains = booster.feature_importance(importance_type="gain")
            raw = {name: float(gain) for name, gain in zip(names, gains)}
            return _normalized_feature_importance(raw)
        except Exception:
            logger.exception("feature_importance_failed model=%s", ModelName.FIRE.value)
            return {}

    def build_model_info(self) -> dict:
        importance = self._fire_feature_importance()
        return {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "data_source": "Live inference via locally deployed fire prediction model",
            "satellite": "N/A - local model deployment",
            "n_train": None,
            "n_test": None,
            "total_records_in_dataset": None,
            "date_range": None,
            "frp_regressor": {
                "algorithm": "LightGBM",
                "target": "Fire radiative power proxy (regression output, MW-equivalent)",
                "rmse_mw": None,
                "mae_mw": None,
                "r2": None,
                "r2_log": None,
                "feature_importance": importance,
            },
            "severity_classifier": {
                "algorithm": "Rule-based thresholding of FRP regressor output",
                "target": (
                    "Fire severity (0=Low<10MW, 1=Medium 10-50MW, "
                    "2=High 50-200MW, 3=Extreme>200MW)"
                ),
                "accuracy": None,
                "classes": SEVERITY_CLASSES,
                "feature_importance": importance,
            },
        }

    def build_coverage(self) -> dict:
        lats = [region["lat"] for region in FIRE_REGIONS]
        lons = [region["lon"] for region in FIRE_REGIONS]
        return {
            "lat_range": [min(lats), max(lats)],
            "lon_range": [min(lons), max(lons)],
            "n_years": None,
        }

    def build_model_meta(self) -> dict:
        info = self.build_model_info()
        info["coverage"] = self.build_coverage()
        return info

    def build_predictions(self) -> dict:
        fire_model = self._registry.get(ModelName.FIRE.value)
        model_info = self.build_model_info()
        year = datetime.now(timezone.utc).year

        regional_predictions = []
        for region in FIRE_REGIONS:
            for season in SEASONS:
                month = SEASON_MONTH[season]
                for level, brightness in BRIGHTNESS_LEVELS.items():
                    features = [
                        region["lat"],
                        region["lon"],
                        brightness,
                        DEFAULT_SCAN,
                        DEFAULT_TRACK,
                        year,
                        month,
                        SEASON_DAY,
                    ]
                    frp_predicted = fire_model.predict(features)
                    severity = _severity_from_frp(frp_predicted)
                    regional_predictions.append(
                        {
                            "region": region["name"],
                            "lat": region["lat"],
                            "lon": region["lon"],
                            "season": season,
                            "brightness_level": level,
                            "brightness": brightness,
                            "frp_predicted": round(frp_predicted, 2),
                            "severity": severity,
                            "severity_label": SEVERITY_CLASSES[severity],
                            "severity_prob": _severity_probabilities(frp_predicted),
                        }
                    )

        return {
            "model_info": model_info,
            "regional_predictions": regional_predictions,
            "feature_columns": FIRE_FEATURES,
            "severity_classes": SEVERITY_CLASSES,
        }
