"""Tests for the fire ML insights endpoints."""


def test_ml_predictions_success(client):
    response = client.get("/api/v1/ml/predictions")
    assert response.status_code == 200
    body = response.json()
    assert "model_info" in body
    assert "regional_predictions" in body
    assert len(body["regional_predictions"]) == 15 * 4 * 4
    prediction = body["regional_predictions"][0]
    assert set(["region", "lat", "lon", "season", "brightness_level", "brightness",
                "frp_predicted", "severity", "severity_label", "severity_prob"]).issubset(prediction)
    assert len(prediction["severity_prob"]) == 4
    assert body["feature_columns"] == [
        "latitude", "longitude", "brightness", "scan", "track", "year", "month", "day"
    ]
    assert body["severity_classes"] == ["Low", "Medium", "High", "Extreme"]


def test_ml_model_meta_success(client):
    response = client.get("/api/v1/ml/model-meta")
    assert response.status_code == 200
    body = response.json()
    assert body["frp_regressor"]["algorithm"] == "LightGBM"
    assert body["severity_classifier"]["algorithm"] == "Rule-based thresholding of FRP regressor output"
    importance = body["frp_regressor"]["feature_importance"]
    assert abs(sum(importance.values()) - 1.0) < 0.01
    assert "coverage" in body
    assert "lat_range" in body["coverage"]
    assert "lon_range" in body["coverage"]
