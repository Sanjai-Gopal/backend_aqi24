"""Tests for the temperature prediction endpoint."""


VALID_PAYLOAD = {"year": 2026, "month": 8, "day": 6, "dayofweek": 3}


def test_predict_temperature_success(client):
    response = client.post("/api/v1/predict/temperature", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["model"] == "Temperature Model"
    assert isinstance(body["prediction"], float)


def test_predict_temperature_invalid_month(client):
    payload = {**VALID_PAYLOAD, "month": 13}
    response = client.post("/api/v1/predict/temperature", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "error"
    assert body["error_code"] == "VALIDATION_ERROR"


def test_predict_temperature_invalid_calendar_date(client):
    payload = {**VALID_PAYLOAD, "month": 2, "day": 30}
    response = client.post("/api/v1/predict/temperature", json=payload)
    assert response.status_code == 422


def test_predict_temperature_missing_field(client):
    payload = {"year": 2026, "month": 8, "day": 6}
    response = client.post("/api/v1/predict/temperature", json=payload)
    assert response.status_code == 422
