"""Tests for the fire archive prediction endpoint."""


VALID_PAYLOAD = {
    "latitude": 34.05,
    "longitude": -118.24,
    "brightness": 320.5,
    "scan": 1.2,
    "track": 1.1,
    "year": 2026,
    "month": 8,
    "day": 6,
}


def test_predict_fire_success(client):
    response = client.post("/api/v1/predict/fire", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["model"] == "Fire Archive Model"
    assert isinstance(body["prediction"], float)


def test_predict_fire_invalid_latitude(client):
    payload = {**VALID_PAYLOAD, "latitude": 200.0}
    response = client.post("/api/v1/predict/fire", json=payload)
    assert response.status_code == 422


def test_predict_fire_negative_brightness(client):
    payload = {**VALID_PAYLOAD, "brightness": -5.0}
    response = client.post("/api/v1/predict/fire", json=payload)
    assert response.status_code == 422
