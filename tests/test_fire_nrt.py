"""Tests for the fire NRT prediction endpoint."""


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


def test_predict_fire_nrt_success(client):
    response = client.post("/api/v1/predict/fire-nrt", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["model"] == "Fire NRT Model"
    assert isinstance(body["prediction"], float)


def test_predict_fire_nrt_invalid_longitude(client):
    payload = {**VALID_PAYLOAD, "longitude": 200.0}
    response = client.post("/api/v1/predict/fire-nrt", json=payload)
    assert response.status_code == 422
