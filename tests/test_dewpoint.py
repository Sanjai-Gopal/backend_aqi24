"""Tests for the dew point prediction endpoint."""


VALID_PAYLOAD = {"year": 2026, "month": 8, "day": 6, "dayofweek": 3}


def test_predict_dewpoint_success(client):
    response = client.post("/api/v1/predict/dewpoint", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["model"] == "Dew Point Model"
    assert isinstance(body["prediction"], float)


def test_predict_dewpoint_invalid_year(client):
    payload = {**VALID_PAYLOAD, "year": 1500}
    response = client.post("/api/v1/predict/dewpoint", json=payload)
    assert response.status_code == 422
