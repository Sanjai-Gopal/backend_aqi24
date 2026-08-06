"""Tests for root and health endpoints."""


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "version" in body


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"healthy", "degraded"}
    assert isinstance(body["models"], list)
    assert len(body["models"]) == 4


def test_model_status_endpoint(client):
    response = client.get("/api/v1/models/status")
    assert response.status_code == 200
    body = response.json()
    model_names = {m["name"] for m in body["models"]}
    assert model_names == {
        "temperature_model",
        "dewpoint_model",
        "fire_prediction_model",
        "fire_nrt_prediction_model",
    }
