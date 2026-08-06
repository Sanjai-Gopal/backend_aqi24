"""Shared pytest fixtures for the test suite."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ml.model_registry import model_registry


@pytest.fixture(scope="session", autouse=True)
def _load_models_once():
    """Ensure real models are loaded once for the whole test session."""
    model_registry.load_all()
    app.state.model_registry = model_registry
    yield


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client
