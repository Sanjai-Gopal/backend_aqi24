"""
Centralized application configuration.

All runtime configuration is sourced from environment variables (or a `.env`
file in local development) via `pydantic-settings`. Nothing in this module
should be hard-coded per-environment -- use `.env` / real environment
variables to configure dev, staging, and production deployments.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root (fire-ai-backend/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application-wide settings, loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------
    PROJECT_NAME: str = "Wildfire Prediction API"
    PROJECT_DESCRIPTION: str = (
        "Enterprise-grade Machine Learning API for temperature, dew point, "
        "and wildfire occurrence prediction."
    )
    VERSION: str = "2.0.0"
    ENVIRONMENT: Literal["development", "staging", "production", "testing"] = "development"
    DEBUG: bool = False

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    API_V1_PREFIX: str = "/api/v1"

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = Field(default_factory=lambda: ["*"])
    CORS_ALLOW_HEADERS: List[str] = Field(default_factory=lambda: ["*"])

    @field_validator("CORS_ORIGINS", "CORS_ALLOW_METHODS", "CORS_ALLOW_HEADERS", mode="before")
    @classmethod
    def _split_csv(cls, value):
        """Allow comma-separated strings in .env, e.g. CORS_ORIGINS=a.com,b.com"""
        if isinstance(value, str):
            if value.strip() == "*":
                return ["*"]
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    # ------------------------------------------------------------------
    # Trusted hosts
    # ------------------------------------------------------------------
    ALLOWED_HOSTS: List[str] = Field(default_factory=lambda: ["*"])

    @field_validator("ALLOWED_HOSTS", mode="before")
    @classmethod
    def _split_hosts(cls, value):
        if isinstance(value, str):
            if value.strip() == "*":
                return ["*"]
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # ------------------------------------------------------------------
    # Machine learning models
    # ------------------------------------------------------------------
    ML_MODELS_DIR: Path = BASE_DIR / "ml_models"
    TEMPERATURE_MODEL_FILENAME: str = "temperature_model.pkl"
    DEWPOINT_MODEL_FILENAME: str = "dewpoint_model.pkl"
    FIRE_MODEL_FILENAME: str = "fire_prediction_model.pkl"
    FIRE_NRT_MODEL_FILENAME: str = "fire_nrt_prediction_model.pkl"

    # Fail startup if any model fails to load (recommended for production).
    MODEL_LOAD_STRICT: bool = True

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path = BASE_DIR / "logs"
    LOG_FILE: str = "app.log"
    ERROR_LOG_FILE: str = "error.log"
    LOG_ROTATION_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
    LOG_ROTATION_BACKUP_COUNT: int = 5
    LOG_JSON: bool = False

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------
    API_KEY_ENABLED: bool = False
    API_KEY: str = Field(default="", repr=False)
    API_KEY_HEADER_NAME: str = "X-API-Key"

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Using `lru_cache` means the environment is only parsed once per process,
    while still allowing dependency-injection / overrides in tests via
    `app.dependency_overrides`.
    """
    return Settings()


settings = get_settings()
