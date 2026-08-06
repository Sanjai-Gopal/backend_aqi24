"""
Professional logging configuration.

Provides:
    * Console logging (human-readable, colorized in a TTY).
    * Rotating file handler for all logs (`logs/app.log`).
    * A dedicated rotating error-log file (`logs/error.log`) containing
      only WARNING+ records, so operators can tail failures in isolation.
    * Optional JSON formatting for log-aggregation pipelines
      (Datadog, ELK, CloudWatch, etc.) controlled by `settings.LOG_JSON`.
"""

import json
import logging
import logging.config
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id

        return json.dumps(payload)


def configure_logging() -> None:
    """Configure root and application loggers. Call once at startup."""

    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    app_log_path: Path = settings.LOG_DIR / settings.LOG_FILE
    error_log_path: Path = settings.LOG_DIR / settings.ERROR_LOG_FILE

    plain_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    formatter_key = "json" if settings.LOG_JSON else "plain"

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "plain": {
                "format": plain_format,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                "()": JSONFormatter,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.LOG_LEVEL,
                "formatter": formatter_key,
                "stream": sys.stdout,
            },
            "app_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": settings.LOG_LEVEL,
                "formatter": formatter_key,
                "filename": str(app_log_path),
                "maxBytes": settings.LOG_ROTATION_MAX_BYTES,
                "backupCount": settings.LOG_ROTATION_BACKUP_COUNT,
                "encoding": "utf-8",
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "WARNING",
                "formatter": formatter_key,
                "filename": str(error_log_path),
                "maxBytes": settings.LOG_ROTATION_MAX_BYTES,
                "backupCount": settings.LOG_ROTATION_BACKUP_COUNT,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "app": {
                "handlers": ["console", "app_file", "error_file"],
                "level": settings.LOG_LEVEL,
                "propagate": False,
            },
            "uvicorn": {
                "handlers": ["console", "app_file"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["console", "app_file", "error_file"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["console", "app_file"],
                "level": "INFO",
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["console", "app_file", "error_file"],
            "level": settings.LOG_LEVEL,
        },
    }

    logging.config.dictConfig(logging_config)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the `app` hierarchy."""
    return logging.getLogger(f"app.{name}")
