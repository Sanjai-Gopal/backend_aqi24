# Wildfire Prediction API

Enterprise-grade FastAPI backend serving four Machine Learning models for
temperature, dew point, and wildfire-related prediction — built on Scikit-learn,
LightGBM, and Joblib, refactored into a clean, modular, production-ready
architecture.

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Models](#models)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the API](#running-the-api)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Deployment](#deployment)
- [Logging](#logging)
- [Security](#security)

## Architecture

The application follows a layered, clean-architecture style:

```
HTTP request
   → Middleware (logging, rate limiting, security headers, CORS)
   → API Router (app/api/v1) — HTTP concerns, request/response schemas
   → Service (app/services) — business logic, framework-agnostic
   → ML Registry (app/ml) — cached model loading & inference
   → Response (validated Pydantic schema, consistent envelope)
```

Cross-cutting concerns (configuration, logging, exceptions, security) live in
`app/core` and are shared by every layer above.

## Project Structure

```
fire-ai-backend/
├── app/
│   ├── main.py                  # App factory, lifespan, exception handlers
│   ├── core/
│   │   ├── config.py             # Environment-driven settings (pydantic-settings)
│   │   ├── logging_config.py     # Rotating file + console + error logs
│   │   ├── exceptions.py         # Custom exception hierarchy
│   │   └── security.py           # API-key dependency, security headers
│   ├── middleware/
│   │   ├── logging_middleware.py # Request ID + timing + structured logs
│   │   ├── rate_limit.py         # In-memory sliding-window rate limiter
│   │   └── security_headers.py   # Hardening headers on every response
│   ├── api/v1/
│   │   ├── api.py                # Router aggregator
│   │   ├── deps.py                # Dependency injection
│   │   └── endpoints/
│   │       ├── health.py
│   │       ├── temperature.py
│   │       ├── dewpoint.py
│   │       ├── fire.py
│   │       └── fire_nrt.py
│   ├── schemas/                  # Pydantic request/response models
│   ├── services/
│   │   └── prediction_service.py # Framework-agnostic business logic
│   ├── ml/
│   │   ├── base.py               # LoadedModel: caching, validation, timing
│   │   └── model_registry.py     # Singleton registry of all models
│   └── utils/
│       └── constants.py          # Feature orderings, enums, bounds
├── ml_models/                    # Your trained .pkl models (unchanged)
├── tests/                        # Pytest suite (health + all 4 predictors)
├── logs/                         # Rotating app.log / error.log
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── pytest.ini
```

## Models

| Model file                        | Endpoint                  | Features (in order)                                                        |
|------------------------------------|----------------------------|------------------------------------------------------------------------------|
| `temperature_model.pkl`            | `POST /api/v1/predict/temperature` | `year, month, day, dayofweek`                                       |
| `dewpoint_model.pkl`                | `POST /api/v1/predict/dewpoint`    | `year, month, day, dayofweek`                                       |
| `fire_prediction_model.pkl`         | `POST /api/v1/predict/fire`        | `latitude, longitude, brightness, scan, track, year, month, day`    |
| `fire_nrt_prediction_model.pkl`     | `POST /api/v1/predict/fire-nrt`    | `latitude, longitude, brightness, scan, track, year, month, day`    |

Feature order and count were **verified directly against the trained
estimators** (`n_features_in_` / `booster_.feature_name()`), not assumed —
this preserves exact compatibility with your original models. Note that
`dewpoint_model.pkl` was present in your original `ml_models/` folder but had
no endpoint in the prototype; it's now fully wired up as `/predict/dewpoint`.

Models are loaded **once** at application startup via `ModelRegistry` and
cached in memory for the lifetime of the process — no per-request disk I/O.

## Installation

### Prerequisites

- Python 3.11+ (tested on 3.12)
- pip

### Local setup

```bash
git clone <your-repo-url>
cd fire-ai-backend

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env              # adjust values as needed
```

## Configuration

All configuration is environment-driven via `app/core/config.py`
(`pydantic-settings`). Copy `.env.example` to `.env` and adjust. Key
variables:

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `development` | `development` / `staging` / `production` / `testing` |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `ALLOWED_HOSTS` | `*` | Comma-separated trusted hosts |
| `RATE_LIMIT_ENABLED` | `true` | Toggle the rate limiter |
| `RATE_LIMIT_REQUESTS` | `100` | Max requests per window per client |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window size |
| `MODEL_LOAD_STRICT` | `true` | Fail startup if any model fails to load |
| `LOG_LEVEL` | `INFO` | Root log level |
| `LOG_JSON` | `false` | Emit structured JSON logs (for log aggregators) |
| `API_KEY_ENABLED` | `false` | Require `X-API-Key` header on requests |

## Running the API

### Development

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production (multi-worker)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once running, interactive documentation is available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI schema: `http://localhost:8000/openapi.json`

### Example request

```bash
curl -X POST http://localhost:8000/api/v1/predict/temperature \
  -H "Content-Type: application/json" \
  -d '{"year": 2026, "month": 8, "day": 6, "dayofweek": 3}'
```

```json
{
  "status": "success",
  "model": "Temperature Model",
  "prediction": 27.43,
  "unit": "celsius",
  "request_id": "8f14e45f-ceea-4c9c-8f3c-0d1f7a5f2e1a",
  "timestamp": "2026-08-06T10:00:00Z"
}
```

### Error responses

Every error (validation, model failure, unexpected exception) is returned in
a single consistent envelope:

```json
{
  "status": "error",
  "error_code": "VALIDATION_ERROR",
  "message": "Request validation failed.",
  "details": { "errors": [ ... ] },
  "request_id": "8f14e45f-ceea-4c9c-8f3c-0d1f7a5f2e1a",
  "timestamp": "2026-08-06T10:00:00Z"
}
```

## Testing

```bash
pytest                       # run the full suite
pytest --cov=app             # with coverage
```

Tests cover: health/model-status, successful predictions for all four models,
and validation-failure paths (out-of-range values, invalid calendar dates,
missing fields).

## Deployment

### Docker

```bash
docker build -t wildfire-prediction-api .
docker run -p 8000:8000 --env-file .env wildfire-prediction-api
```

### Docker Compose

```bash
docker compose up --build -d
docker compose logs -f
docker compose down
```

The image is a multi-stage build (builder + slim runtime), runs as a
non-root user, and ships with a `HEALTHCHECK` against `/health`.

### Kubernetes / cloud notes

- Use `/health` as both liveness and readiness probe.
- Mount `.env` values as a `ConfigMap`/`Secret` rather than baking them into
  the image.
- For multi-instance deployments, replace the in-memory rate limiter
  (`app/middleware/rate_limit.py`) with a Redis-backed one (e.g. `slowapi`)
  so limits are shared across pods.
- Models are loaded per-process; scale horizontally with multiple replicas
  rather than relying on a single multi-worker container for very high
  throughput.

## Logging

- Console: human-readable (or JSON if `LOG_JSON=true`).
- `logs/app.log`: all application logs, rotated at 10 MB (5 backups).
- `logs/error.log`: WARNING+ only, for fast triage of failures.
- Every request is tagged with a `request_id` (propagated via the
  `X-Request-ID` header) for end-to-end correlation.

## Security

- CORS, trusted-host, and security-header middleware enabled by default.
- Optional API-key authentication (`API_KEY_ENABLED=true`), checked via the
  `X-API-Key` header.
- In-memory rate limiting (fixed window, per client IP).
- Strict Pydantic validation on every request, including real calendar-date
  and physical-bound checks (latitude/longitude ranges, positive brightness,
  etc.).
- No secrets are hard-coded; everything is environment-driven via `.env`
  (excluded from version control).
"# backend_aqi24" 
