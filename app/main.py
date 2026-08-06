"""
Application entry point.

Builds the FastAPI application via a factory function (`create_app`), wires
up middleware, versioned routers, startup/shutdown lifecycle management for
ML models, and global exception handlers that translate every error into a
consistent JSON envelope.
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.api import api_router
from app.core.config import settings
from app.core.exceptions import AppException
from app.core.logging_config import configure_logging, get_logger
from app.middleware.logging_middleware import RequestLoggingMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.ml.model_registry import ModelRegistry
from app.schemas.common import RootResponse

configure_logging()
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML models on startup; release resources on shutdown."""
    logger.info(
        "startup name=%s version=%s environment=%s",
        settings.PROJECT_NAME,
        settings.VERSION,
        settings.ENVIRONMENT,
    )
    start = time.perf_counter()

    registry = ModelRegistry()
    try:
        registry.load_all()
    except RuntimeError:
        logger.critical("startup_aborted reason=model_load_failure")
        raise
    app.state.model_registry = registry

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("startup_complete duration_ms=%.2f", duration_ms)

    yield

    logger.info("shutdown name=%s", settings.PROJECT_NAME)


def create_app() -> FastAPI:
    """Application factory. Keeps import-time side effects to a minimum."""

    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Middleware (executed bottom-up relative to registration order)
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )

    if settings.ALLOWED_HOSTS != ["*"]:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    @app.get("/", response_model=RootResponse, tags=["Root"], summary="API root")
    def root() -> RootResponse:
        return RootResponse(
            status="success",
            message=f"{settings.PROJECT_NAME} running successfully",
            version=settings.VERSION,
            docs_url="/docs",
        )

    # Unversioned convenience alias for orchestrators/load balancers
    # (Kubernetes, ELB, Docker HEALTHCHECK) that expect a stable /health path.
    @app.get("/health", tags=["Health"], summary="Unversioned health check alias")
    def health_alias(request: Request):
        from app.api.v1.endpoints.health import health_check

        return health_check(request)

    register_exception_handlers(app)
    return app


def register_exception_handlers(app: FastAPI) -> None:
    """Attach global exception handlers producing a consistent error envelope."""

    def _error_body(request: Request, error_code: str, message: str, details: dict | None = None) -> dict:
        return {
            "status": "error",
            "error_code": error_code,
            "message": message,
            "details": details or {},
            "request_id": getattr(request.state, "request_id", None),
        }

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        logger.warning(
            "app_exception error_code=%s message=%s path=%s",
            exc.error_code,
            exc.message,
            request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(request, exc.error_code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = jsonable_encoder(exc.errors())
        logger.info("validation_error path=%s errors=%s", request.url.path, errors)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_body(
                request,
                "VALIDATION_ERROR",
                "Request validation failed.",
                {"errors": errors},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(request, "HTTP_ERROR", str(exc.detail)),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception path=%s", request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body(
                request,
                "INTERNAL_ERROR",
                "An unexpected error occurred. Please try again later.",
            ),
        )


app = create_app()
