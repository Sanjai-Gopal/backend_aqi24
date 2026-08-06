"""
Lightweight, dependency-free rate limiting middleware.

Uses a fixed-window counter per client IP, held in memory. This is
appropriate for a single-process deployment or as a first line of defense
in front of an API gateway. For multi-instance production deployments,
back this with Redis (e.g. `slowapi` + Redis storage) instead.
"""

import time
from collections import defaultdict
from threading import Lock

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger("rate_limit")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window request-count limiter, keyed by client IP."""

    def __init__(self, app):
        super().__init__(app)
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)

        # Health checks are exempt so orchestrators (k8s, ELB) never get throttled.
        if request.url.path in {"/health", "/api/v1/health", "/"}:
            return await call_next(request)

        client_key = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = settings.RATE_LIMIT_WINDOW_SECONDS
        limit = settings.RATE_LIMIT_REQUESTS

        with self._lock:
            hits = self._hits[client_key]
            # Drop timestamps outside the current window.
            cutoff = now - window
            while hits and hits[0] < cutoff:
                hits.pop(0)

            if len(hits) >= limit:
                retry_after = max(0, int(window - (now - hits[0])))
                logger.warning("rate_limit_exceeded client=%s path=%s", client_key, request.url.path)
                return JSONResponse(
                    status_code=429,
                    content={
                        "status": "error",
                        "error_code": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests. Please slow down.",
                        "retry_after_seconds": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )

            hits.append(now)

        return await call_next(request)
