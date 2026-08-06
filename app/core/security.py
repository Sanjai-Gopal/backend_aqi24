"""
Security utilities.

Includes an optional API-key dependency (disabled by default, enabled via
`API_KEY_ENABLED=true` in the environment) that routers can attach with
`Depends(verify_api_key)`.
"""

from fastapi import Header, HTTPException, status

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger("security")


async def verify_api_key(
    x_api_key: str = Header(default=None, alias="X-API-Key"),
) -> None:
    """
    Validate the `X-API-Key` header against the configured API key.

    No-op when `API_KEY_ENABLED` is False, which is the default for local
    development. Enable it in production via the environment.
    """
    if not settings.API_KEY_ENABLED:
        return

    if not x_api_key or x_api_key != settings.API_KEY:
        logger.warning("Rejected request with invalid or missing API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


# Security headers applied to every response by SecurityHeadersMiddleware.
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-XSS-Protection": "1; mode=block",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}
