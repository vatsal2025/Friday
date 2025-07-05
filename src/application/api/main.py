"""Main API application for the Friday AI Trading System.

This module initializes the FastAPI application, registers all the routers,
and provides the entry point for running the API server.
"""

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.infrastructure.config import get_config
from src.infrastructure.logging import get_logger
from src.application.api.api_gateway import create_api_gateway
from src.application.api.auth_router import router as auth_router
from src.application.api.broker_router import router as broker_router
from src.application.api.notification_router import router as notification_router
from src.application.api.event_router import router as event_router

# Create logger
logger = get_logger(__name__)

# Load API configuration
API_CONFIG = get_config("API_CONFIG")


def create_app() -> FastAPI:
    """Create the FastAPI application.

    Returns:
        FastAPI: The FastAPI application.
    """
    # Create API gateway
    app = create_api_gateway()

    # Register exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Global exception handler.

        Args:
            request: The request object.
            exc: The exception.

        Returns:
            JSONResponse: The error response.
        """
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected error occurred. Please try again later."},
        )

    # Register routers
    app.include_router(auth_router)
    app.include_router(broker_router)
    app.include_router(notification_router)
    app.include_router(event_router)

    # Add health check endpoint
    @app.get("/health", tags=["health"])
    async def health_check() -> dict:
        """Health check endpoint.

        Returns:
            dict: The health status.
        """
        return {"status": "ok"}

    # Add API version endpoint
    @app.get("/version", tags=["version"])
    async def version() -> dict:
        """API version endpoint.

        Returns:
            dict: The API version.
        """
        return {"version": "1.0.0"}

    return app


def run_api_server() -> None:
    """Run the API server."""
    app = create_app()
    host = API_CONFIG.get("host", "0.0.0.0")
    port = API_CONFIG.get("port", 8000)
    debug = API_CONFIG.get("debug", False)
    workers = API_CONFIG.get("workers", 1)

    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(
        "src.application.api.main:create_app",
        host=host,
        port=port,
        reload=debug,
        workers=workers,
        factory=True,
    )


if __name__ == "__main__":
    run_api_server()