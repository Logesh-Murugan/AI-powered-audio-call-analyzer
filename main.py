"""
AI-Generated Voice Detection API

This is the main application entry point for the AI Voice Detection API.
It initializes the FastAPI application, registers middleware, routes,
and exception handlers.
"""

from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.routes import router as detection_router
from api.middleware import APIKeyMiddleware
from utils.exceptions import APIException
from utils.logger import get_logger
from core.config import get_settings


# Initialize logger
logger = get_logger("main")

# Initialize settings
settings = get_settings()

# Create FastAPI application
app = FastAPI(
    title="AI Voice Detection API",
    version="1.0.0",
    description="Detect AI-generated voices in audio files using machine learning",
    docs_url="/docs",
    redoc_url="/redoc"
)


# Register middleware
app.add_middleware(APIKeyMiddleware)


# Global exception handler for custom exceptions
@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """
    Global exception handler for custom API exceptions.
    
    This handler catches all APIException instances and converts them
    to properly formatted JSON error responses without exposing
    internal implementation details or stack traces.
    
    Args:
        request: The incoming request that caused the exception
        exc: The APIException instance
        
    Returns:
        JSONResponse: Formatted error response
    """
    logger.error(
        f"API exception occurred: {exc.__class__.__name__} - {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "detail": exc.detail,
            "status_code": exc.status_code
        }
    )


# Global exception handler for unexpected exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for unexpected exceptions.
    
    This handler catches all unhandled exceptions and returns a generic
    error response without exposing internal details or stack traces.
    The full exception is logged internally for debugging.
    
    Args:
        request: The incoming request that caused the exception
        exc: The exception instance
        
    Returns:
        JSONResponse: Generic error response
    """
    logger.error(
        f"Unexpected exception occurred: {exc.__class__.__name__} - {str(exc)}",
        exc_info=True,
        extra={
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "Internal server error",
            "status_code": 500
        }
    )


# Include routers
app.include_router(
    detection_router,
    prefix="/api/v1",
    tags=["detection"]
)


# Health check endpoint
@app.get("/health", tags=["monitoring"])
async def health_check() -> dict:
    """
    Health check endpoint for monitoring and deployment verification.
    
    This endpoint is used by deployment platforms (like Render.com) to verify
    that the application is running and healthy. It checks basic application
    status and returns a timestamp.
    
    Returns:
        dict: Health status information including:
            - status: "healthy" if application is running
            - timestamp: Current UTC timestamp in ISO format
            - version: API version
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


# Root endpoint
@app.get("/", tags=["info"])
async def root() -> dict:
    """
    Root endpoint providing API information.
    
    Returns:
        dict: Basic API information and links to documentation
    """
    return {
        "name": "AI Voice Detection API",
        "version": "1.0.0",
        "description": "Detect AI-generated voices in audio files",
        "docs": "/docs",
        "health": "/health"
    }


# Application startup event
@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler.
    
    This function is called when the application starts up.
    It logs startup information and configuration details.
    """
    logger.info("=" * 60)
    logger.info("AI Voice Detection API Starting Up")
    logger.info("=" * 60)
    logger.info(f"Version: 1.0.0")
    logger.info(f"Model: {settings.model_name}")
    logger.info(f"Model Cache: {settings.model_cache_dir}")
    logger.info(f"Log Level: {settings.log_level}")
    logger.info(f"API Keys Configured: {len(settings.api_keys)}")
    logger.info("=" * 60)


# Application shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown event handler.
    
    This function is called when the application shuts down.
    It logs shutdown information and performs cleanup if needed.
    """
    logger.info("=" * 60)
    logger.info("AI Voice Detection API Shutting Down")
    logger.info("=" * 60)


if __name__ == "__main__":
    """
    Main entry point for running the application directly.
    
    This is used for local development. In production, use uvicorn directly:
    uvicorn main:app --host 0.0.0.0 --port 8000
    """
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=True  # Enable auto-reload for development
    )
