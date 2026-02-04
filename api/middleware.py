"""
Authentication middleware for the AI Voice Detection API.

This module implements API key-based authentication middleware that validates
the x-api-key header on all incoming requests.
"""

from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from core.config import get_settings
from utils.exceptions import AuthenticationError, AuthorizationError


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware for validating API keys in request headers.
    
    This middleware checks for the presence and validity of the x-api-key header
    on all incoming requests. Requests without valid API keys are rejected with
    appropriate HTTP status codes.
    
    Attributes:
        app: The FastAPI application instance
    """
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Process each request and validate the API key.
        
        This method is called for every incoming request. It validates the
        x-api-key header against the configured API keys and either allows
        the request to proceed or returns an error response.
        
        Public endpoints (health check, docs, root) bypass authentication.
        
        Args:
            request: The incoming HTTP request
            call_next: Callable to invoke the next middleware or endpoint
            
        Returns:
            Response: Either the endpoint response or an error response
            
        Raises:
            AuthenticationError: If the x-api-key header is missing (401)
            AuthorizationError: If the x-api-key header is invalid (403)
        """
        # Public endpoints that don't require authentication
        public_paths = ["/health", "/", "/docs", "/redoc", "/openapi.json"]
        
        # Allow public endpoints to bypass authentication
        if request.url.path in public_paths:
            response = await call_next(request)
            return response
        
        # Get the API key from the request header
        api_key = request.headers.get("x-api-key")
        
        # Check if API key is missing
        if api_key is None or api_key == "":
            return JSONResponse(
                status_code=401,
                content={
                    "error": "AuthenticationError",
                    "detail": "API key required",
                    "status_code": 401
                }
            )
        
        # Get valid API keys from settings
        settings = get_settings()
        valid_api_keys = settings.api_keys
        
        # Check if the provided API key is valid
        if api_key not in valid_api_keys:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "AuthorizationError",
                    "detail": "Invalid API key",
                    "status_code": 403
                }
            )
        
        # API key is valid, proceed with the request
        response = await call_next(request)
        return response
