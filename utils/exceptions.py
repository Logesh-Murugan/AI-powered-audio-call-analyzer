"""
Custom exception classes for the AI Voice Detection API.

This module defines a hierarchy of exceptions used throughout the application
to handle various error conditions with appropriate HTTP status codes.
"""


class APIException(Exception):
    """
    Base exception class for all API errors.
    
    Attributes:
        status_code: HTTP status code to return
        detail: Human-readable error description
    """
    
    def __init__(self, detail: str, status_code: int = 500):
        """
        Initialize the API exception.
        
        Args:
            detail: Human-readable error description
            status_code: HTTP status code (default: 500)
        """
        self.detail = detail
        self.status_code = status_code
        super().__init__(self.detail)


class AuthenticationError(APIException):
    """
    Exception raised when authentication fails (missing API key).
    
    Returns HTTP 401 Unauthorized.
    """
    
    def __init__(self, detail: str = "API key required"):
        """
        Initialize authentication error.
        
        Args:
            detail: Error description (default: "API key required")
        """
        super().__init__(detail=detail, status_code=401)


class AuthorizationError(APIException):
    """
    Exception raised when authorization fails (invalid API key).
    
    Returns HTTP 403 Forbidden.
    """
    
    def __init__(self, detail: str = "Invalid API key"):
        """
        Initialize authorization error.
        
        Args:
            detail: Error description (default: "Invalid API key")
        """
        super().__init__(detail=detail, status_code=403)


class AudioDecodingError(APIException):
    """
    Exception raised when audio decoding fails.
    
    Returns HTTP 400 Bad Request.
    """
    
    def __init__(self, detail: str = "Failed to decode audio data"):
        """
        Initialize audio decoding error.
        
        Args:
            detail: Error description (default: "Failed to decode audio data")
        """
        super().__init__(detail=detail, status_code=400)


class FeatureExtractionError(APIException):
    """
    Exception raised when feature extraction fails.
    
    Returns HTTP 400 Bad Request.
    """
    
    def __init__(self, detail: str = "Failed to extract audio features"):
        """
        Initialize feature extraction error.
        
        Args:
            detail: Error description (default: "Failed to extract audio features")
        """
        super().__init__(detail=detail, status_code=400)


class ModelInferenceError(APIException):
    """
    Exception raised when model inference fails.
    
    Returns HTTP 500 Internal Server Error.
    """
    
    def __init__(self, detail: str = "Prediction failed, please try again"):
        """
        Initialize model inference error.
        
        Args:
            detail: Error description (default: "Prediction failed, please try again")
        """
        super().__init__(detail=detail, status_code=500)
