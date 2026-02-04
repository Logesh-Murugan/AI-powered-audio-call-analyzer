"""
Unit tests for authentication middleware.

These tests verify that the APIKeyMiddleware correctly validates API keys
and returns appropriate HTTP status codes for different authentication scenarios.
"""

import os
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.middleware import APIKeyMiddleware
import core.config as config_module


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset settings singleton before and after each test."""
    config_module._settings = None
    yield
    config_module._settings = None
    # Clean up environment variables
    if "API_KEYS" in os.environ:
        del os.environ["API_KEYS"]


@pytest.fixture
def test_app():
    """Create a test FastAPI application with authentication middleware."""
    app = FastAPI()
    app.add_middleware(APIKeyMiddleware)
    
    @app.get("/test-endpoint")
    async def test_endpoint():
        return {"message": "success", "authenticated": True}
    
    return app


@pytest.fixture
def valid_api_key():
    """Provide a valid API key for testing."""
    return "test-valid-key-12345"


@pytest.fixture
def setup_valid_key(valid_api_key):
    """Set up environment with a valid API key."""
    os.environ["API_KEYS"] = valid_api_key
    config_module._settings = None
    yield valid_api_key
    if "API_KEYS" in os.environ:
        del os.environ["API_KEYS"]
    config_module._settings = None


def test_missing_api_key_returns_401(test_app, setup_valid_key):
    """
    Test that requests without an API key header return 401 status.
    
    Requirements: 2.2
    """
    client = TestClient(test_app)
    
    # Make request without x-api-key header
    response = client.get("/test-endpoint")
    
    # Verify 401 status code
    assert response.status_code == 401, \
        f"Expected 401 for missing API key, got {response.status_code}"
    
    # Verify error response structure
    response_data = response.json()
    assert "error" in response_data, "Response should contain 'error' field"
    assert "detail" in response_data, "Response should contain 'detail' field"
    assert "status_code" in response_data, "Response should contain 'status_code' field"
    
    # Verify error details
    assert response_data["error"] == "AuthenticationError", \
        "Error type should be AuthenticationError"
    assert "API key required" in response_data["detail"], \
        "Error detail should mention API key required"
    assert response_data["status_code"] == 401, \
        "Status code in response body should be 401"


def test_empty_api_key_returns_401(test_app, setup_valid_key):
    """
    Test that requests with an empty API key header return 401 status.
    
    Requirements: 2.2
    """
    client = TestClient(test_app)
    
    # Make request with empty x-api-key header
    response = client.get("/test-endpoint", headers={"x-api-key": ""})
    
    # Verify 401 status code
    assert response.status_code == 401, \
        f"Expected 401 for empty API key, got {response.status_code}"
    
    # Verify error response structure
    response_data = response.json()
    assert response_data["error"] == "AuthenticationError", \
        "Error type should be AuthenticationError"
    assert "API key required" in response_data["detail"], \
        "Error detail should mention API key required"


def test_invalid_api_key_returns_403(test_app, setup_valid_key):
    """
    Test that requests with an invalid API key return 403 status.
    
    Requirements: 2.3
    """
    client = TestClient(test_app)
    
    # Make request with invalid x-api-key header
    invalid_key = "invalid-key-xyz"
    response = client.get("/test-endpoint", headers={"x-api-key": invalid_key})
    
    # Verify 403 status code
    assert response.status_code == 403, \
        f"Expected 403 for invalid API key, got {response.status_code}"
    
    # Verify error response structure
    response_data = response.json()
    assert "error" in response_data, "Response should contain 'error' field"
    assert "detail" in response_data, "Response should contain 'detail' field"
    assert "status_code" in response_data, "Response should contain 'status_code' field"
    
    # Verify error details
    assert response_data["error"] == "AuthorizationError", \
        "Error type should be AuthorizationError"
    assert "Invalid API key" in response_data["detail"], \
        "Error detail should mention invalid API key"
    assert response_data["status_code"] == 403, \
        "Status code in response body should be 403"


def test_valid_api_key_allows_access(test_app, setup_valid_key, valid_api_key):
    """
    Test that requests with a valid API key are allowed through.
    
    Requirements: 2.2, 2.3
    """
    client = TestClient(test_app)
    
    # Make request with valid x-api-key header
    response = client.get("/test-endpoint", headers={"x-api-key": valid_api_key})
    
    # Verify 200 status code
    assert response.status_code == 200, \
        f"Expected 200 for valid API key, got {response.status_code}"
    
    # Verify successful response
    response_data = response.json()
    assert "message" in response_data, "Response should contain 'message' field"
    assert response_data["message"] == "success", \
        "Response message should be 'success'"
    assert response_data.get("authenticated") is True, \
        "Response should indicate successful authentication"
    
    # Verify no error fields in response
    assert "error" not in response_data, \
        "Successful response should not contain 'error' field"


def test_multiple_valid_api_keys(test_app):
    """
    Test that multiple valid API keys can be configured and all work.
    
    Requirements: 2.2, 2.3
    """
    # Set up multiple valid API keys
    key1 = "valid-key-1"
    key2 = "valid-key-2"
    key3 = "valid-key-3"
    os.environ["API_KEYS"] = f"{key1},{key2},{key3}"
    config_module._settings = None
    
    client = TestClient(test_app)
    
    # Test each valid key
    for key in [key1, key2, key3]:
        response = client.get("/test-endpoint", headers={"x-api-key": key})
        assert response.status_code == 200, \
            f"Expected 200 for valid API key '{key}', got {response.status_code}"
        assert response.json()["message"] == "success", \
            f"Valid key '{key}' should allow access"
    
    # Test an invalid key
    response = client.get("/test-endpoint", headers={"x-api-key": "invalid-key"})
    assert response.status_code == 403, \
        "Invalid key should return 403"


def test_api_key_with_whitespace(test_app):
    """
    Test that API keys with surrounding whitespace are handled correctly.
    
    Requirements: 2.2, 2.3
    """
    # Set up API key with whitespace in environment (should be trimmed)
    valid_key = "valid-key-with-spaces"
    os.environ["API_KEYS"] = f"  {valid_key}  , another-key "
    config_module._settings = None
    
    client = TestClient(test_app)
    
    # Test with exact key (no whitespace)
    response = client.get("/test-endpoint", headers={"x-api-key": valid_key})
    assert response.status_code == 200, \
        "Trimmed key should work"
    
    # Test with key including whitespace (should fail)
    response = client.get("/test-endpoint", headers={"x-api-key": f"  {valid_key}  "})
    assert response.status_code == 403, \
        "Key with whitespace should not match trimmed key"


def test_case_sensitive_api_keys(test_app):
    """
    Test that API key validation is case-sensitive.
    
    Requirements: 2.3
    """
    valid_key = "ValidKey123"
    os.environ["API_KEYS"] = valid_key
    config_module._settings = None
    
    client = TestClient(test_app)
    
    # Test with correct case
    response = client.get("/test-endpoint", headers={"x-api-key": valid_key})
    assert response.status_code == 200, \
        "Exact case match should succeed"
    
    # Test with different case
    response = client.get("/test-endpoint", headers={"x-api-key": valid_key.lower()})
    assert response.status_code == 403, \
        "Different case should fail (case-sensitive)"
    
    response = client.get("/test-endpoint", headers={"x-api-key": valid_key.upper()})
    assert response.status_code == 403, \
        "Different case should fail (case-sensitive)"
