"""
Unit tests for health check endpoint.

These tests verify that the /health endpoint returns the correct status
and response format for monitoring and deployment verification.
"""

import os
import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import core.config as config_module


@pytest.fixture(autouse=True)
def setup_environment(monkeypatch):
    """Set up environment variables for testing."""
    # Reset settings singleton
    config_module._settings = None
    
    # Set required environment variables
    monkeypatch.setenv("API_KEYS", "test-key-12345")
    
    yield
    
    # Clean up
    config_module._settings = None


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    # Import app after environment is set up
    from main import app
    return TestClient(app)


def test_health_check_returns_200(client):
    """
    Test that the health check endpoint returns 200 status.
    
    Requirements: 9.5
    """
    response = client.get("/health")
    
    # Verify 200 status code
    assert response.status_code == 200, \
        f"Expected 200 for health check, got {response.status_code}"


def test_health_check_response_format(client):
    """
    Test that the health check endpoint returns the correct response format.
    
    The response should contain:
    - status: "healthy"
    - timestamp: ISO format timestamp
    - version: API version string
    
    Requirements: 9.5
    """
    response = client.get("/health")
    
    # Verify successful response
    assert response.status_code == 200, \
        "Health check should return 200"
    
    # Parse response JSON
    data = response.json()
    
    # Verify required fields are present
    assert "status" in data, \
        "Response should contain 'status' field"
    assert "timestamp" in data, \
        "Response should contain 'timestamp' field"
    assert "version" in data, \
        "Response should contain 'version' field"
    
    # Verify field values
    assert data["status"] == "healthy", \
        "Status should be 'healthy'"
    assert data["version"] == "1.0.0", \
        "Version should be '1.0.0'"
    
    # Verify timestamp is in ISO format
    timestamp = data["timestamp"]
    assert isinstance(timestamp, str), \
        "Timestamp should be a string"
    
    # Verify timestamp can be parsed as ISO format
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp)
        assert parsed_timestamp is not None, \
            "Timestamp should be parseable as ISO format"
    except ValueError:
        pytest.fail(f"Timestamp '{timestamp}' is not in valid ISO format")


def test_health_check_no_authentication_required(client):
    """
    Test that the health check endpoint does not require authentication.
    
    This is important for deployment platforms to monitor the service
    without needing API keys.
    
    Requirements: 9.5
    """
    # Make request without any authentication headers
    response = client.get("/health")
    
    # Verify successful response without authentication
    assert response.status_code == 200, \
        "Health check should not require authentication"
    
    data = response.json()
    assert data["status"] == "healthy", \
        "Health check should return healthy status without authentication"
