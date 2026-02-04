"""
Integration tests for AI Voice Detection API.

These tests verify the complete detection flow from request to response,
including authentication, validation, audio processing, and error handling.
"""

import base64
import os
import pytest
import numpy as np
import soundfile as sf
from io import BytesIO
from fastapi.testclient import TestClient
import core.config as config_module


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset settings singleton before and after each test."""
    config_module._settings = None
    yield
    config_module._settings = None
    # Clean up environment variables
    for key in ["API_KEYS", "MODEL_NAME", "MODEL_CACHE_DIR", "LOG_LEVEL"]:
        if key in os.environ:
            del os.environ[key]


@pytest.fixture
def setup_environment():
    """Set up environment variables for testing."""
    os.environ["API_KEYS"] = "test-valid-key-12345"
    os.environ["MODEL_NAME"] = "facebook/wav2vec2-base"
    os.environ["MODEL_CACHE_DIR"] = "/tmp/models"
    os.environ["LOG_LEVEL"] = "ERROR"
    config_module._settings = None
    yield
    # Cleanup is handled by reset_settings fixture


@pytest.fixture
def client(setup_environment):
    """Create a test client for the FastAPI application."""
    from main import app
    return TestClient(app)


@pytest.fixture
def valid_api_key():
    """Provide a valid API key for testing."""
    return "test-valid-key-12345"


@pytest.fixture
def valid_audio_base64():
    """Generate valid audio data encoded as base64."""
    # Generate a simple sine wave audio
    sample_rate = 16000
    duration = 1.0  # 1 second
    frequency = 440.0  # A4 note
    
    # Generate audio samples
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    # Create an in-memory audio buffer in WAV format
    audio_buffer = BytesIO()
    sf.write(audio_buffer, audio_data, sample_rate, format='WAV')
    audio_buffer.seek(0)
    
    # Encode to base64
    audio_bytes = audio_buffer.read()
    return base64.b64encode(audio_bytes).decode()


def test_complete_detection_flow_with_valid_audio(client, valid_api_key, valid_audio_base64):
    """
    Test complete detection flow with valid audio.
    
    This test verifies the entire pipeline:
    1. Authentication passes with valid API key
    2. Request validation succeeds
    3. Audio decoding succeeds
    4. Feature extraction succeeds
    5. Model prediction succeeds
    6. Response contains all required fields
    
    Requirements: 1.2
    """
    # Prepare request
    request_data = {
        "language": "en",
        "audio_format": "wav",
        "audio_base64": valid_audio_base64
    }
    headers = {"x-api-key": valid_api_key}
    
    # Make request
    response = client.post("/api/v1/detect", json=request_data, headers=headers)
    
    # Verify successful response
    assert response.status_code == 200, \
        f"Expected 200 for valid request, got {response.status_code}. Response: {response.json()}"
    
    # Verify response structure
    data = response.json()
    assert "is_ai_generated" in data, "Response should contain 'is_ai_generated' field"
    assert "confidence" in data, "Response should contain 'confidence' field"
    assert "detected_language" in data, "Response should contain 'detected_language' field"
    assert "message" in data, "Response should contain 'message' field"
    
    # Verify field types
    assert isinstance(data["is_ai_generated"], bool), \
        "is_ai_generated should be boolean"
    assert isinstance(data["confidence"], (int, float)), \
        "confidence should be numeric"
    assert isinstance(data["detected_language"], str), \
        "detected_language should be string"
    assert isinstance(data["message"], str), \
        "message should be string"
    
    # Verify confidence bounds
    assert 0.0 <= data["confidence"] <= 1.0, \
        f"confidence should be between 0.0 and 1.0, got {data['confidence']}"
    
    # Verify language preservation
    assert data["detected_language"] == "en", \
        "detected_language should match input language"
    
    # Verify message is non-empty
    assert len(data["message"]) > 0, \
        "message should not be empty"


def test_authentication_flow(client, valid_audio_base64):
    """
    Test authentication flow with various API key scenarios.
    
    This test verifies:
    1. Missing API key returns 401
    2. Invalid API key returns 403
    3. Valid API key allows access
    
    Requirements: 2.1
    """
    request_data = {
        "language": "en",
        "audio_format": "wav",
        "audio_base64": valid_audio_base64
    }
    
    # Test 1: Missing API key
    response = client.post("/api/v1/detect", json=request_data)
    assert response.status_code == 401, \
        f"Expected 401 for missing API key, got {response.status_code}"
    assert "error" in response.json(), \
        "Error response should contain 'error' field"
    
    # Test 2: Invalid API key
    headers = {"x-api-key": "invalid-key-xyz"}
    response = client.post("/api/v1/detect", json=request_data, headers=headers)
    assert response.status_code == 403, \
        f"Expected 403 for invalid API key, got {response.status_code}"
    assert "error" in response.json(), \
        "Error response should contain 'error' field"
    
    # Test 3: Valid API key
    headers = {"x-api-key": "test-valid-key-12345"}
    response = client.post("/api/v1/detect", json=request_data, headers=headers)
    assert response.status_code == 200, \
        f"Expected 200 for valid API key, got {response.status_code}"


def test_validation_errors(client, valid_api_key):
    """
    Test validation errors for missing or invalid fields.
    
    This test verifies that the API properly validates:
    1. Missing required fields
    2. Invalid base64 encoding
    3. Invalid audio format
    
    Requirements: 3.3
    """
    headers = {"x-api-key": valid_api_key}
    
    # Test 1: Missing language field
    request_data = {
        "audio_format": "mp3",
        "audio_base64": base64.b64encode(b"test").decode()
    }
    response = client.post("/api/v1/detect", json=request_data, headers=headers)
    assert response.status_code == 422, \
        f"Expected 422 for missing language field, got {response.status_code}"
    
    # Test 2: Missing audio_format field
    request_data = {
        "language": "en",
        "audio_base64": base64.b64encode(b"test").decode()
    }
    response = client.post("/api/v1/detect", json=request_data, headers=headers)
    assert response.status_code == 422, \
        f"Expected 422 for missing audio_format field, got {response.status_code}"
    
    # Test 3: Missing audio_base64 field
    request_data = {
        "language": "en",
        "audio_format": "mp3"
    }
    response = client.post("/api/v1/detect", json=request_data, headers=headers)
    assert response.status_code == 422, \
        f"Expected 422 for missing audio_base64 field, got {response.status_code}"
    
    # Test 4: Invalid base64 encoding
    request_data = {
        "language": "en",
        "audio_format": "mp3",
        "audio_base64": "!!!invalid-base64!!!"
    }
    response = client.post("/api/v1/detect", json=request_data, headers=headers)
    assert response.status_code == 422, \
        f"Expected 422 for invalid base64, got {response.status_code}"
    
    # Test 5: Invalid audio format
    request_data = {
        "language": "en",
        "audio_format": "invalid",
        "audio_base64": base64.b64encode(b"test").decode()
    }
    response = client.post("/api/v1/detect", json=request_data, headers=headers)
    assert response.status_code == 422, \
        f"Expected 422 for invalid audio format, got {response.status_code}"


def test_error_handling_for_invalid_audio(client, valid_api_key):
    """
    Test error handling for invalid audio data.
    
    This test verifies that the API properly handles:
    1. Corrupted audio data
    2. Empty audio data (caught by validation)
    3. Non-audio binary data
    
    Requirements: 4.5
    """
    headers = {"x-api-key": valid_api_key}
    
    # Test 1: Empty audio data (caught by validation - min_length=1)
    request_data = {
        "language": "en",
        "audio_format": "mp3",
        "audio_base64": base64.b64encode(b"").decode()
    }
    response = client.post("/api/v1/detect", json=request_data, headers=headers)
    # Empty base64 is caught by validation (422) due to min_length constraint
    assert response.status_code == 422, \
        f"Expected 422 for empty audio (validation error), got {response.status_code}"
    
    # Test 2: Non-audio binary data
    request_data = {
        "language": "en",
        "audio_format": "mp3",
        "audio_base64": base64.b64encode(b"not-valid-audio-data").decode()
    }
    response = client.post("/api/v1/detect", json=request_data, headers=headers)
    # Should return 400 (audio processing error) or 500 (model error)
    assert response.status_code in [400, 500], \
        f"Expected 400 or 500 for invalid audio, got {response.status_code}"
    
    # Test 3: Very short audio data (might cause feature extraction issues)
    request_data = {
        "language": "en",
        "audio_format": "mp3",
        "audio_base64": base64.b64encode(b"x" * 10).decode()
    }
    response = client.post("/api/v1/detect", json=request_data, headers=headers)
    # Should return 400 (audio processing error) or 500 (model error)
    assert response.status_code in [400, 500], \
        f"Expected 400 or 500 for short audio, got {response.status_code}"
    
    # Verify error responses have proper structure
    if response.status_code >= 400:
        data = response.json()
        # Should have either 'detail' (HTTPException) or 'error'/'detail' (custom)
        assert "detail" in data or "error" in data, \
            "Error response should contain error information"


def test_model_inference_error_handling(client, valid_api_key):
    """
    Test error handling for model inference failures.
    
    This test verifies that the API properly handles model inference errors
    and returns appropriate error responses.
    
    Requirements: 5.5
    """
    headers = {"x-api-key": valid_api_key}
    
    # Generate audio that might cause model issues (very unusual audio)
    # Create audio with extreme values
    sample_rate = 16000
    duration = 0.1  # Very short
    num_samples = int(sample_rate * duration)
    
    # Create audio with unusual characteristics
    audio_data = np.random.random(num_samples).astype(np.float32) * 2 - 1  # Random noise
    
    audio_buffer = BytesIO()
    sf.write(audio_buffer, audio_data, sample_rate, format='WAV')
    audio_buffer.seek(0)
    audio_base64 = base64.b64encode(audio_buffer.read()).decode()
    
    request_data = {
        "language": "en",
        "audio_format": "wav",
        "audio_base64": audio_base64
    }
    
    response = client.post("/api/v1/detect", json=request_data, headers=headers)
    
    # The model should either succeed or return an error
    # It should NOT crash
    assert response.status_code in [200, 400, 500], \
        f"Expected 200, 400, or 500, got {response.status_code}"
    
    # If it's an error, verify proper error structure
    if response.status_code >= 400:
        data = response.json()
        assert "detail" in data or "error" in data, \
            "Error response should contain error information"
        
        # Verify no stack traces are exposed
        response_text = str(data).lower()
        assert "traceback" not in response_text, \
            "Error response should not contain stack traces"


def test_multiple_requests_with_same_client(client, valid_api_key, valid_audio_base64):
    """
    Test that multiple requests work correctly with the same client.
    
    This verifies that:
    1. The model service is reused across requests
    2. No state leakage between requests
    3. Consistent results for same input
    
    Requirements: 1.2
    """
    headers = {"x-api-key": valid_api_key}
    request_data = {
        "language": "en",
        "audio_format": "wav",
        "audio_base64": valid_audio_base64
    }
    
    # Make multiple requests
    responses = []
    for i in range(3):
        response = client.post("/api/v1/detect", json=request_data, headers=headers)
        assert response.status_code == 200, \
            f"Request {i+1} failed with status {response.status_code}"
        responses.append(response.json())
    
    # Verify all responses have the same structure
    for i, data in enumerate(responses):
        assert "is_ai_generated" in data, f"Response {i+1} missing is_ai_generated"
        assert "confidence" in data, f"Response {i+1} missing confidence"
        assert "detected_language" in data, f"Response {i+1} missing detected_language"
        assert "message" in data, f"Response {i+1} missing message"
    
    # Verify consistent results for same input
    # (Results should be deterministic for the same audio)
    first_result = responses[0]
    for i, result in enumerate(responses[1:], start=2):
        assert result["is_ai_generated"] == first_result["is_ai_generated"], \
            f"Request {i} has different is_ai_generated than request 1"
        assert abs(result["confidence"] - first_result["confidence"]) < 0.01, \
            f"Request {i} has significantly different confidence than request 1"


def test_different_languages(client, valid_api_key, valid_audio_base64):
    """
    Test that different language codes are handled correctly.
    
    This verifies that:
    1. Different language codes are accepted
    2. Language is preserved in response
    
    Requirements: 1.2
    """
    headers = {"x-api-key": valid_api_key}
    
    languages = ["en", "es", "fr", "de", "zh", "ja"]
    
    for language in languages:
        request_data = {
            "language": language,
            "audio_format": "wav",
            "audio_base64": valid_audio_base64
        }
        
        response = client.post("/api/v1/detect", json=request_data, headers=headers)
        assert response.status_code == 200, \
            f"Request with language '{language}' failed with status {response.status_code}"
        
        data = response.json()
        assert data["detected_language"] == language, \
            f"Expected detected_language '{language}', got '{data['detected_language']}'"


def test_error_response_format(client, valid_api_key):
    """
    Test that error responses follow the correct format.
    
    This verifies that all error responses:
    1. Have proper JSON structure
    2. Contain error information
    3. Don't expose internal details
    
    Requirements: 3.3, 4.5, 5.5
    """
    headers = {"x-api-key": valid_api_key}
    
    # Trigger a validation error
    request_data = {
        "language": "en",
        "audio_format": "mp3",
        "audio_base64": "!!!invalid!!!"
    }
    response = client.post("/api/v1/detect", json=request_data, headers=headers)
    
    # Verify error response structure
    assert response.status_code == 422, \
        f"Expected 422 for validation error, got {response.status_code}"
    
    data = response.json()
    assert isinstance(data, dict), "Error response should be a JSON object"
    assert "detail" in data, "Error response should contain 'detail' field"
    
    # Verify no stack traces
    response_text = str(data).lower()
    assert "traceback" not in response_text, \
        "Error response should not contain stack traces"
    assert "file \"" not in response_text, \
        "Error response should not contain file paths"


def test_health_endpoint_integration(client):
    """
    Test that the health endpoint works correctly in integration.
    
    This verifies that:
    1. Health endpoint is accessible
    2. Returns correct format
    3. Doesn't require authentication
    
    Requirements: 1.2
    """
    # Test without authentication
    response = client.get("/health")
    
    assert response.status_code == 200, \
        f"Expected 200 for health check, got {response.status_code}"
    
    data = response.json()
    assert "status" in data, "Health response should contain 'status' field"
    assert data["status"] == "healthy", "Status should be 'healthy'"
    assert "timestamp" in data, "Health response should contain 'timestamp' field"
    assert "version" in data, "Health response should contain 'version' field"
