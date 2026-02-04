"""
Unit tests for API schemas.
"""
import base64
import pytest
from pydantic import ValidationError
from api.schemas import DetectionRequest, DetectionResponse, ErrorResponse


def test_detection_request_valid():
    """Test that valid DetectionRequest is accepted."""
    audio_data = b"test audio data"
    audio_base64 = base64.b64encode(audio_data).decode()
    
    request = DetectionRequest(
        language="en",
        audio_format="mp3",
        audio_base64=audio_base64
    )
    
    assert request.language == "en"
    assert request.audio_format == "mp3"
    assert request.audio_base64 == audio_base64


def test_detection_request_invalid_base64():
    """Test that invalid base64 raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        DetectionRequest(
            language="en",
            audio_format="mp3",
            audio_base64="not-valid-base64!!!"
        )
    
    assert "Invalid base64 encoding" in str(exc_info.value)


def test_detection_request_missing_fields():
    """Test that missing required fields raise validation error."""
    with pytest.raises(ValidationError):
        DetectionRequest(language="en")


def test_detection_request_invalid_audio_format():
    """Test that invalid audio format raises validation error."""
    audio_data = b"test audio data"
    audio_base64 = base64.b64encode(audio_data).decode()
    
    with pytest.raises(ValidationError):
        DetectionRequest(
            language="en",
            audio_format="invalid",
            audio_base64=audio_base64
        )


def test_detection_response_valid():
    """Test that valid DetectionResponse is created."""
    response = DetectionResponse(
        is_ai_generated=True,
        confidence=0.87,
        detected_language="en",
        message="Voice detected as AI-generated"
    )
    
    assert response.is_ai_generated is True
    assert response.confidence == 0.87
    assert response.detected_language == "en"
    assert response.message == "Voice detected as AI-generated"


def test_detection_response_confidence_bounds():
    """Test that confidence must be between 0 and 1."""
    # Valid bounds
    DetectionResponse(
        is_ai_generated=True,
        confidence=0.0,
        detected_language="en",
        message="Test"
    )
    
    DetectionResponse(
        is_ai_generated=True,
        confidence=1.0,
        detected_language="en",
        message="Test"
    )
    
    # Invalid bounds
    with pytest.raises(ValidationError):
        DetectionResponse(
            is_ai_generated=True,
            confidence=1.5,
            detected_language="en",
            message="Test"
        )
    
    with pytest.raises(ValidationError):
        DetectionResponse(
            is_ai_generated=True,
            confidence=-0.1,
            detected_language="en",
            message="Test"
        )


def test_error_response_valid():
    """Test that valid ErrorResponse is created."""
    error = ErrorResponse(
        error="AudioDecodingError",
        detail="Failed to decode audio data",
        status_code=400
    )
    
    assert error.error == "AudioDecodingError"
    assert error.detail == "Failed to decode audio data"
    assert error.status_code == 400
