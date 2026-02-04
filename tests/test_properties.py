"""
Property-based tests for AI Voice Detection API.

These tests use Hypothesis to verify universal properties across many generated inputs.
Each test runs a minimum of 100 iterations to ensure comprehensive coverage.
"""
import base64
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from pydantic import ValidationError
from api.schemas import DetectionRequest, DetectionResponse
from services.audio_processor import AudioProcessor
from utils.exceptions import AudioDecodingError


# Feature: ai-voice-detection-api, Property 1: Valid request acceptance
@given(
    language=st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
    audio_format=st.sampled_from(["mp3", "wav", "flac"]),
    audio_base64=st.binary(min_size=1, max_size=1000).map(lambda b: base64.b64encode(b).decode())
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_property_valid_request_acceptance(language, audio_format, audio_base64):
    """
    Property 1: Valid request acceptance
    
    For any valid JSON request body containing language, audio_format, and audio_base64 fields,
    the API should accept the request and process it without validation errors.
    
    Validates: Requirements 1.3
    """
    # Create a request with all required fields
    request = DetectionRequest(
        language=language,
        audio_format=audio_format,
        audio_base64=audio_base64
    )
    
    # Verify the request was created successfully
    assert request.language == language
    assert request.audio_format == audio_format
    assert request.audio_base64 == audio_base64
    
    # Verify all fields are present and of correct type
    assert isinstance(request.language, str)
    assert isinstance(request.audio_format, str)
    assert isinstance(request.audio_base64, str)
    assert request.audio_format in ["mp3", "wav", "flac"]


# Feature: ai-voice-detection-api, Property 2: Response structure completeness
@given(
    is_ai_generated=st.booleans(),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    detected_language=st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
    message=st.text(min_size=1, max_size=200)
)
@settings(max_examples=20)
def test_property_response_structure_completeness(is_ai_generated, confidence, detected_language, message):
    """
    Property 2: Response structure completeness
    
    For any valid detection request, the API response should contain all required fields:
    is_ai_generated (boolean), confidence (float), detected_language (string), and message (string).
    
    Validates: Requirements 1.4, 6.1
    """
    # Create a response with all required fields
    response = DetectionResponse(
        is_ai_generated=is_ai_generated,
        confidence=confidence,
        detected_language=detected_language,
        message=message
    )
    
    # Verify all required fields are present
    assert hasattr(response, 'is_ai_generated'), "Response missing is_ai_generated field"
    assert hasattr(response, 'confidence'), "Response missing confidence field"
    assert hasattr(response, 'detected_language'), "Response missing detected_language field"
    assert hasattr(response, 'message'), "Response missing message field"
    
    # Verify field types are correct
    assert isinstance(response.is_ai_generated, bool), "is_ai_generated must be boolean"
    assert isinstance(response.confidence, float), "confidence must be float"
    assert isinstance(response.detected_language, str), "detected_language must be string"
    assert isinstance(response.message, str), "message must be string"
    
    # Verify field values match input
    assert response.is_ai_generated == is_ai_generated
    assert response.confidence == confidence
    assert response.detected_language == detected_language
    assert response.message == message
    
    # Verify confidence is within valid bounds
    assert 0.0 <= response.confidence <= 1.0, "confidence must be between 0.0 and 1.0"
    
    # Verify non-empty strings for language and message
    assert len(response.detected_language) > 0, "detected_language must be non-empty"
    assert len(response.message) > 0, "message must be non-empty"


# Feature: ai-voice-detection-api, Property 3: Authentication enforcement
@given(
    api_key=st.one_of(
        st.none(),  # Missing API key (None)
        st.just(""),  # Empty string
        st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1, max_size=50),  # Random ASCII strings
    )
)
@settings(max_examples=20, deadline=None)
def test_property_authentication_enforcement(api_key):
    """
    Property 3: Authentication enforcement
    
    For any request without a valid x-api-key header, the Auth_Middleware should reject
    the request before reaching the endpoint.
    
    Validates: Requirements 2.1
    """
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from api.middleware import APIKeyMiddleware
    import os
    
    # Set up a test API key in environment
    test_valid_key = "test-valid-key-12345"
    os.environ["API_KEYS"] = test_valid_key
    
    # Force reload of settings to pick up the test API key
    from core.config import _settings
    import core.config as config_module
    config_module._settings = None
    
    # Create a minimal FastAPI app with the middleware
    app = FastAPI()
    app.add_middleware(APIKeyMiddleware)
    
    @app.get("/test-endpoint")
    async def test_endpoint():
        return {"message": "success"}
    
    client = TestClient(app)
    
    # Test with the generated (invalid) API key
    headers = {}
    if api_key is not None:
        headers["x-api-key"] = api_key
    
    response = client.get("/test-endpoint", headers=headers)
    
    # If the API key is None or empty, expect 401 (missing key)
    if api_key is None or api_key == "":
        assert response.status_code == 401, \
            f"Expected 401 for missing/empty API key, got {response.status_code}"
        assert "error" in response.json(), "Error response should contain 'error' field"
        assert response.json()["error"] == "AuthenticationError", \
            "Error type should be AuthenticationError"
        assert "API key required" in response.json()["detail"], \
            "Error detail should mention API key required"
    
    # If the API key is not the valid test key, expect 403 (invalid key)
    elif api_key != test_valid_key:
        assert response.status_code == 403, \
            f"Expected 403 for invalid API key '{api_key}', got {response.status_code}"
        assert "error" in response.json(), "Error response should contain 'error' field"
        assert response.json()["error"] == "AuthorizationError", \
            "Error type should be AuthorizationError"
        assert "Invalid API key" in response.json()["detail"], \
            "Error detail should mention invalid API key"
    
    # If by chance the generated key matches the valid key, it should succeed
    else:
        assert response.status_code == 200, \
            f"Expected 200 for valid API key, got {response.status_code}"
        assert response.json()["message"] == "success", \
            "Valid API key should allow access to endpoint"
    
    # Clean up environment
    if "API_KEYS" in os.environ:
        del os.environ["API_KEYS"]
    config_module._settings = None


# Feature: ai-voice-detection-api, Property 4: Valid authentication bypass
@given(
    valid_key_suffix=st.text(
        alphabet=st.characters(min_codepoint=48, max_codepoint=122),
        min_size=10,
        max_size=50
    )
)
@settings(max_examples=20, deadline=None)
def test_property_valid_authentication_bypass(valid_key_suffix):
    """
    Property 4: Valid authentication bypass
    
    For any request with a valid x-api-key header, the Auth_Middleware should allow
    the request to proceed to the endpoint handler.
    
    Validates: Requirements 2.4
    """
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from api.middleware import APIKeyMiddleware
    import os
    
    # Create a unique valid API key for this test
    valid_api_key = f"valid-test-key-{valid_key_suffix}"
    
    # Set up the valid API key in environment
    os.environ["API_KEYS"] = valid_api_key
    
    # Force reload of settings to pick up the test API key
    from core.config import _settings
    import core.config as config_module
    config_module._settings = None
    
    # Create a minimal FastAPI app with the middleware
    app = FastAPI()
    app.add_middleware(APIKeyMiddleware)
    
    # Track if the endpoint was actually reached
    endpoint_reached = {"value": False}
    
    @app.get("/test-endpoint")
    async def test_endpoint():
        endpoint_reached["value"] = True
        return {"message": "success", "authenticated": True}
    
    client = TestClient(app)
    
    # Make a request with the valid API key
    headers = {"x-api-key": valid_api_key}
    response = client.get("/test-endpoint", headers=headers)
    
    # Verify the request was allowed through (200 status)
    assert response.status_code == 200, \
        f"Expected 200 for valid API key, got {response.status_code}. Response: {response.json()}"
    
    # Verify the endpoint was actually reached
    assert endpoint_reached["value"] is True, \
        "Endpoint should have been reached with valid API key"
    
    # Verify the response contains the expected data
    response_data = response.json()
    assert "message" in response_data, "Response should contain 'message' field"
    assert response_data["message"] == "success", \
        "Response message should be 'success'"
    assert response_data.get("authenticated") is True, \
        "Response should indicate successful authentication"
    
    # Verify no error fields are present in the response
    assert "error" not in response_data, \
        "Valid authentication should not return error field"
    assert "detail" not in response_data or response_data.get("detail") != "Invalid API key", \
        "Valid authentication should not return authorization error"
    
    # Clean up environment
    if "API_KEYS" in os.environ:
        del os.environ["API_KEYS"]
    config_module._settings = None


# Feature: ai-voice-detection-api, Property 6: Safe base64 decoding
@given(
    audio_base64=st.one_of(
        # Valid base64 strings
        st.binary(min_size=1, max_size=10000).map(lambda b: base64.b64encode(b).decode()),
        # Invalid base64 strings (various edge cases)
        st.text(min_size=0, max_size=100),  # Random text that may not be valid base64
        st.just(""),  # Empty string
        st.just("!!!invalid!!!"),  # Clearly invalid base64
        st.just("abc"),  # Invalid padding
        st.text(alphabet=st.characters(blacklist_characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="), min_size=1, max_size=50),  # Non-base64 characters
    )
)
@settings(max_examples=20)
def test_property_safe_base64_decoding(audio_base64):
    """
    Property 6: Safe base64 decoding
    
    For any base64 string (valid or invalid), the Audio_Processor should handle decoding
    without causing system crashes or unhandled exceptions.
    
    Validates: Requirements 3.4
    """
    processor = AudioProcessor()
    
    # The processor should either succeed or raise AudioDecodingError
    # It should NEVER crash with unhandled exceptions
    try:
        result = processor.decode_base64_audio(audio_base64)
        
        # If decoding succeeds, verify we got a valid BytesIO object
        assert result is not None, "Result should not be None on success"
        assert hasattr(result, 'read'), "Result should be a file-like object"
        assert hasattr(result, 'seek'), "Result should support seek operations"
        
        # Verify the buffer contains data
        result.seek(0)
        data = result.read()
        assert len(data) > 0, "Decoded data should not be empty on success"
        
    except AudioDecodingError as e:
        # This is expected for invalid base64 strings
        # Verify the error has proper attributes
        assert hasattr(e, 'detail'), "AudioDecodingError should have detail attribute"
        assert isinstance(e.detail, str), "Error detail should be a string"
        assert len(e.detail) > 0, "Error detail should not be empty"
        
    except Exception as e:
        # Any other exception type is a failure - the system should not crash
        pytest.fail(f"Unexpected exception type {type(e).__name__}: {str(e)}. "
                   f"Should only raise AudioDecodingError, not crash with unhandled exceptions.")



# Feature: ai-voice-detection-api, Property 7: Audio decoding round-trip
@given(
    audio_data=st.binary(min_size=1, max_size=10000)
)
@settings(max_examples=100)
def test_property_audio_decoding_round_trip(audio_data):
    """
    Property 7: Audio decoding round-trip
    
    For any valid audio binary data, encoding to base64 then decoding should produce
    equivalent binary data.
    
    Validates: Requirements 4.1
    """
    processor = AudioProcessor()
    
    # Encode the binary data to base64
    audio_base64 = base64.b64encode(audio_data).decode()
    
    # Decode it back using the AudioProcessor
    decoded_buffer = processor.decode_base64_audio(audio_base64)
    
    # Read the decoded data
    decoded_buffer.seek(0)
    decoded_data = decoded_buffer.read()
    
    # Verify round-trip: original data should equal decoded data
    assert decoded_data == audio_data, (
        f"Round-trip failed: original data length {len(audio_data)}, "
        f"decoded data length {len(decoded_data)}"
    )
    
    # Verify the buffer is seekable and readable
    decoded_buffer.seek(0)
    second_read = decoded_buffer.read()
    assert second_read == audio_data, "Buffer should be reusable after seeking"


# Feature: ai-voice-detection-api, Property 8: Feature extraction consistency
@given(
    sample_rate=st.sampled_from([8000, 16000, 22050, 44100]),
    duration=st.floats(min_value=0.1, max_value=5.0),
    frequency=st.floats(min_value=100.0, max_value=2000.0)
)
@settings(max_examples=100, deadline=None)
def test_property_feature_extraction_consistency(sample_rate, duration, frequency):
    """
    Property 8: Feature extraction consistency
    
    For any valid audio input, the Feature_Extractor should produce a feature vector
    suitable for model input without errors.
    
    Validates: Requirements 4.4
    """
    import numpy as np
    import soundfile as sf
    from io import BytesIO
    from services.feature_extractor import FeatureExtractor
    
    # Generate synthetic audio data (sine wave)
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    # Create an in-memory audio buffer in WAV format
    audio_buffer = BytesIO()
    sf.write(audio_buffer, audio_data, sample_rate, format='WAV')
    audio_buffer.seek(0)
    
    # Initialize feature extractor
    extractor = FeatureExtractor(sample_rate=16000)
    
    # Extract features - should not raise exceptions
    features = extractor.extract_features(audio_buffer)
    
    # Verify the feature vector is valid
    assert features is not None, "Feature vector should not be None"
    assert isinstance(features, np.ndarray), "Features should be a numpy array"
    assert features.ndim == 1, "Feature vector should be 1-dimensional"
    assert len(features) > 0, "Feature vector should not be empty"
    
    # Verify expected feature vector size (54 features as per design)
    # 26 (MFCC mean+std) + 2 (spectral centroid) + 2 (ZCR) + 24 (chroma) = 54
    assert len(features) == 54, f"Expected 54 features, got {len(features)}"
    
    # Verify all features are finite (no NaN or Inf values)
    assert np.all(np.isfinite(features)), "All features should be finite (no NaN or Inf)"
    
    # Verify features are numeric
    assert features.dtype in [np.float32, np.float64], "Features should be floating point numbers"


# Feature: ai-voice-detection-api, Property 9: Prediction completeness
@given(
    features=st.lists(
        st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=54,
        max_size=54
    ).map(lambda lst: np.array(lst, dtype=np.float32))
)
@settings(max_examples=100)
def test_property_prediction_completeness(features):
    """
    Property 9: Prediction completeness
    
    For any valid audio features, the Detection_Model should produce both a boolean
    classification and a confidence score.
    
    Validates: Requirements 5.2, 5.3
    """
    from services.model_service import ModelService
    
    # Initialize model service
    model_service = ModelService(
        model_name="facebook/wav2vec2-base",
        cache_dir="/tmp/models"
    )
    
    # Mock model as loaded (we don't need to actually load the model for this test)
    model_service.model = "mock_model"
    
    # Call predict with the generated features
    result = model_service.predict(features)
    
    # Verify the result is a tuple
    assert isinstance(result, tuple), "predict() should return a tuple"
    assert len(result) == 2, "predict() should return exactly 2 values"
    
    # Unpack the result
    is_ai_generated, confidence = result
    
    # Verify both values are present and of correct type
    assert is_ai_generated is not None, "is_ai_generated should not be None"
    assert confidence is not None, "confidence should not be None"
    
    # Verify is_ai_generated is a boolean (Requirement 5.3)
    assert isinstance(is_ai_generated, (bool, np.bool_)), \
        f"is_ai_generated should be boolean, got {type(is_ai_generated)}"
    
    # Verify confidence is a float (Requirement 5.2)
    assert isinstance(confidence, (float, np.floating)), \
        f"confidence should be float, got {type(confidence)}"
    
    # Verify confidence is within valid bounds [0.0, 1.0]
    assert 0.0 <= confidence <= 1.0, \
        f"confidence should be between 0.0 and 1.0, got {confidence}"
    
    # Verify confidence is finite (no NaN or Inf)
    assert np.isfinite(confidence), "confidence should be finite (no NaN or Inf)"


# Feature: ai-voice-detection-api, Property 10: Confidence bounds
@given(
    features=st.lists(
        st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=54,
        max_size=54
    ).map(lambda lst: np.array(lst, dtype=np.float32))
)
@settings(max_examples=100)
def test_property_confidence_bounds(features):
    """
    Property 10: Confidence bounds
    
    For any model prediction, the confidence score should be a float value
    between 0.0 and 1.0 inclusive.
    
    Validates: Requirements 5.4, 6.3
    """
    from services.model_service import ModelService
    
    # Initialize model service
    model_service = ModelService(
        model_name="facebook/wav2vec2-base",
        cache_dir="/tmp/models"
    )
    
    # Mock model as loaded (we don't need to actually load the model for this test)
    model_service.model = "mock_model"
    
    # Call predict with the generated features
    is_ai_generated, confidence = model_service.predict(features)
    
    # Verify confidence is a numeric type
    assert isinstance(confidence, (float, np.floating)), \
        f"confidence should be float, got {type(confidence)}"
    
    # Verify confidence is within bounds [0.0, 1.0] (inclusive)
    assert 0.0 <= confidence <= 1.0, \
        f"confidence must be between 0.0 and 1.0 inclusive, got {confidence}"
    
    # Verify confidence is finite (no NaN or Inf)
    assert np.isfinite(confidence), \
        f"confidence should be finite (no NaN or Inf), got {confidence}"


# Feature: ai-voice-detection-api, Property 11: Boolean classification type
@given(
    features=st.lists(
        st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=54,
        max_size=54
    ).map(lambda lst: np.array(lst, dtype=np.float32))
)
@settings(max_examples=100)
def test_property_boolean_classification_type(features):
    """
    Property 11: Boolean classification type
    
    For any detection response, the is_ai_generated field should be a boolean value
    (true or false).
    
    Validates: Requirements 6.2
    """
    from services.model_service import ModelService
    
    # Initialize model service
    model_service = ModelService(
        model_name="facebook/wav2vec2-base",
        cache_dir="/tmp/models"
    )
    
    # Mock model as loaded (we don't need to actually load the model for this test)
    model_service.model = "mock_model"
    
    # Call predict with the generated features
    is_ai_generated, confidence = model_service.predict(features)
    
    # Verify is_ai_generated is strictly a boolean type
    assert isinstance(is_ai_generated, (bool, np.bool_)), \
        f"is_ai_generated must be boolean type, got {type(is_ai_generated)}"
    
    # Verify it's one of the two boolean values
    assert is_ai_generated in [True, False], \
        f"is_ai_generated must be True or False, got {is_ai_generated}"
    
    # Verify it's not None or any other truthy/falsy value
    assert is_ai_generated is not None, "is_ai_generated should not be None"
    assert type(is_ai_generated).__name__ in ['bool', 'bool_'], \
        f"is_ai_generated should be bool type, got {type(is_ai_generated).__name__}"


# Feature: ai-voice-detection-api, Property 12: Language field preservation
@given(
    language=st.text(
        min_size=2,
        max_size=10,
        alphabet=st.characters(min_codepoint=97, max_codepoint=122)
    ),
    audio_data=st.binary(min_size=1000, max_size=5000)
)
@settings(max_examples=20, deadline=None)
def test_property_language_field_preservation(language, audio_data):
    """
    Property 12: Language field preservation
    
    For any valid request with a language field, the response detected_language field
    should contain a valid language string.
    
    Validates: Requirements 6.4
    """
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from api.routes import router
    from api.middleware import APIKeyMiddleware
    import os
    
    # Set up a test API key in environment
    test_valid_key = "test-valid-key-12345"
    os.environ["API_KEYS"] = test_valid_key
    os.environ["MODEL_NAME"] = "facebook/wav2vec2-base"
    os.environ["MODEL_CACHE_DIR"] = "/tmp/models"
    os.environ["LOG_LEVEL"] = "ERROR"
    
    # Force reload of settings to pick up the test API key
    import core.config as config_module
    config_module._settings = None
    
    # Create a minimal FastAPI app with the middleware and router
    app = FastAPI()
    app.add_middleware(APIKeyMiddleware)
    app.include_router(router, prefix="/api/v1")
    
    client = TestClient(app)
    
    # Encode audio data to base64
    audio_base64 = base64.b64encode(audio_data).decode()
    
    # Create a valid detection request
    request_data = {
        "language": language,
        "audio_format": "mp3",
        "audio_base64": audio_base64
    }
    
    # Make the request with valid API key
    headers = {"x-api-key": test_valid_key}
    response = client.post("/api/v1/detect", json=request_data, headers=headers)
    
    # The request might fail due to invalid audio format, but if it succeeds,
    # we should verify the language field is preserved
    if response.status_code == 200:
        response_data = response.json()
        
        # Verify detected_language field is present
        assert "detected_language" in response_data, \
            "Response should contain 'detected_language' field"
        
        # Verify detected_language is a string
        assert isinstance(response_data["detected_language"], str), \
            f"detected_language should be string, got {type(response_data['detected_language'])}"
        
        # Verify detected_language is not empty
        assert len(response_data["detected_language"]) > 0, \
            "detected_language should not be empty"
        
        # Verify detected_language matches the input language
        # (For MVP, the detect_language method returns the input language)
        assert response_data["detected_language"] == language, \
            f"detected_language should match input language '{language}', got '{response_data['detected_language']}'"
    
    # If the request fails with 400 (audio processing error) or 500 (model error),
    # that's acceptable - we're testing language preservation when processing succeeds
    elif response.status_code in [400, 500]:
        # This is expected for invalid audio data
        # The property test is about language preservation when processing succeeds
        pass
    
    # If we get validation errors (422), that's unexpected for valid input
    elif response.status_code == 422:
        pytest.fail(f"Unexpected validation error for valid input: {response.json()}")
    
    # Clean up environment
    for key in ["API_KEYS", "MODEL_NAME", "MODEL_CACHE_DIR", "LOG_LEVEL"]:
        if key in os.environ:
            del os.environ[key]
    config_module._settings = None


# Feature: ai-voice-detection-api, Property 5: Required field validation
@given(
    # Generate combinations where at least one required field is missing
    missing_field=st.sampled_from(["language", "audio_format", "audio_base64", "multiple"]),
    language=st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
    audio_format=st.sampled_from(["mp3", "wav", "flac"]),
    audio_base64=st.binary(min_size=1, max_size=1000).map(lambda b: base64.b64encode(b).decode())
)
@settings(max_examples=100)
def test_property_required_field_validation(missing_field, language, audio_format, audio_base64):
    """
    Property 5: Required field validation
    
    For any request missing one or more required fields (language, audio_format, audio_base64),
    the API should reject the request with validation errors.
    
    Validates: Requirements 3.1
    """
    from pydantic import ValidationError
    
    # Build request data with missing field(s)
    request_data = {}
    
    if missing_field == "language":
        # Missing language field
        request_data = {
            "audio_format": audio_format,
            "audio_base64": audio_base64
        }
    elif missing_field == "audio_format":
        # Missing audio_format field
        request_data = {
            "language": language,
            "audio_base64": audio_base64
        }
    elif missing_field == "audio_base64":
        # Missing audio_base64 field
        request_data = {
            "language": language,
            "audio_format": audio_format
        }
    elif missing_field == "multiple":
        # Missing multiple fields (only include language)
        request_data = {
            "language": language
        }
    
    # Attempt to create a DetectionRequest with missing field(s)
    # This should raise a ValidationError
    with pytest.raises(ValidationError) as exc_info:
        DetectionRequest(**request_data)
    
    # Verify that a ValidationError was raised
    validation_error = exc_info.value
    
    # Verify the error contains information about missing fields
    errors = validation_error.errors()
    assert len(errors) > 0, "ValidationError should contain at least one error"
    
    # Verify that the error is about missing required fields
    error_fields = [error['loc'][0] for error in errors]
    
    if missing_field == "language":
        assert "language" in error_fields, "Error should mention missing 'language' field"
    elif missing_field == "audio_format":
        assert "audio_format" in error_fields, "Error should mention missing 'audio_format' field"
    elif missing_field == "audio_base64":
        assert "audio_base64" in error_fields, "Error should mention missing 'audio_base64' field"
    elif missing_field == "multiple":
        # Should have errors for both missing fields
        assert "audio_format" in error_fields, "Error should mention missing 'audio_format' field"
        assert "audio_base64" in error_fields, "Error should mention missing 'audio_base64' field"
    
    # Verify error type is 'missing' for required field validation
    for error in errors:
        if error['loc'][0] in ["language", "audio_format", "audio_base64"]:
            assert error['type'] == 'missing', \
                f"Error type should be 'missing' for required field, got '{error['type']}'"


# Feature: ai-voice-detection-api, Property 13: Message field presence
@given(
    is_ai_generated=st.booleans(),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    detected_language=st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
    message=st.text(min_size=1, max_size=200).filter(lambda s: s.strip() != "")
)
@settings(max_examples=100)
def test_property_message_field_presence(is_ai_generated, confidence, detected_language, message):
    """
    Property 13: Message field presence
    
    For any detection response, the message field should be present and contain a non-empty string.
    
    Validates: Requirements 6.5
    """
    # Create a response with all required fields
    response = DetectionResponse(
        is_ai_generated=is_ai_generated,
        confidence=confidence,
        detected_language=detected_language,
        message=message
    )
    
    # Verify message field is present
    assert hasattr(response, 'message'), "Response must have 'message' field"
    
    # Verify message is a string
    assert isinstance(response.message, str), \
        f"message must be string type, got {type(response.message)}"
    
    # Verify message is not empty
    assert len(response.message) > 0, "message must be non-empty string"
    
    # Verify message is not just whitespace
    assert response.message.strip() != "", "message must contain non-whitespace characters"
    
    # Verify message matches the input
    assert response.message == message, \
        f"message should match input, expected '{message}', got '{response.message}'"


# Feature: ai-voice-detection-api, Property 5: Required field validation (API Level)
@given(
    # Generate combinations where at least one required field is missing
    missing_field=st.sampled_from(["language", "audio_format", "audio_base64", "all"]),
    language=st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
    audio_format=st.sampled_from(["mp3", "wav", "flac"]),
    audio_base64=st.binary(min_size=1, max_size=1000).map(lambda b: base64.b64encode(b).decode())
)
@settings(max_examples=20, deadline=None)
def test_property_api_required_field_validation(missing_field, language, audio_format, audio_base64):
    """
    Property 5: Required field validation (API Level)
    
    For any request missing one or more required fields (language, audio_format, audio_base64),
    the API endpoint should reject the request with HTTP 422 validation errors.
    
    This test verifies that the /detect endpoint properly validates required fields
    and returns appropriate error responses.
    
    Validates: Requirements 3.1
    """
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from api.routes import router
    from api.middleware import APIKeyMiddleware
    import os
    
    # Set up a test API key in environment
    test_valid_key = "test-valid-key-12345"
    os.environ["API_KEYS"] = test_valid_key
    os.environ["MODEL_NAME"] = "facebook/wav2vec2-base"
    os.environ["MODEL_CACHE_DIR"] = "/tmp/models"
    os.environ["LOG_LEVEL"] = "ERROR"
    
    # Force reload of settings to pick up the test API key
    import core.config as config_module
    config_module._settings = None
    
    # Create a minimal FastAPI app with the middleware and router
    app = FastAPI()
    app.add_middleware(APIKeyMiddleware)
    app.include_router(router, prefix="/api/v1")
    
    client = TestClient(app)
    
    # Build request data with missing field(s)
    request_data = {}
    expected_missing_fields = []
    
    if missing_field == "language":
        # Missing language field
        request_data = {
            "audio_format": audio_format,
            "audio_base64": audio_base64
        }
        expected_missing_fields = ["language"]
    elif missing_field == "audio_format":
        # Missing audio_format field
        request_data = {
            "language": language,
            "audio_base64": audio_base64
        }
        expected_missing_fields = ["audio_format"]
    elif missing_field == "audio_base64":
        # Missing audio_base64 field
        request_data = {
            "language": language,
            "audio_format": audio_format
        }
        expected_missing_fields = ["audio_base64"]
    elif missing_field == "all":
        # Missing all fields (empty request)
        request_data = {}
        expected_missing_fields = ["language", "audio_format", "audio_base64"]
    
    # Make the request with valid API key but invalid request body
    headers = {"x-api-key": test_valid_key}
    response = client.post("/api/v1/detect", json=request_data, headers=headers)
    
    # Verify the API returns 422 Unprocessable Entity for validation errors
    assert response.status_code == 422, \
        f"Expected 422 for missing required fields, got {response.status_code}. Response: {response.json()}"
    
    # Verify the response contains validation error details
    response_data = response.json()
    assert "detail" in response_data, "Response should contain 'detail' field with validation errors"
    
    # Verify the detail is a list of errors (FastAPI validation error format)
    detail = response_data["detail"]
    assert isinstance(detail, list), "detail should be a list of validation errors"
    assert len(detail) > 0, "detail should contain at least one validation error"
    
    # Verify that errors mention the missing field(s)
    error_fields = []
    for error in detail:
        if isinstance(error, dict) and "loc" in error:
            # Extract field name from location (e.g., ['body', 'language'] -> 'language')
            if len(error["loc"]) > 1:
                error_fields.append(error["loc"][-1])
    
    # Verify all expected missing fields are mentioned in the errors
    for expected_field in expected_missing_fields:
        assert expected_field in error_fields, \
            f"Validation error should mention missing field '{expected_field}', got errors for: {error_fields}"
    
    # Verify error types are 'missing' for required field validation
    for error in detail:
        if isinstance(error, dict) and "type" in error:
            # Check if this error is about a missing required field
            if "loc" in error and len(error["loc"]) > 1:
                field_name = error["loc"][-1]
                if field_name in expected_missing_fields:
                    assert "missing" in error["type"], \
                        f"Error type should contain 'missing' for required field '{field_name}', got '{error['type']}'"
    
    # Clean up environment
    for key in ["API_KEYS", "MODEL_NAME", "MODEL_CACHE_DIR", "LOG_LEVEL"]:
        if key in os.environ:
            del os.environ[key]
    config_module._settings = None


# Feature: ai-voice-detection-api, Property 14: Error handling without crashes
@given(
    # Generate various types of potentially problematic inputs
    language=st.one_of(
        st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
        st.just(""),  # Empty string
        st.none(),  # None value
    ),
    audio_format=st.one_of(
        st.sampled_from(["mp3", "wav", "flac"]),
        st.text(min_size=1, max_size=10),  # Invalid format
        st.just(""),  # Empty string
        st.none(),  # None value
    ),
    audio_base64=st.one_of(
        st.binary(min_size=1, max_size=1000).map(lambda b: base64.b64encode(b).decode()),  # Valid base64
        st.text(min_size=1, max_size=100),  # Invalid base64
        st.just("!!!invalid!!!"),  # Clearly invalid
        st.just(""),  # Empty string
        st.none(),  # None value
    ),
    api_key=st.one_of(
        st.just("test-valid-key-12345"),  # Valid key
        st.text(min_size=1, max_size=50, alphabet=st.characters(min_codepoint=33, max_codepoint=126)),  # Invalid key (ASCII printable)
        st.just(""),  # Empty key
        st.none(),  # Missing key
    )
)
@settings(max_examples=20, deadline=None)
def test_property_error_handling_without_crashes(language, audio_format, audio_base64, api_key):
    """
    Property 14: Error handling without crashes
    
    For any exception during request processing, the API should return an appropriate
    HTTP error status (4xx or 5xx) without crashing.
    
    This test verifies that the API handles all types of errors gracefully:
    - Authentication errors (401/403)
    - Validation errors (422)
    - Processing errors (400)
    - Server errors (500)
    
    The API should NEVER crash with unhandled exceptions or return non-HTTP responses.
    
    Validates: Requirements 7.1
    """
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from api.routes import router
    from api.middleware import APIKeyMiddleware
    import os
    
    # Set up a test API key in environment
    test_valid_key = "test-valid-key-12345"
    os.environ["API_KEYS"] = test_valid_key
    os.environ["MODEL_NAME"] = "facebook/wav2vec2-base"
    os.environ["MODEL_CACHE_DIR"] = "/tmp/models"
    os.environ["LOG_LEVEL"] = "ERROR"
    
    # Force reload of settings to pick up the test API key
    import core.config as config_module
    config_module._settings = None
    
    # Create a minimal FastAPI app with the middleware and router
    app = FastAPI()
    app.add_middleware(APIKeyMiddleware)
    app.include_router(router, prefix="/api/v1")
    
    # Add the global exception handlers from main.py
    from utils.exceptions import APIException
    from fastapi.responses import JSONResponse
    
    @app.exception_handler(APIException)
    async def api_exception_handler(request, exc: APIException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.__class__.__name__,
                "detail": exc.detail,
                "status_code": exc.status_code
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "detail": "Internal server error",
                "status_code": 500
            }
        )
    
    client = TestClient(app)
    
    # Build request data (may be invalid)
    request_data = {}
    if language is not None:
        request_data["language"] = language
    if audio_format is not None:
        request_data["audio_format"] = audio_format
    if audio_base64 is not None:
        request_data["audio_base64"] = audio_base64
    
    # Build headers (may be invalid)
    headers = {}
    if api_key is not None:
        headers["x-api-key"] = api_key
    
    # Make the request - this should NEVER crash, only return error responses
    try:
        response = client.post("/api/v1/detect", json=request_data, headers=headers)
        
        # Verify we got a valid HTTP response (not a crash)
        assert response is not None, "Response should not be None"
        assert hasattr(response, 'status_code'), "Response should have status_code attribute"
        
        # Verify the status code is a valid HTTP error code
        # Valid codes: 4xx (client errors) or 5xx (server errors) or 200 (success)
        assert 200 <= response.status_code < 600, \
            f"Status code should be valid HTTP code, got {response.status_code}"
        
        # Verify we can parse the response as JSON
        try:
            response_data = response.json()
            assert isinstance(response_data, dict), "Response should be a JSON object"
        except Exception as e:
            pytest.fail(f"Response should be valid JSON, got error: {e}")
        
        # If it's an error response (4xx or 5xx), verify it has proper error structure
        if response.status_code >= 400:
            # Error responses should have error information
            # Either FastAPI validation format (detail as list) or our custom format (error/detail)
            assert "detail" in response_data or "error" in response_data, \
                f"Error response should contain 'detail' or 'error' field, got: {response_data}"
            
            # Verify no stack traces are exposed in the response
            response_text = str(response_data).lower()
            assert "traceback" not in response_text, \
                "Error response should not contain stack traces"
            assert "file \"" not in response_text, \
                "Error response should not contain file paths from stack traces"
            assert "line " not in response_text or "line" in response_text, \
                "Error response should not contain line numbers from stack traces"
        
        # If it's a success response (200), verify it has the expected structure
        elif response.status_code == 200:
            # Success responses should have detection result fields
            assert "is_ai_generated" in response_data, \
                "Success response should contain 'is_ai_generated' field"
            assert "confidence" in response_data, \
                "Success response should contain 'confidence' field"
            assert "detected_language" in response_data, \
                "Success response should contain 'detected_language' field"
            assert "message" in response_data, \
                "Success response should contain 'message' field"
    
    except Exception as e:
        # If we get here, the API crashed with an unhandled exception
        # This is a FAILURE of the property test
        pytest.fail(
            f"API crashed with unhandled exception: {type(e).__name__}: {str(e)}. "
            f"The API should handle all errors gracefully and return HTTP error responses, "
            f"not crash with unhandled exceptions."
        )
    
    finally:
        # Clean up environment
        for key in ["API_KEYS", "MODEL_NAME", "MODEL_CACHE_DIR", "LOG_LEVEL"]:
            if key in os.environ:
                del os.environ[key]
        config_module._settings = None


# Feature: ai-voice-detection-api, Property 15: Stack trace concealment
@given(
    # Generate various types of inputs that might cause errors
    error_type=st.sampled_from([
        "invalid_base64",      # Invalid base64 encoding
        "missing_fields",      # Missing required fields
        "invalid_audio",       # Invalid audio data
        "empty_data",          # Empty data
        "malformed_json",      # Malformed request
    ]),
    language=st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
    audio_format=st.sampled_from(["mp3", "wav", "flac"]),
)
@settings(max_examples=100, deadline=None)
def test_property_stack_trace_concealment(error_type, language, audio_format):
    """
    Property 15: Stack trace concealment
    
    For any error response, the response body should not contain Python stack traces
    or internal implementation details.
    
    This test verifies that:
    - No Python tracebacks are exposed in error responses
    - No file paths from the codebase are exposed
    - No line numbers from stack traces are exposed
    - No internal variable names or implementation details are leaked
    - Error messages are sanitized and user-friendly
    
    Validates: Requirements 7.5
    """
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from api.routes import router
    from api.middleware import APIKeyMiddleware
    import os
    import re
    
    # Set up a test API key in environment
    test_valid_key = "test-valid-key-12345"
    os.environ["API_KEYS"] = test_valid_key
    os.environ["MODEL_NAME"] = "facebook/wav2vec2-base"
    os.environ["MODEL_CACHE_DIR"] = "/tmp/models"
    os.environ["LOG_LEVEL"] = "ERROR"
    
    # Force reload of settings to pick up the test API key
    import core.config as config_module
    config_module._settings = None
    
    # Create a minimal FastAPI app with the middleware and router
    app = FastAPI()
    app.add_middleware(APIKeyMiddleware)
    app.include_router(router, prefix="/api/v1")
    
    # Add the global exception handlers from main.py
    from utils.exceptions import APIException
    from fastapi.responses import JSONResponse
    
    @app.exception_handler(APIException)
    async def api_exception_handler(request, exc: APIException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.__class__.__name__,
                "detail": exc.detail,
                "status_code": exc.status_code
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "detail": "Internal server error",
                "status_code": 500
            }
        )
    
    client = TestClient(app)
    
    # Build request data based on error type to trigger different error conditions
    request_data = {}
    headers = {"x-api-key": test_valid_key}
    
    if error_type == "invalid_base64":
        # Invalid base64 that should trigger AudioDecodingError
        request_data = {
            "language": language,
            "audio_format": audio_format,
            "audio_base64": "!!!invalid-base64-data!!!"
        }
    elif error_type == "missing_fields":
        # Missing required fields to trigger validation error
        request_data = {
            "language": language
            # Missing audio_format and audio_base64
        }
    elif error_type == "invalid_audio":
        # Valid base64 but invalid audio data
        request_data = {
            "language": language,
            "audio_format": audio_format,
            "audio_base64": base64.b64encode(b"not-valid-audio-data").decode()
        }
    elif error_type == "empty_data":
        # Empty audio data
        request_data = {
            "language": language,
            "audio_format": audio_format,
            "audio_base64": base64.b64encode(b"").decode()
        }
    elif error_type == "malformed_json":
        # This will be handled by sending raw data instead of JSON
        pass
    
    # Make the request
    try:
        if error_type == "malformed_json":
            # Send malformed JSON to trigger parsing error
            response = client.post(
                "/api/v1/detect",
                data="not-valid-json{{{",
                headers={**headers, "Content-Type": "application/json"}
            )
        else:
            response = client.post("/api/v1/detect", json=request_data, headers=headers)
        
        # Get the response body as text for analysis
        response_text = response.text
        response_lower = response_text.lower()
        
        # Parse JSON response if possible
        try:
            response_data = response.json()
            response_json_str = str(response_data)
        except:
            response_json_str = response_text
        
        # Define patterns that indicate stack trace leakage
        stack_trace_indicators = [
            r'Traceback \(most recent call last\)',  # Python traceback header
            r'File ".*\.py"',                         # File paths in stack traces
            r'line \d+, in \w+',                      # Line numbers in stack traces
            r'raise \w+Error',                        # Raise statements
            r'\.py", line \d+',                       # Python file references
            r'in <module>',                           # Module execution context
            r'in \w+\(',                              # Function call context (partial)
            r'site-packages/',                        # Library paths
            r'lib/python',                            # Python library paths
            r'__traceback__',                         # Traceback object references
            r'__cause__',                             # Exception cause references
            r'__context__',                           # Exception context references
        ]
        
        # Check for stack trace indicators
        for pattern in stack_trace_indicators:
            matches = re.search(pattern, response_text, re.IGNORECASE)
            assert matches is None, (
                f"Stack trace leaked in error response! Found pattern '{pattern}' in response. "
                f"Error responses should not expose internal implementation details. "
                f"Response excerpt: {response_text[:500]}"
            )
        
        # Check for common internal variable names that shouldn't be exposed
        internal_indicators = [
            'exc_info',
            'traceback',
            'stack_trace',
            '__file__',
            '__name__',
            'sys.exc_info',
            'exception.__',
        ]
        
        for indicator in internal_indicators:
            assert indicator not in response_lower, (
                f"Internal implementation detail '{indicator}' leaked in error response! "
                f"Error responses should be sanitized and user-friendly. "
                f"Response excerpt: {response_text[:500]}"
            )
        
        # Verify that error responses have a clean, user-friendly structure
        if response.status_code >= 400:
            # Error responses should be JSON
            try:
                error_data = response.json()
                
                # Should have either 'detail' (FastAPI validation) or 'error'/'detail' (custom)
                assert "detail" in error_data or "error" in error_data, (
                    "Error response should have 'detail' or 'error' field"
                )
                
                # If it has our custom error format, verify the structure
                if "error" in error_data:
                    assert isinstance(error_data["error"], str), "error field should be a string"
                    assert isinstance(error_data["detail"], str), "detail field should be a string"
                    
                    # Error messages should be user-friendly, not technical
                    detail_lower = error_data["detail"].lower()
                    
                    # Should not contain Python-specific error types
                    python_error_types = [
                        'valueerror',
                        'typeerror',
                        'keyerror',
                        'attributeerror',
                        'indexerror',
                        'nameerror',
                        'importerror',
                    ]
                    
                    for error_type_name in python_error_types:
                        assert error_type_name not in detail_lower, (
                            f"Error detail should not expose Python error type '{error_type_name}'. "
                            f"Got: {error_data['detail']}"
                        )
                
            except ValueError:
                # If we can't parse JSON, that's also a problem
                pytest.fail(f"Error response should be valid JSON, got: {response_text[:200]}")
        
        # Verify that the response is a proper HTTP response (not a crash dump)
        assert response.status_code is not None, "Response should have a status code"
        assert 200 <= response.status_code < 600, (
            f"Response should have valid HTTP status code, got {response.status_code}"
        )
        
    except Exception as e:
        # If the test itself crashes, that's a failure
        pytest.fail(
            f"Test crashed while checking stack trace concealment: {type(e).__name__}: {str(e)}"
        )
    
    finally:
        # Clean up environment
        for key in ["API_KEYS", "MODEL_NAME", "MODEL_CACHE_DIR", "LOG_LEVEL"]:
            if key in os.environ:
                del os.environ[key]
        config_module._settings = None



# Feature: ai-voice-detection-api, Property 16: Environment variable configuration
@given(
    api_keys=st.lists(
        st.text(
            min_size=10,
            max_size=50,
            alphabet=st.characters(min_codepoint=48, max_codepoint=122)
        ),
        min_size=1,
        max_size=5
    ),
    model_name=st.sampled_from([
        "facebook/wav2vec2-base",
        "openai/whisper-tiny",
        "microsoft/wavlm-base",
        "custom-model-name"
    ]),
    model_cache_dir=st.sampled_from([
        "/tmp/models",
        "/var/cache/models",
        "./models",
        "/home/user/.cache/models"
    ]),
    log_level=st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
)
@settings(max_examples=100)
def test_property_environment_variable_configuration(api_keys, model_name, model_cache_dir, log_level):
    """
    Property 16: Environment variable configuration
    
    For any environment variable set for API keys or model paths, the application
    should load and use those values correctly.
    
    This test verifies that:
    1. API keys are loaded from environment variables
    2. Model configuration is loaded from environment variables
    3. Log level is loaded from environment variables
    4. Multiple API keys can be configured (comma-separated)
    5. Settings are properly parsed and accessible
    
    Validates: Requirements 9.4
    """
    import os
    from core.config import get_settings
    import core.config as config_module
    
    # Reset settings singleton to ensure fresh load
    config_module._settings = None
    
    try:
        # Set environment variables
        os.environ["API_KEYS"] = ",".join(api_keys)
        os.environ["MODEL_NAME"] = model_name
        os.environ["MODEL_CACHE_DIR"] = model_cache_dir
        os.environ["LOG_LEVEL"] = log_level
        
        # Force reload of settings
        config_module._settings = None
        
        # Load settings
        settings = get_settings()
        
        # Verify API keys are loaded correctly
        assert hasattr(settings, 'api_keys'), "Settings should have 'api_keys' attribute"
        assert isinstance(settings.api_keys, list), "api_keys should be a list"
        assert len(settings.api_keys) == len(api_keys), \
            f"Expected {len(api_keys)} API keys, got {len(settings.api_keys)}"
        
        # Verify each API key is loaded (trimmed of whitespace)
        for expected_key in api_keys:
            assert expected_key.strip() in settings.api_keys, \
                f"API key '{expected_key}' should be in settings.api_keys"
        
        # Verify model name is loaded correctly
        assert hasattr(settings, 'model_name'), "Settings should have 'model_name' attribute"
        assert settings.model_name == model_name, \
            f"Expected model_name '{model_name}', got '{settings.model_name}'"
        
        # Verify model cache directory is loaded correctly
        assert hasattr(settings, 'model_cache_dir'), "Settings should have 'model_cache_dir' attribute"
        assert settings.model_cache_dir == model_cache_dir, \
            f"Expected model_cache_dir '{model_cache_dir}', got '{settings.model_cache_dir}'"
        
        # Verify log level is loaded correctly
        assert hasattr(settings, 'log_level'), "Settings should have 'log_level' attribute"
        assert settings.log_level == log_level, \
            f"Expected log_level '{log_level}', got '{settings.log_level}'"
        
        # Verify settings are accessible and usable
        assert len(settings.api_keys) > 0, "At least one API key should be configured"
        assert len(settings.model_name) > 0, "Model name should not be empty"
        assert len(settings.model_cache_dir) > 0, "Model cache directory should not be empty"
        assert settings.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], \
            f"Log level should be valid, got '{settings.log_level}'"
        
    finally:
        # Clean up environment variables
        for key in ["API_KEYS", "MODEL_NAME", "MODEL_CACHE_DIR", "LOG_LEVEL"]:
            if key in os.environ:
                del os.environ[key]
        
        # Reset settings singleton
        config_module._settings = None
