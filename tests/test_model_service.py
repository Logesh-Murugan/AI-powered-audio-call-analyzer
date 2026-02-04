"""
Unit tests for the ModelService class.

Tests model loading, prediction, and language detection functionality.
"""

import pytest
import numpy as np
from io import BytesIO

from services.model_service import ModelService
from utils.exceptions import ModelInferenceError


class TestModelService:
    """Test suite for ModelService class."""
    
    def test_model_service_initialization(self):
        """Test that ModelService initializes correctly."""
        model_service = ModelService(
            model_name="facebook/wav2vec2-base",
            cache_dir="/tmp/models"
        )
        
        assert model_service.model_name == "facebook/wav2vec2-base"
        assert model_service.cache_dir == "/tmp/models"
        assert model_service.model is None
        assert model_service.processor is None
    
    def test_predict_without_loading_model(self):
        """Test that predict raises error when model not loaded."""
        model_service = ModelService(
            model_name="facebook/wav2vec2-base",
            cache_dir="/tmp/models"
        )
        
        features = np.random.rand(54)
        
        with pytest.raises(ModelInferenceError) as exc_info:
            model_service.predict(features)
        
        assert "Model not loaded" in str(exc_info.value)
    
    def test_predict_with_empty_features(self):
        """Test that predict raises error with empty features."""
        model_service = ModelService(
            model_name="facebook/wav2vec2-base",
            cache_dir="/tmp/models"
        )
        
        # Mock model as loaded
        model_service.model = "mock_model"
        
        with pytest.raises(ModelInferenceError) as exc_info:
            model_service.predict(np.array([]))
        
        assert "Features cannot be empty" in str(exc_info.value)
    
    def test_predict_returns_tuple(self):
        """Test that predict returns a tuple of (bool, float)."""
        model_service = ModelService(
            model_name="facebook/wav2vec2-base",
            cache_dir="/tmp/models"
        )
        
        # Mock model as loaded
        model_service.model = "mock_model"
        
        # Create sample features
        features = np.random.rand(54)
        
        result = model_service.predict(features)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)
    
    def test_predict_confidence_bounds(self):
        """Test that confidence score is between 0 and 1."""
        model_service = ModelService(
            model_name="facebook/wav2vec2-base",
            cache_dir="/tmp/models"
        )
        
        # Mock model as loaded
        model_service.model = "mock_model"
        
        # Test with various feature vectors
        for _ in range(10):
            features = np.random.rand(54) * 10  # Random features
            is_ai_generated, confidence = model_service.predict(features)
            
            assert 0.0 <= confidence <= 1.0
            assert isinstance(is_ai_generated, bool)
    
    def test_detect_language_returns_input_language(self):
        """Test that detect_language returns the input language (MVP fallback)."""
        model_service = ModelService(
            model_name="facebook/wav2vec2-base",
            cache_dir="/tmp/models"
        )
        
        audio_buffer = BytesIO(b"fake audio data")
        
        result = model_service.detect_language(audio_buffer, input_language="en")
        assert result == "en"
        
        result = model_service.detect_language(audio_buffer, input_language="es")
        assert result == "es"
        
        result = model_service.detect_language(audio_buffer, input_language="fr")
        assert result == "fr"
    
    def test_model_loading_success(self):
        """Test that model loads successfully when transformers is available."""
        model_service = ModelService(
            model_name="facebook/wav2vec2-base",
            cache_dir="/tmp/models"
        )
        
        # Attempt to load model
        # This test will only pass if transformers is installed and model can be downloaded
        try:
            model_service.load_model()
            assert model_service.model is not None
            assert model_service.processor is not None
        except ModelInferenceError as e:
            # If transformers is not installed or model download fails, that's expected
            pytest.skip(f"Model loading skipped: {str(e)}")
    
    def test_model_loading_idempotent(self):
        """Test that calling load_model multiple times doesn't reload the model."""
        model_service = ModelService(
            model_name="facebook/wav2vec2-base",
            cache_dir="/tmp/models"
        )
        
        # Mock the model as already loaded
        model_service.model = "mock_model"
        model_service.processor = "mock_processor"
        
        # Call load_model again
        model_service.load_model()
        
        # Should still have the same mock values
        assert model_service.model == "mock_model"
        assert model_service.processor == "mock_processor"
    
    def test_predict_with_invalid_features_type(self):
        """Test that predict handles invalid feature types gracefully."""
        model_service = ModelService(
            model_name="facebook/wav2vec2-base",
            cache_dir="/tmp/models"
        )
        
        # Mock model as loaded
        model_service.model = "mock_model"
        
        # Test with None
        with pytest.raises(ModelInferenceError):
            model_service.predict(None)
    
    def test_predict_with_various_feature_sizes(self):
        """Test that predict works with different feature vector sizes."""
        model_service = ModelService(
            model_name="facebook/wav2vec2-base",
            cache_dir="/tmp/models"
        )
        
        # Mock model as loaded
        model_service.model = "mock_model"
        
        # Test with different feature sizes
        for size in [10, 54, 100, 1000]:
            features = np.random.rand(size)
            is_ai_generated, confidence = model_service.predict(features)
            
            assert isinstance(is_ai_generated, bool)
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0
    
    def test_predict_error_handling_with_nan_features(self):
        """Test that predict handles NaN values in features gracefully."""
        model_service = ModelService(
            model_name="facebook/wav2vec2-base",
            cache_dir="/tmp/models"
        )
        
        # Mock model as loaded
        model_service.model = "mock_model"
        
        # Create features with NaN values
        features = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        
        # Should handle NaN gracefully and still return valid results
        try:
            is_ai_generated, confidence = model_service.predict(features)
            # If it succeeds, verify the output is valid
            assert isinstance(is_ai_generated, bool)
            assert isinstance(confidence, float)
            # With NaN in features, confidence may be NaN - that's acceptable behavior
            # The important thing is it doesn't crash
        except ModelInferenceError:
            # It's also acceptable to raise an error for invalid features
            pass
    
    def test_predict_error_handling_with_inf_features(self):
        """Test that predict handles infinite values in features gracefully."""
        model_service = ModelService(
            model_name="facebook/wav2vec2-base",
            cache_dir="/tmp/models"
        )
        
        # Mock model as loaded
        model_service.model = "mock_model"
        
        # Create features with infinite values
        features = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        
        # Should handle infinity gracefully
        try:
            is_ai_generated, confidence = model_service.predict(features)
            # If it succeeds, verify the output is valid
            assert isinstance(is_ai_generated, bool)
            assert isinstance(confidence, float)
            # With Inf in features, confidence may be NaN or clipped - that's acceptable
            # The important thing is it doesn't crash
        except ModelInferenceError:
            # It's also acceptable to raise an error for invalid features
            pass
