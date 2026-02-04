"""
Unit tests for configuration management.

Tests environment variable loading and validation for the Settings class.
Requirements: 9.4
"""

import pytest
from pydantic import ValidationError
from core.config import Settings, get_settings
import os


class TestConfigurationLoading:
    """Test suite for configuration loading from environment variables."""
    
    def test_load_all_environment_variables(self, monkeypatch):
        """
        Test that all environment variables are loaded correctly.
        
        Validates that Settings class properly loads and parses all
        configuration values from environment variables.
        """
        # Set all environment variables
        monkeypatch.setenv("API_KEYS", "test-key-1,test-key-2,test-key-3")
        monkeypatch.setenv("MODEL_NAME", "custom/model-name")
        monkeypatch.setenv("MODEL_CACHE_DIR", "/custom/cache/dir")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("PORT", "9000")
        monkeypatch.setenv("HOST", "127.0.0.1")
        
        # Load settings
        settings = Settings()
        
        # Verify all values are loaded correctly
        assert settings.api_keys == ["test-key-1", "test-key-2", "test-key-3"]
        assert settings.model_name == "custom/model-name"
        assert settings.model_cache_dir == "/custom/cache/dir"
        assert settings.log_level == "DEBUG"
        assert settings.port == 9000
        assert settings.host == "127.0.0.1"
    
    def test_load_with_default_values(self, monkeypatch):
        """
        Test that default values are used when optional variables are not set.
        
        Only API_KEYS is required; other fields should use defaults.
        """
        # Set only required variable
        monkeypatch.setenv("API_KEYS", "required-key")
        
        # Clear optional variables to ensure defaults are used
        monkeypatch.delenv("MODEL_NAME", raising=False)
        monkeypatch.delenv("MODEL_CACHE_DIR", raising=False)
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        monkeypatch.delenv("PORT", raising=False)
        monkeypatch.delenv("HOST", raising=False)
        
        # Load settings
        settings = Settings()
        
        # Verify required field is loaded
        assert settings.api_keys == ["required-key"]
        
        # Verify default values are used
        assert settings.model_name == "facebook/wav2vec2-base"
        assert settings.model_cache_dir == "/tmp/models"
        assert settings.log_level == "INFO"
        assert settings.port == 8000
        assert settings.host == "0.0.0.0"
    
    def test_api_keys_parsing_single_key(self, monkeypatch):
        """Test that a single API key is parsed correctly."""
        monkeypatch.setenv("API_KEYS", "single-key")
        
        settings = Settings()
        
        assert settings.api_keys == ["single-key"]
    
    def test_api_keys_parsing_multiple_keys(self, monkeypatch):
        """Test that multiple comma-separated API keys are parsed correctly."""
        monkeypatch.setenv("API_KEYS", "key1,key2,key3,key4")
        
        settings = Settings()
        
        assert settings.api_keys == ["key1", "key2", "key3", "key4"]
    
    def test_api_keys_parsing_with_whitespace(self, monkeypatch):
        """Test that API keys with surrounding whitespace are trimmed."""
        monkeypatch.setenv("API_KEYS", "  key1  ,  key2  ,  key3  ")
        
        settings = Settings()
        
        assert settings.api_keys == ["key1", "key2", "key3"]
    
    def test_log_level_case_insensitive(self, monkeypatch):
        """Test that log level is converted to uppercase."""
        monkeypatch.setenv("API_KEYS", "test-key")
        monkeypatch.setenv("LOG_LEVEL", "debug")
        
        settings = Settings()
        
        assert settings.log_level == "DEBUG"


class TestMissingRequiredVariables:
    """Test suite for missing required environment variables."""
    
    def test_missing_api_keys_raises_error(self, monkeypatch):
        """
        Test that missing API_KEYS raises a validation error.
        
        API_KEYS is a required field and must be provided.
        """
        # Ensure API_KEYS is not set
        monkeypatch.delenv("API_KEYS", raising=False)
        
        # Attempt to load settings should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        # Verify the error is about the missing field
        errors = exc_info.value.errors()
        # Check if any error is about the api_keys field being required
        assert any(
            "api_keys" in str(error.get("loc", "")).lower() or 
            error.get("type") == "missing" or
            "required" in str(error.get("msg", "")).lower()
            for error in errors
        )
    
    def test_empty_api_keys_raises_error(self, monkeypatch):
        """
        Test that empty API_KEYS string raises a validation error.
        
        API_KEYS must contain at least one valid key.
        """
        monkeypatch.setenv("API_KEYS", "")
        
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        # Verify the error message
        errors = exc_info.value.errors()
        assert any("API_KEYS is required and cannot be empty" in str(error) for error in errors)
    
    def test_whitespace_only_api_keys_raises_error(self, monkeypatch):
        """
        Test that API_KEYS with only whitespace raises a validation error.
        
        After trimming, there must be at least one valid key.
        """
        monkeypatch.setenv("API_KEYS", "   ,   ,   ")
        
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        # Verify the error message
        errors = exc_info.value.errors()
        assert any("must contain at least one valid key" in str(error) for error in errors)
    
    def test_invalid_log_level_raises_error(self, monkeypatch):
        """
        Test that invalid LOG_LEVEL raises a validation error.
        
        LOG_LEVEL must be one of the standard Python logging levels.
        """
        monkeypatch.setenv("API_KEYS", "test-key")
        monkeypatch.setenv("LOG_LEVEL", "INVALID")
        
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        # Verify the error is about invalid log level
        errors = exc_info.value.errors()
        assert any("LOG_LEVEL must be one of" in str(error) for error in errors)
    
    def test_invalid_port_raises_error(self, monkeypatch):
        """
        Test that invalid PORT number raises a validation error.
        
        PORT must be between 1 and 65535.
        """
        monkeypatch.setenv("API_KEYS", "test-key")
        monkeypatch.setenv("PORT", "70000")
        
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        # Verify the error is about invalid port
        errors = exc_info.value.errors()
        assert any("PORT must be between 1 and 65535" in str(error) for error in errors)
    
    def test_negative_port_raises_error(self, monkeypatch):
        """Test that negative PORT number raises a validation error."""
        monkeypatch.setenv("API_KEYS", "test-key")
        monkeypatch.setenv("PORT", "-1")
        
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        errors = exc_info.value.errors()
        assert any("PORT must be between 1 and 65535" in str(error) for error in errors)


class TestGetSettingsSingleton:
    """Test suite for the get_settings singleton function."""
    
    def test_get_settings_returns_settings_instance(self, monkeypatch):
        """Test that get_settings returns a Settings instance."""
        monkeypatch.setenv("API_KEYS", "test-key")
        
        # Reset the global settings instance
        import core.config
        core.config._settings = None
        
        settings = get_settings()
        
        assert isinstance(settings, Settings)
        assert settings.api_keys == ["test-key"]
    
    def test_get_settings_returns_same_instance(self, monkeypatch):
        """
        Test that get_settings returns the same instance on multiple calls.
        
        This validates the singleton pattern implementation.
        """
        monkeypatch.setenv("API_KEYS", "test-key")
        
        # Reset the global settings instance
        import core.config
        core.config._settings = None
        
        settings1 = get_settings()
        settings2 = get_settings()
        
        # Verify both calls return the exact same object
        assert settings1 is settings2
