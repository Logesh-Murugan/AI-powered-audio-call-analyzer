"""
Configuration management for the AI Voice Detection API.

This module handles loading and validating environment variables
using Pydantic Settings for type-safe configuration management.
"""

from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings are loaded from environment variables or .env file.
    Required settings will raise validation errors if not provided.
    """
    
    # API Configuration
    api_keys: str = Field(
        ...,
        description="Comma-separated list of valid API keys for authentication",
        alias="API_KEYS"
    )
    
    # Model Configuration
    model_name: str = Field(
        default="facebook/wav2vec2-base",
        description="HuggingFace model name for voice detection",
        alias="MODEL_NAME"
    )
    
    model_cache_dir: str = Field(
        default="/tmp/models",
        description="Directory to cache downloaded models",
        alias="MODEL_CACHE_DIR"
    )
    
    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        alias="LOG_LEVEL"
    )
    
    # Server Configuration (optional, used for local development)
    port: int = Field(
        default=8000,
        description="Server port number",
        alias="PORT"
    )
    
    host: str = Field(
        default="0.0.0.0",
        description="Server host address",
        alias="HOST"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @field_validator("api_keys", mode="after")
    @classmethod
    def parse_api_keys(cls, v) -> List[str]:
        """
        Parse comma-separated API keys from environment variable.
        
        Args:
            v: Raw API_KEYS value (string)
            
        Returns:
            List[str]: List of API keys
            
        Raises:
            ValueError: If API_KEYS is empty or invalid
        """
        if v is None or v == "":
            raise ValueError("API_KEYS is required and cannot be empty")
        
        if isinstance(v, str):
            keys = [key.strip() for key in v.split(",") if key.strip()]
            if not keys:
                raise ValueError("API_KEYS must contain at least one valid key")
            return keys
        elif isinstance(v, list):
            if not v:
                raise ValueError("API_KEYS must contain at least one valid key")
            return v
        else:
            raise ValueError(f"API_KEYS must be a string or list, got {type(v)}")
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """
        Validate that log level is one of the standard Python logging levels.
        
        Args:
            v: Log level string
            
        Returns:
            str: Uppercase log level
            
        Raises:
            ValueError: If log level is invalid
        """
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(
                f"LOG_LEVEL must be one of {valid_levels}, got '{v}'"
            )
        return v_upper
    
    @field_validator("port")
    @classmethod
    def validate_port(cls, v):
        """
        Validate that port number is in valid range.
        
        Args:
            v: Port number
            
        Returns:
            int: Validated port number
            
        Raises:
            ValueError: If port is out of valid range
        """
        if not (1 <= v <= 65535):
            raise ValueError(f"PORT must be between 1 and 65535, got {v}")
        return v


# Global settings instance
# This will be initialized once and reused throughout the application
_settings: Settings | None = None


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    This function implements a singleton pattern to ensure settings
    are loaded only once during application startup.
    
    Returns:
        Settings: The global settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
