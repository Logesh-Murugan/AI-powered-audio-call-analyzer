"""
API request and response schemas using Pydantic models.
"""
import base64
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class DetectionRequest(BaseModel):
    """Request schema for voice detection endpoint."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "language": "en",
                "audio_format": "mp3",
                "audio_base64": "SGVsbG8gV29ybGQ="
            }
        }
    )
    
    language: str = Field(
        ...,
        description="Expected language of the audio",
        json_schema_extra={"example": "en"}
    )
    audio_format: str = Field(
        ...,
        description="Audio file format",
        pattern="^(mp3|wav|flac)$",
        json_schema_extra={"example": "mp3"}
    )
    audio_base64: str = Field(
        ...,
        description="Base64-encoded audio data",
        min_length=1
    )
    
    @field_validator('audio_base64')
    @classmethod
    def validate_base64(cls, v):
        """Validates base64 encoding format."""
        try:
            base64.b64decode(v, validate=True)
            return v
        except Exception:
            raise ValueError("Invalid base64 encoding")


class DetectionResponse(BaseModel):
    """Response schema for voice detection results."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "is_ai_generated": True,
                "confidence": 0.87,
                "detected_language": "en",
                "message": "Voice detected as AI-generated with high confidence"
            }
        }
    )
    
    is_ai_generated: bool = Field(
        ...,
        description="Whether the voice is AI-generated"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the prediction"
    )
    detected_language: str = Field(
        ...,
        description="Detected or provided language"
    )
    message: str = Field(
        ...,
        description="Human-readable result description"
    )
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        """Validates that message contains non-whitespace characters."""
        if not v or not v.strip():
            raise ValueError("Message must contain non-whitespace characters")
        return v


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "AudioDecodingError",
                "detail": "Failed to decode audio data",
                "status_code": 400
            }
        }
    )
    
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    status_code: int = Field(..., description="HTTP status code")
