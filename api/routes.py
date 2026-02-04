"""
API routes for the AI Voice Detection API.

This module defines the detection endpoint that processes audio files
and returns AI voice detection results.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Annotated

from api.schemas import DetectionRequest, DetectionResponse
from services.audio_processor import AudioProcessor
from services.feature_extractor import FeatureExtractor
from services.model_service import ModelService
from core.config import get_settings
from utils.exceptions import (
    APIException,
    AudioDecodingError,
    FeatureExtractionError,
    ModelInferenceError
)
from utils.logger import get_logger


logger = get_logger("routes")

# Create router
router = APIRouter(tags=["detection"])


# Dependency injection functions
def get_audio_processor() -> AudioProcessor:
    """Dependency for AudioProcessor."""
    return AudioProcessor()


def get_feature_extractor() -> FeatureExtractor:
    """Dependency for FeatureExtractor."""
    return FeatureExtractor(sample_rate=16000)


def get_model_service() -> ModelService:
    """
    Dependency for ModelService.
    
    Loads the model on first request and reuses it for subsequent requests.
    """
    settings = get_settings()
    service = ModelService(
        model_name=settings.model_name,
        cache_dir=settings.model_cache_dir
    )
    # Load model if not already loaded
    if service.model is None:
        service.load_model()
    return service


@router.post("/detect", response_model=DetectionResponse)
async def detect_voice(
    request: DetectionRequest,
    audio_processor: Annotated[AudioProcessor, Depends(get_audio_processor)],
    feature_extractor: Annotated[FeatureExtractor, Depends(get_feature_extractor)],
    model_service: Annotated[ModelService, Depends(get_model_service)]
) -> DetectionResponse:
    """
    Detect whether audio contains AI-generated or human voice.
    
    This endpoint processes base64-encoded audio data through the following pipeline:
    1. Decode base64 audio to binary
    2. Extract audio features using librosa
    3. Predict AI generation using pretrained model
    4. Return detection results with confidence score
    
    Args:
        request: Detection request containing language, audio format, and base64 audio
        audio_processor: Audio processing service (injected)
        feature_extractor: Feature extraction service (injected)
        model_service: Model inference service (injected)
    
    Returns:
        DetectionResponse: Detection results including:
            - is_ai_generated: Boolean classification
            - confidence: Confidence score (0.0 to 1.0)
            - detected_language: Detected or provided language
            - message: Human-readable result description
    
    Raises:
        HTTPException: 400 for audio processing errors, 500 for model errors
    """
    logger.info(f"Received detection request for language={request.language}, format={request.audio_format}")
    
    try:
        # Step 1: Decode base64 audio
        logger.debug("Decoding base64 audio data")
        audio_buffer = audio_processor.decode_base64_audio(request.audio_base64)
        
        # Step 2: Extract audio features
        logger.debug("Extracting audio features")
        features = feature_extractor.extract_features(audio_buffer)
        
        # Step 3: Run model prediction
        logger.debug("Running model prediction")
        is_ai_generated, confidence = model_service.predict(features)
        
        # Step 4: Detect language (for MVP, returns input language)
        detected_language = model_service.detect_language(audio_buffer, request.language)
        
        # Step 5: Build response message
        if is_ai_generated:
            if confidence >= 0.8:
                message = "Voice detected as AI-generated with high confidence"
            elif confidence >= 0.6:
                message = "Voice detected as AI-generated with moderate confidence"
            else:
                message = "Voice detected as AI-generated with low confidence"
        else:
            if confidence <= 0.2:
                message = "Voice detected as human with high confidence"
            elif confidence <= 0.4:
                message = "Voice detected as human with moderate confidence"
            else:
                message = "Voice detected as human with low confidence"
        
        # Build response
        response = DetectionResponse(
            is_ai_generated=is_ai_generated,
            confidence=confidence,
            detected_language=detected_language,
            message=message
        )
        
        logger.info(f"Detection complete: is_ai_generated={is_ai_generated}, confidence={confidence:.3f}")
        return response
        
    except AudioDecodingError as e:
        logger.error(f"Audio decoding error: {e.detail}")
        raise HTTPException(
            status_code=e.status_code,
            detail=e.detail
        )
    
    except FeatureExtractionError as e:
        logger.error(f"Feature extraction error: {e.detail}")
        raise HTTPException(
            status_code=e.status_code,
            detail=e.detail
        )
    
    except ModelInferenceError as e:
        logger.error(f"Model inference error: {e.detail}")
        raise HTTPException(
            status_code=e.status_code,
            detail=e.detail
        )
    
    except APIException as e:
        logger.error(f"API exception: {e.detail}")
        raise HTTPException(
            status_code=e.status_code,
            detail=e.detail
        )
    
    except Exception as e:
        logger.error(f"Unexpected error during detection: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
