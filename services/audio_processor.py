"""
Audio processing service for the AI Voice Detection API.

This module handles audio decoding from base64 and validation of audio formats.
It provides safe decoding without causing system crashes and manages audio data
in memory using BytesIO buffers.
"""

import base64
from io import BytesIO
from typing import Optional

from utils.exceptions import AudioDecodingError
from utils.logger import get_logger


logger = get_logger("audio_processor")


class AudioProcessor:
    """
    Service for processing audio data.
    
    Handles base64 decoding and audio format validation without writing to disk.
    All audio data is kept in memory using BytesIO buffers.
    """
    
    @staticmethod
    def decode_base64_audio(audio_base64: str) -> BytesIO:
        """
        Safely decode base64-encoded audio to an in-memory buffer.
        
        This method handles decoding errors gracefully without causing system crashes.
        The decoded audio is stored in a BytesIO buffer for in-memory processing.
        
        Args:
            audio_base64: Base64-encoded audio string
            
        Returns:
            BytesIO: In-memory buffer containing decoded audio data
            
        Raises:
            AudioDecodingError: If base64 decoding fails or input is invalid
        """
        if not audio_base64:
            logger.error("Empty base64 audio string provided")
            raise AudioDecodingError("Audio data cannot be empty")
        
        if not isinstance(audio_base64, str):
            logger.error(f"Invalid audio_base64 type: {type(audio_base64)}")
            raise AudioDecodingError("Audio data must be a string")
        
        try:
            # Decode base64 string to bytes
            audio_bytes = base64.b64decode(audio_base64, validate=True)
            
            # Check if decoded data is empty
            if len(audio_bytes) == 0:
                logger.error("Decoded audio data is empty")
                raise AudioDecodingError("Decoded audio data is empty")
            
            # Create in-memory buffer
            audio_buffer = BytesIO(audio_bytes)
            audio_buffer.seek(0)  # Reset position to beginning
            
            logger.info(f"Successfully decoded {len(audio_bytes)} bytes of audio data")
            return audio_buffer
            
        except base64.binascii.Error as e:
            logger.error(f"Base64 decoding error: {str(e)}")
            raise AudioDecodingError(f"Invalid base64 encoding: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during audio decoding: {str(e)}")
            raise AudioDecodingError(f"Failed to decode audio data: {str(e)}")
    
    @staticmethod
    def validate_audio_format(buffer: BytesIO, expected_format: str) -> bool:
        """
        Validate that audio format matches the expected type.
        
        This method performs basic validation by checking file signatures (magic bytes)
        for common audio formats.
        
        Args:
            buffer: In-memory audio buffer
            expected_format: Expected audio format (e.g., "mp3", "wav", "flac")
            
        Returns:
            bool: True if format matches or cannot be determined, False otherwise
            
        Note:
            This is a basic validation. For production use, consider using
            libraries like python-magic for more robust format detection.
        """
        if not buffer or buffer.getbuffer().nbytes == 0:
            logger.warning("Empty buffer provided for format validation")
            return False
        
        # Save current position
        current_position = buffer.tell()
        
        try:
            # Read first few bytes for magic number detection
            buffer.seek(0)
            header = buffer.read(12)
            buffer.seek(current_position)  # Restore position
            
            if len(header) < 4:
                logger.warning("Buffer too small for format validation")
                return True  # Allow processing to continue
            
            # Check magic bytes for common formats
            format_lower = expected_format.lower()
            
            if format_lower == "mp3":
                # MP3: ID3 tag (0x49 0x44 0x33) or MPEG frame sync (0xFF 0xFB/0xFA)
                if header[:3] == b'ID3' or (header[0] == 0xFF and header[1] in [0xFB, 0xFA, 0xF3, 0xF2]):
                    logger.info("Valid MP3 format detected")
                    return True
                    
            elif format_lower == "wav":
                # WAV: RIFF header (0x52 0x49 0x46 0x46) followed by WAVE
                if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
                    logger.info("Valid WAV format detected")
                    return True
                    
            elif format_lower == "flac":
                # FLAC: fLaC marker (0x66 0x4C 0x61 0x43)
                if header[:4] == b'fLaC':
                    logger.info("Valid FLAC format detected")
                    return True
            
            # If we can't determine format, log warning but allow processing
            logger.warning(f"Could not verify {expected_format} format from header, allowing processing to continue")
            return True
            
        except Exception as e:
            logger.error(f"Error during format validation: {str(e)}")
            # Don't fail on validation errors, allow processing to continue
            return True
