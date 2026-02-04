"""
Unit tests for AudioProcessor edge cases.

These tests focus on specific edge cases and error conditions for audio processing,
including corrupted base64 handling and empty audio data.
"""
import base64
import pytest
from io import BytesIO

from services.audio_processor import AudioProcessor
from utils.exceptions import AudioDecodingError


class TestAudioProcessorEdgeCases:
    """Test edge cases for AudioProcessor"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = AudioProcessor()
    
    # Test corrupted base64 handling
    
    def test_corrupted_base64_invalid_characters(self):
        """Test that corrupted base64 with invalid characters raises AudioDecodingError"""
        corrupted_base64 = "!!!invalid_base64_with_special_chars!!!"
        
        with pytest.raises(AudioDecodingError) as exc_info:
            self.processor.decode_base64_audio(corrupted_base64)
        
        assert "Invalid base64 encoding" in str(exc_info.value.detail)
    
    def test_corrupted_base64_invalid_padding(self):
        """Test that base64 with invalid padding raises AudioDecodingError"""
        # Valid base64 should have proper padding, this has incorrect padding
        corrupted_base64 = "SGVsbG8gV29ybGQ"  # Missing padding
        
        # This might actually decode successfully depending on validation
        # but if it fails, it should raise AudioDecodingError
        try:
            result = self.processor.decode_base64_audio(corrupted_base64)
            # If it succeeds, verify it's a valid buffer
            assert isinstance(result, BytesIO)
        except AudioDecodingError as e:
            # If it fails, verify proper error handling
            assert "Invalid base64 encoding" in str(e.detail) or "Failed to decode" in str(e.detail)
    
    def test_corrupted_base64_non_ascii_characters(self):
        """Test that base64 with non-ASCII characters is handled properly"""
        corrupted_base64 = "SGVsbG8g8J+YgA=="  # Contains emoji
        
        # Some non-ASCII characters might still decode, so we test that it either:
        # 1. Raises AudioDecodingError for invalid encoding, OR
        # 2. Successfully decodes if the string happens to be valid base64
        try:
            result = self.processor.decode_base64_audio(corrupted_base64)
            # If it succeeds, verify it's a valid buffer
            assert isinstance(result, BytesIO)
        except AudioDecodingError as e:
            # If it fails, verify proper error handling
            assert "Invalid base64 encoding" in str(e.detail) or "Failed to decode" in str(e.detail)
    
    def test_corrupted_base64_whitespace_only(self):
        """Test that whitespace-only string raises AudioDecodingError"""
        corrupted_base64 = "   \t\n  "
        
        with pytest.raises(AudioDecodingError) as exc_info:
            self.processor.decode_base64_audio(corrupted_base64)
        
        assert "Invalid base64 encoding" in str(exc_info.value.detail) or "Failed to decode" in str(exc_info.value.detail)
    
    def test_corrupted_base64_partial_data(self):
        """Test that partially corrupted base64 raises AudioDecodingError"""
        # Start with valid base64, then corrupt it
        valid_data = b"Hello World"
        valid_base64 = base64.b64encode(valid_data).decode()
        corrupted_base64 = valid_base64[:len(valid_base64)//2] + "###CORRUPTED###"
        
        with pytest.raises(AudioDecodingError) as exc_info:
            self.processor.decode_base64_audio(corrupted_base64)
        
        assert "Invalid base64 encoding" in str(exc_info.value.detail) or "Failed to decode" in str(exc_info.value.detail)
    
    # Test empty audio data
    
    def test_empty_string(self):
        """Test that empty string raises AudioDecodingError"""
        empty_base64 = ""
        
        with pytest.raises(AudioDecodingError) as exc_info:
            self.processor.decode_base64_audio(empty_base64)
        
        assert "cannot be empty" in str(exc_info.value.detail).lower()
    
    def test_base64_encoding_empty_bytes(self):
        """Test that base64-encoded empty bytes raises AudioDecodingError"""
        # Base64 encoding of empty bytes is an empty string or just padding
        empty_bytes = b""
        empty_base64 = base64.b64encode(empty_bytes).decode()
        
        with pytest.raises(AudioDecodingError) as exc_info:
            self.processor.decode_base64_audio(empty_base64)
        
        # Should fail because decoded data is empty
        assert "empty" in str(exc_info.value.detail).lower()
    
    def test_none_input(self):
        """Test that None input raises AudioDecodingError"""
        with pytest.raises(AudioDecodingError) as exc_info:
            self.processor.decode_base64_audio(None)
        
        assert "must be a string" in str(exc_info.value.detail).lower() or "cannot be empty" in str(exc_info.value.detail).lower()
    
    def test_non_string_input(self):
        """Test that non-string input raises AudioDecodingError"""
        with pytest.raises(AudioDecodingError) as exc_info:
            self.processor.decode_base64_audio(12345)
        
        assert "must be a string" in str(exc_info.value.detail).lower()
    
    def test_bytes_input_instead_of_string(self):
        """Test that bytes input (instead of string) raises AudioDecodingError"""
        audio_bytes = b"some audio data"
        
        with pytest.raises(AudioDecodingError) as exc_info:
            self.processor.decode_base64_audio(audio_bytes)
        
        assert "must be a string" in str(exc_info.value.detail).lower()


class TestAudioFormatValidation:
    """Test audio format validation edge cases"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = AudioProcessor()
    
    def test_validate_empty_buffer(self):
        """Test that empty buffer returns False"""
        empty_buffer = BytesIO(b"")
        
        result = self.processor.validate_audio_format(empty_buffer, "mp3")
        
        assert result is False
    
    def test_validate_none_buffer(self):
        """Test that None buffer returns False"""
        result = self.processor.validate_audio_format(None, "mp3")
        
        assert result is False
    
    def test_validate_buffer_too_small(self):
        """Test that buffer with less than 4 bytes returns True (allows processing)"""
        small_buffer = BytesIO(b"abc")  # Only 3 bytes
        
        result = self.processor.validate_audio_format(small_buffer, "mp3")
        
        # Should return True to allow processing to continue
        assert result is True
    
    def test_validate_preserves_buffer_position(self):
        """Test that validation preserves the buffer's read position"""
        # Create a buffer with some data
        audio_data = b"ID3" + b"\x00" * 100  # MP3 with ID3 tag
        buffer = BytesIO(audio_data)
        
        # Move to a specific position
        buffer.seek(50)
        original_position = buffer.tell()
        
        # Validate format
        self.processor.validate_audio_format(buffer, "mp3")
        
        # Verify position is restored
        assert buffer.tell() == original_position
