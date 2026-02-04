"""
Unit tests for FeatureExtractor.

These tests focus on valid audio samples and error handling for invalid audio,
covering edge cases and error conditions for feature extraction.
"""
import numpy as np
import pytest
from io import BytesIO
import wave
import struct

from services.feature_extractor import FeatureExtractor
from utils.exceptions import FeatureExtractionError


class TestFeatureExtractorValidAudio:
    """Test feature extraction with valid audio samples"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.extractor = FeatureExtractor(sample_rate=16000)
    
    def _create_synthetic_audio(self, duration: float = 1.0, frequency: int = 440, sample_rate: int = 16000) -> BytesIO:
        """
        Create a synthetic audio WAV file in memory.
        
        Args:
            duration: Duration in seconds
            frequency: Frequency in Hz
            sample_rate: Sample rate in Hz
            
        Returns:
            BytesIO: In-memory WAV file
        """
        # Generate sine wave
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples, False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create WAV file in memory
        buffer = BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        buffer.seek(0)
        return buffer
    
    def test_extract_features_from_valid_audio(self):
        """Test that features are successfully extracted from valid audio"""
        audio_buffer = self._create_synthetic_audio(duration=1.0, frequency=440)
        
        features = self.extractor.extract_features(audio_buffer)
        
        # Verify features are returned as numpy array
        assert isinstance(features, np.ndarray)
        
        # Verify feature vector has expected shape (54 features total)
        # 13 MFCC mean + 13 MFCC std + 1 spectral centroid mean + 1 spectral centroid std
        # + 1 zcr mean + 1 zcr std + 12 chroma mean + 12 chroma std = 54
        assert features.shape == (54,)
        
        # Verify features contain valid numeric values (not NaN or Inf)
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()
    
    def test_extract_features_different_frequencies(self):
        """Test feature extraction with different audio frequencies"""
        frequencies = [220, 440, 880]  # Different musical notes
        
        for freq in frequencies:
            audio_buffer = self._create_synthetic_audio(duration=0.5, frequency=freq)
            features = self.extractor.extract_features(audio_buffer)
            
            assert isinstance(features, np.ndarray)
            assert features.shape == (54,)
            assert not np.isnan(features).any()
    
    def test_extract_features_different_durations(self):
        """Test feature extraction with different audio durations"""
        durations = [0.5, 1.0, 2.0]
        
        for duration in durations:
            audio_buffer = self._create_synthetic_audio(duration=duration, frequency=440)
            features = self.extractor.extract_features(audio_buffer)
            
            # Feature vector size should be consistent regardless of duration
            assert features.shape == (54,)
            assert not np.isnan(features).any()
    
    def test_extract_features_preserves_buffer(self):
        """Test that feature extraction doesn't corrupt the buffer"""
        audio_buffer = self._create_synthetic_audio(duration=1.0, frequency=440)
        original_data = audio_buffer.getvalue()
        
        self.extractor.extract_features(audio_buffer)
        
        # Verify buffer data is unchanged
        audio_buffer.seek(0)
        assert audio_buffer.getvalue() == original_data
    
    def test_extract_features_multiple_calls(self):
        """Test that multiple extractions from same buffer produce consistent results"""
        audio_buffer = self._create_synthetic_audio(duration=1.0, frequency=440)
        
        features1 = self.extractor.extract_features(audio_buffer)
        features2 = self.extractor.extract_features(audio_buffer)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(features1, features2)


class TestFeatureExtractorInvalidAudio:
    """Test error handling for invalid audio"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.extractor = FeatureExtractor(sample_rate=16000)
    
    def test_empty_buffer_raises_error(self):
        """Test that empty buffer raises FeatureExtractionError"""
        empty_buffer = BytesIO(b"")
        
        with pytest.raises(FeatureExtractionError) as exc_info:
            self.extractor.extract_features(empty_buffer)
        
        # Empty buffer causes librosa to fail with format not recognised
        assert "failed" in str(exc_info.value.detail).lower()
    
    def test_none_buffer_raises_error(self):
        """Test that None buffer raises FeatureExtractionError"""
        with pytest.raises(FeatureExtractionError) as exc_info:
            self.extractor.extract_features(None)
        
        assert "empty" in str(exc_info.value.detail).lower()
    
    def test_corrupted_audio_data_raises_error(self):
        """Test that corrupted audio data raises FeatureExtractionError"""
        # Create buffer with random bytes that aren't valid audio
        corrupted_buffer = BytesIO(b"This is not valid audio data at all!")
        
        with pytest.raises(FeatureExtractionError) as exc_info:
            self.extractor.extract_features(corrupted_buffer)
        
        assert "failed" in str(exc_info.value.detail).lower()
    
    def test_invalid_wav_header_raises_error(self):
        """Test that invalid WAV header raises FeatureExtractionError"""
        # Create buffer with partial/invalid WAV header
        invalid_wav = BytesIO(b"RIFF\x00\x00\x00\x00WAVE")
        
        with pytest.raises(FeatureExtractionError) as exc_info:
            self.extractor.extract_features(invalid_wav)
        
        assert "failed" in str(exc_info.value.detail).lower()
    
    def test_truncated_audio_file_raises_error(self):
        """Test that truncated audio file raises FeatureExtractionError"""
        # Create a valid WAV header but truncate the data
        buffer = BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            # Write minimal data
            wav_file.writeframes(b"\x00\x00")
        
        # Truncate the buffer
        truncated_data = buffer.getvalue()[:20]
        truncated_buffer = BytesIO(truncated_data)
        
        with pytest.raises(FeatureExtractionError) as exc_info:
            self.extractor.extract_features(truncated_buffer)
        
        assert "failed" in str(exc_info.value.detail).lower()
    
    def test_non_audio_binary_data_raises_error(self):
        """Test that non-audio binary data raises FeatureExtractionError"""
        # Create buffer with binary data that's not audio
        non_audio_buffer = BytesIO(bytes(range(256)))
        
        with pytest.raises(FeatureExtractionError) as exc_info:
            self.extractor.extract_features(non_audio_buffer)
        
        assert "failed" in str(exc_info.value.detail).lower()


class TestFeatureExtractorConfiguration:
    """Test feature extractor configuration and initialization"""
    
    def test_default_sample_rate(self):
        """Test that default sample rate is 16000 Hz"""
        extractor = FeatureExtractor()
        assert extractor.sample_rate == 16000
    
    def test_custom_sample_rate(self):
        """Test that custom sample rate is set correctly"""
        custom_rate = 22050
        extractor = FeatureExtractor(sample_rate=custom_rate)
        assert extractor.sample_rate == custom_rate
    
    def test_different_sample_rates_produce_valid_features(self):
        """Test that different sample rates still produce valid features"""
        sample_rates = [8000, 16000, 22050, 44100]
        
        for rate in sample_rates:
            extractor = FeatureExtractor(sample_rate=rate)
            
            # Create audio at the target sample rate
            num_samples = rate  # 1 second of audio
            t = np.linspace(0, 1, num_samples, False)
            audio_data = np.sin(2 * np.pi * 440 * t)
            audio_data = (audio_data * 32767).astype(np.int16)
            
            buffer = BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(rate)
                wav_file.writeframes(audio_data.tobytes())
            
            buffer.seek(0)
            features = extractor.extract_features(buffer)
            
            # Verify features are valid
            assert features.shape == (54,)
            assert not np.isnan(features).any()
