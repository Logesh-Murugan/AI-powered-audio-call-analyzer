"""
Feature extraction service for the AI Voice Detection API.

This module handles audio feature extraction using librosa. It extracts
various audio features including MFCC, spectral centroid, zero crossing rate,
and chroma features for use in AI voice detection models.
"""

import numpy as np
import librosa
from io import BytesIO
from typing import Dict, Any

from utils.exceptions import FeatureExtractionError
from utils.logger import get_logger


logger = get_logger("feature_extractor")


class FeatureExtractor:
    """
    Service for extracting audio features using librosa.
    
    Extracts multiple audio features suitable for AI voice detection:
    - MFCC (Mel-frequency cepstral coefficients)
    - Spectral centroid
    - Zero crossing rate
    - Chroma features
    
    Attributes:
        sample_rate: Target sample rate for audio processing (default: 16000 Hz)
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the feature extractor.
        
        Args:
            sample_rate: Target sample rate in Hz (default: 16000)
        """
        self.sample_rate = sample_rate
        logger.info(f"FeatureExtractor initialized with sample_rate={sample_rate}")
    
    def extract_features(self, audio_buffer: BytesIO) -> np.ndarray:
        """
        Extract audio features from an in-memory audio buffer.
        
        This method loads audio data using librosa and extracts multiple features:
        - MFCC: 13 coefficients capturing timbral characteristics
        - Spectral centroid: Center of mass of the spectrum
        - Zero crossing rate: Rate of sign changes in the signal
        - Chroma features: 12 pitch class features
        
        Args:
            audio_buffer: In-memory buffer containing audio data
            
        Returns:
            np.ndarray: Feature vector combining all extracted features
            
        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        if not audio_buffer:
            logger.error("Empty audio buffer provided")
            raise FeatureExtractionError("Audio buffer cannot be empty")
        
        try:
            # Reset buffer position to beginning
            audio_buffer.seek(0)
            
            # Load audio data using librosa
            # sr=self.sample_rate resamples to target rate
            # mono=True converts to mono if stereo
            logger.info("Loading audio data with librosa")
            y, sr = librosa.load(audio_buffer, sr=self.sample_rate, mono=True)
            
            # Check if audio data is valid
            if len(y) == 0:
                logger.error("Loaded audio data is empty")
                raise FeatureExtractionError("Audio data is empty after loading")
            
            logger.info(f"Audio loaded: {len(y)} samples at {sr} Hz")
            
            # Extract features
            features = self._extract_all_features(y, sr)
            
            logger.info(f"Successfully extracted feature vector of shape {features.shape}")
            return features
            
        except FeatureExtractionError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error during feature extraction: {str(e)}")
            raise FeatureExtractionError(f"Failed to extract features: {str(e)}")
    
    def _extract_all_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract all audio features from loaded audio data.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            np.ndarray: Combined feature vector
            
        Raises:
            FeatureExtractionError: If any feature extraction fails
        """
        try:
            features_dict = {}
            
            # Extract MFCC (Mel-frequency cepstral coefficients)
            # n_mfcc=13 is standard for speech/audio analysis
            logger.debug("Extracting MFCC features")
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            # Take mean across time axis
            features_dict['mfcc_mean'] = np.mean(mfcc, axis=1)
            features_dict['mfcc_std'] = np.std(mfcc, axis=1)
            
            # Extract spectral centroid
            # Indicates where the "center of mass" of the spectrum is
            logger.debug("Extracting spectral centroid")
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features_dict['spectral_centroid_mean'] = np.mean(spectral_centroid)
            features_dict['spectral_centroid_std'] = np.std(spectral_centroid)
            
            # Extract zero crossing rate
            # Rate at which signal changes from positive to negative or vice versa
            logger.debug("Extracting zero crossing rate")
            zcr = librosa.feature.zero_crossing_rate(y)
            features_dict['zcr_mean'] = np.mean(zcr)
            features_dict['zcr_std'] = np.std(zcr)
            
            # Extract chroma features
            # Represents the 12 different pitch classes
            logger.debug("Extracting chroma features")
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features_dict['chroma_mean'] = np.mean(chroma, axis=1)
            features_dict['chroma_std'] = np.std(chroma, axis=1)
            
            # Combine all features into a single vector
            feature_vector = self._combine_features(features_dict)
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error extracting individual features: {str(e)}")
            raise FeatureExtractionError(f"Feature extraction failed: {str(e)}")
    
    def _combine_features(self, features_dict: Dict[str, Any]) -> np.ndarray:
        """
        Combine all extracted features into a single feature vector.
        
        Args:
            features_dict: Dictionary of extracted features
            
        Returns:
            np.ndarray: Combined feature vector
        """
        feature_list = []
        
        # MFCC mean and std (13 + 13 = 26 features)
        feature_list.append(features_dict['mfcc_mean'])
        feature_list.append(features_dict['mfcc_std'])
        
        # Spectral centroid mean and std (2 features)
        feature_list.append(np.array([features_dict['spectral_centroid_mean']]))
        feature_list.append(np.array([features_dict['spectral_centroid_std']]))
        
        # Zero crossing rate mean and std (2 features)
        feature_list.append(np.array([features_dict['zcr_mean']]))
        feature_list.append(np.array([features_dict['zcr_std']]))
        
        # Chroma mean and std (12 + 12 = 24 features)
        feature_list.append(features_dict['chroma_mean'])
        feature_list.append(features_dict['chroma_std'])
        
        # Concatenate all features (total: 26 + 2 + 2 + 24 = 54 features)
        combined = np.concatenate(feature_list)
        
        logger.debug(f"Combined feature vector shape: {combined.shape}")
        return combined
