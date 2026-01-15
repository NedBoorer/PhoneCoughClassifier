"""
Phone Cough Classifier - Audio Feature Extraction
Extracts comprehensive acoustic features using librosa
"""
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Feature extraction constants
SAMPLE_RATE = 16000
N_MFCC = 20
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048


class FeatureExtractor:
    """Extract acoustic features from audio for cough classification"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._librosa = None
    
    @property
    def librosa(self):
        """Lazy load librosa"""
        if self._librosa is None:
            import librosa
            self._librosa = librosa
        return self._librosa
    
    def load_audio(self, audio_path: str, duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """Load audio file and resample to target sample rate"""
        try:
            y, sr = self.librosa.load(
                audio_path,
                sr=self.sample_rate,
                duration=duration,
                mono=True
            )
            return y, sr
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            raise
    
    def extract_mfcc(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract MFCC features (20 coefficients + deltas)"""
        mfcc = self.librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            n_mfcc=N_MFCC,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT
        )
        
        # Compute deltas
        mfcc_delta = self.librosa.feature.delta(mfcc)
        mfcc_delta2 = self.librosa.feature.delta(mfcc, order=2)
        
        return {
            "mfcc_mean": np.mean(mfcc, axis=1),
            "mfcc_std": np.std(mfcc, axis=1),
            "mfcc_delta_mean": np.mean(mfcc_delta, axis=1),
            "mfcc_delta_std": np.std(mfcc_delta, axis=1),
            "mfcc_delta2_mean": np.mean(mfcc_delta2, axis=1),
            "mfcc_delta2_std": np.std(mfcc_delta2, axis=1),
        }
    
    def extract_spectral(self, y: np.ndarray) -> Dict[str, float]:
        """Extract spectral features"""
        # Spectral centroid
        centroid = self.librosa.feature.spectral_centroid(
            y=y, sr=self.sample_rate, hop_length=HOP_LENGTH
        )
        
        # Spectral bandwidth
        bandwidth = self.librosa.feature.spectral_bandwidth(
            y=y, sr=self.sample_rate, hop_length=HOP_LENGTH
        )
        
        # Spectral rolloff
        rolloff = self.librosa.feature.spectral_rolloff(
            y=y, sr=self.sample_rate, hop_length=HOP_LENGTH
        )
        
        # Spectral contrast
        contrast = self.librosa.feature.spectral_contrast(
            y=y, sr=self.sample_rate, hop_length=HOP_LENGTH
        )
        
        # Spectral flatness
        flatness = self.librosa.feature.spectral_flatness(
            y=y, hop_length=HOP_LENGTH
        )
        
        return {
            "spectral_centroid_mean": float(np.mean(centroid)),
            "spectral_centroid_std": float(np.std(centroid)),
            "spectral_bandwidth_mean": float(np.mean(bandwidth)),
            "spectral_bandwidth_std": float(np.std(bandwidth)),
            "spectral_rolloff_mean": float(np.mean(rolloff)),
            "spectral_rolloff_std": float(np.std(rolloff)),
            "spectral_contrast_mean": float(np.mean(contrast)),
            "spectral_contrast_std": float(np.std(contrast)),
            "spectral_flatness_mean": float(np.mean(flatness)),
            "spectral_flatness_std": float(np.std(flatness)),
        }
    
    def extract_temporal(self, y: np.ndarray) -> Dict[str, float]:
        """Extract temporal features"""
        # Zero crossing rate
        zcr = self.librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
        
        # RMS energy
        rms = self.librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
        
        # Duration
        duration = len(y) / self.sample_rate
        
        return {
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "rms_max": float(np.max(rms)),
            "duration": duration,
        }
    
    def extract_rhythm(self, y: np.ndarray) -> Dict[str, float]:
        """Extract rhythm and tempo features"""
        # Onset strength
        onset_env = self.librosa.onset.onset_strength(
            y=y, sr=self.sample_rate, hop_length=HOP_LENGTH
        )
        
        # Tempo
        tempo, _ = self.librosa.beat.beat_track(
            onset_envelope=onset_env, sr=self.sample_rate
        )
        
        # Number of onsets (cough events)
        onsets = self.librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=self.sample_rate
        )
        
        return {
            "tempo": float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo[0]),
            "onset_strength_mean": float(np.mean(onset_env)),
            "onset_strength_std": float(np.std(onset_env)),
            "num_onsets": len(onsets),
        }
    
    def extract_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram (for HeAR model input)"""
        mel_spec = self.librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT
        )
        
        # Convert to log scale
        mel_spec_db = self.librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def extract_all_features(self, audio_path: str) -> Dict:
        """
        Extract all features from audio file.
        Returns a dictionary with all features and the mel spectrogram.
        """
        try:
            # Load audio
            y, sr = self.load_audio(audio_path)
            
            if len(y) < self.sample_rate * 0.3:  # Less than 0.3 seconds
                raise ValueError("Audio too short for analysis")
            
            # Extract all feature groups
            features = {}
            
            # MFCC features
            mfcc_features = self.extract_mfcc(y)
            for key, value in mfcc_features.items():
                if isinstance(value, np.ndarray):
                    for i, v in enumerate(value):
                        features[f"{key}_{i}"] = float(v)
                else:
                    features[key] = float(value)
            
            # Spectral features
            features.update(self.extract_spectral(y))
            
            # Temporal features
            features.update(self.extract_temporal(y))
            
            # Rhythm features
            features.update(self.extract_rhythm(y))
            
            # Mel spectrogram (stored separately for HeAR)
            mel_spec = self.extract_mel_spectrogram(y)
            
            return {
                "features": features,
                "mel_spectrogram": mel_spec,
                "sample_rate": sr,
                "duration": len(y) / sr,
                "num_samples": len(y),
            }
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {audio_path}: {e}")
            raise
    
    def get_feature_vector(self, audio_path: str) -> np.ndarray:
        """Get flat feature vector for sklearn classifiers"""
        result = self.extract_all_features(audio_path)
        features = result["features"]
        
        # Convert to sorted list for consistent ordering
        feature_names = sorted(features.keys())
        feature_vector = np.array([features[name] for name in feature_names])
        
        return feature_vector, feature_names


# Singleton instance
_extractor = None


def get_feature_extractor() -> FeatureExtractor:
    """Get singleton feature extractor instance"""
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractor()
    return _extractor
