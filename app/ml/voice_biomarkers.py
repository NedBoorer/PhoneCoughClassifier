"""
Voice Health Platform - OpenSMILE Voice Biomarker Extraction
Extracts eGeMAPS features for voice health analysis
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VoiceBiomarkers:
    """Collection of voice biomarkers extracted from audio"""
    # Frequency/Pitch
    f0_mean: float = 0.0  # Fundamental frequency mean
    f0_std: float = 0.0   # Pitch variability
    f0_range: float = 0.0  # Pitch range
    
    # Perturbation
    jitter: float = 0.0   # Cycle-to-cycle pitch variation
    shimmer: float = 0.0  # Cycle-to-cycle amplitude variation
    
    # Noise
    hnr: float = 0.0      # Harmonics-to-noise ratio
    
    # Energy
    energy_mean: float = 0.0
    energy_std: float = 0.0
    
    # Temporal
    speaking_rate: float = 0.0  # Syllables/onsets per second
    pause_ratio: float = 0.0    # Ratio of silence
    
    # Spectral
    spectral_centroid: float = 0.0
    spectral_flux: float = 0.0


class VoiceBiomarkerExtractor:
    """
    Extract voice biomarkers using librosa (OpenSMILE fallback if available).
    These biomarkers are used for:
    - Parkinson's detection (jitter, shimmer, HNR)
    - Depression screening (pitch variability, energy, speaking rate)
    - General voice health assessment
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._opensmile = None
        self._use_opensmile = False
        
        # Try to import opensmile
        try:
            import opensmile
            self._opensmile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals
            )
            self._use_opensmile = True
            logger.info("✓ OpenSMILE available for feature extraction")
        except ImportError:
            logger.info("OpenSMILE not installed, using librosa for biomarkers")
    
    def extract_with_librosa(self, audio_path: str) -> VoiceBiomarkers:
        """Extract voice biomarkers using librosa"""
        import librosa
        import scipy.stats
        
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        biomarkers = VoiceBiomarkers()
        
        # --- Fundamental Frequency (F0) ---
        pitches = librosa.yin(y, fmin=50, fmax=500)
        pitches = pitches[~np.isnan(pitches)]
        
        if len(pitches) > 0:
            biomarkers.f0_mean = float(np.mean(pitches))
            biomarkers.f0_std = float(np.std(pitches))
            biomarkers.f0_range = float(np.max(pitches) - np.min(pitches))
            
            # Jitter (pitch perturbation quotient)
            if len(pitches) > 1:
                period_diff = np.abs(np.diff(pitches))
                biomarkers.jitter = float(np.mean(period_diff) / (biomarkers.f0_mean + 1e-8))
        
        # --- Shimmer (amplitude perturbation) ---
        rms = librosa.feature.rms(y=y)[0]
        if len(rms) > 1:
            amp_diff = np.abs(np.diff(rms))
            biomarkers.shimmer = float(np.mean(amp_diff) / (np.mean(rms) + 1e-8))
        
        # --- Harmonics-to-Noise Ratio ---
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        biomarkers.hnr = float(
            np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + 1e-8)
        )
        
        # --- Energy ---
        biomarkers.energy_mean = float(np.mean(rms))
        biomarkers.energy_std = float(np.std(rms))
        
        # --- Speaking Rate (onsets per second) ---
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        duration = len(y) / sr
        biomarkers.speaking_rate = float(len(onsets) / duration) if duration > 0 else 0
        
        # --- Pause Ratio ---
        silence_threshold = 0.01
        is_silent = rms < silence_threshold
        biomarkers.pause_ratio = float(np.mean(is_silent))
        
        # --- Spectral Features ---
        spectral_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        biomarkers.spectral_centroid = float(np.mean(spectral_cent))
        
        # Spectral flux
        spec = np.abs(librosa.stft(y))
        flux = np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0))
        biomarkers.spectral_flux = float(np.mean(flux))
        
        return biomarkers
    
    def extract_with_opensmile(self, audio_path: str) -> VoiceBiomarkers:
        """Extract voice biomarkers using OpenSMILE eGeMAPS"""
        features = self._opensmile.process_file(audio_path)
        
        # Map eGeMAPS features to our biomarkers
        biomarkers = VoiceBiomarkers()
        
        # F0 features
        if "F0semitoneFrom27.5Hz_sma3nz_amean" in features.columns:
            biomarkers.f0_mean = float(features["F0semitoneFrom27.5Hz_sma3nz_amean"].iloc[0])
        if "F0semitoneFrom27.5Hz_sma3nz_stddevNorm" in features.columns:
            biomarkers.f0_std = float(features["F0semitoneFrom27.5Hz_sma3nz_stddevNorm"].iloc[0])
        
        # Jitter
        if "jitterLocal_sma3nz_amean" in features.columns:
            biomarkers.jitter = float(features["jitterLocal_sma3nz_amean"].iloc[0])
        
        # Shimmer  
        if "shimmerLocaldB_sma3nz_amean" in features.columns:
            biomarkers.shimmer = float(features["shimmerLocaldB_sma3nz_amean"].iloc[0])
        
        # HNR
        if "HNRdBACF_sma3nz_amean" in features.columns:
            biomarkers.hnr = float(features["HNRdBACF_sma3nz_amean"].iloc[0])
        
        # Loudness (energy proxy)
        if "loudness_sma3_amean" in features.columns:
            biomarkers.energy_mean = float(features["loudness_sma3_amean"].iloc[0])
        if "loudness_sma3_stddevNorm" in features.columns:
            biomarkers.energy_std = float(features["loudness_sma3_stddevNorm"].iloc[0])
        
        return biomarkers
    
    def extract(self, audio_path: str) -> VoiceBiomarkers:
        """Extract voice biomarkers using best available method"""
        if self._use_opensmile:
            try:
                return self.extract_with_opensmile(audio_path)
            except Exception as e:
                logger.warning(f"OpenSMILE failed, falling back to librosa: {e}")
        
        return self.extract_with_librosa(audio_path)
    
    def to_dict(self, biomarkers: VoiceBiomarkers) -> Dict[str, float]:
        """Convert biomarkers to dictionary"""
        return {
            "f0_mean": biomarkers.f0_mean,
            "f0_std": biomarkers.f0_std,
            "f0_range": biomarkers.f0_range,
            "jitter": biomarkers.jitter,
            "shimmer": biomarkers.shimmer,
            "hnr": biomarkers.hnr,
            "energy_mean": biomarkers.energy_mean,
            "energy_std": biomarkers.energy_std,
            "speaking_rate": biomarkers.speaking_rate,
            "pause_ratio": biomarkers.pause_ratio,
            "spectral_centroid": biomarkers.spectral_centroid,
            "spectral_flux": biomarkers.spectral_flux,
        }


# Singleton instance
_extractor = None


def get_biomarker_extractor() -> VoiceBiomarkerExtractor:
    """Get singleton biomarker extractor"""
    global _extractor
    if _extractor is None:
        _extractor = VoiceBiomarkerExtractor()
    return _extractor
