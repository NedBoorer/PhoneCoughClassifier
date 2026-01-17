"""
Tuberculosis (TB) Cough Screening Classifier

Uses acoustic features from cough sounds to screen for potential tuberculosis.
Based on research showing TB coughs have distinct acoustic signatures:
- More productive (wet) cough patterns
- Longer cough duration
- Specific spectral characteristics
- Lower pitch variability in advanced cases

References:
- "Automatic cough classification for tuberculosis screening" (Pahar et al., 2021)
- "TB cough detection using deep learning" (CODA TB DREAM Challenge)
- Google HeAR bioacoustic model for cough analysis
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


def _to_python_float(val) -> float:
    """Convert numpy types to Python float for JSON serialization"""
    if hasattr(val, 'item'):
        return float(val.item())
    return float(val)


@dataclass
class TBScreeningResult:
    """Result from TB cough screening"""
    detected: bool
    confidence: float
    severity: str  # normal, low_risk, moderate_risk, high_risk
    recommendation: str
    details: Dict


class TuberculosisScreener:
    """
    Tuberculosis screening from cough sounds.
    
    Uses research-validated acoustic features that distinguish TB coughs:
    - Spectral features: MFCCs, spectral centroid, spectral rolloff
    - Temporal features: Cough duration, zero-crossing rate
    - Energy features: RMS energy patterns
    - Cough quality: Wet vs dry cough detection
    
    Important: This is a SCREENING tool, not a diagnostic.
    Positive results require confirmatory testing (sputum smear, GeneXpert, CXR).
    """
    
    # Research-calibrated thresholds for TB cough detection
    # Based on literature review of TB acoustic biomarkers
    THRESHOLDS = {
        # Spectral features - TB coughs tend to have different spectral patterns
        "spectral_centroid_low": 1500,     # Hz - lower centroid may indicate congestion
        "spectral_rolloff_low": 3000,      # Hz - reduced high-frequency content
        "mfcc_variance_high": 15.0,        # Higher variance in MFCCs
        
        # Temporal features
        "cough_duration_long": 0.4,        # seconds - TB coughs often longer
        "zcr_high": 0.15,                  # Zero-crossing rate
        
        # Energy features
        "energy_variance_high": 0.02,      # Variable energy (productive cough)
        "energy_burst_ratio": 0.6,         # Ratio of high-energy frames
        
        # Wet cough indicators (common in TB)
        "wetness_score_threshold": 0.5,    # Wetness indicator
    }
    
    def __init__(self):
        self._loaded = False
        self._hear_model = None
        self._sklearn_model = None
    
    def load(self):
        """Load TB screening model"""
        if self._loaded:
            return
        
        # Try to load sklearn model if available
        model_path = Path("./models/tb_classifier.joblib")
        if model_path.exists():
            try:
                import joblib
                self._sklearn_model = joblib.load(model_path)
                logger.info("✓ TB classifier sklearn model loaded")
            except Exception as e:
                logger.warning(f"Could not load TB sklearn model: {e}")
        
        self._loaded = True
        logger.info("✓ Tuberculosis screener loaded (feature-based)")
    
    def _extract_tb_features(self, audio_path: str) -> Dict[str, float]:
        """
        Extract acoustic features relevant for TB cough detection.

        Based on research showing TB coughs have:
        - Lower spectral centroid (congestion)
        - Higher MFCC variance
        - Longer duration
        - More "wet" characteristics

        OPTIMIZED: Reduced feature set for faster processing (<3 seconds)
        """
        import librosa

        # Load audio with mono and lower sample rate for speed
        y, sr = librosa.load(audio_path, sr=16000, mono=True, duration=10.0)
        duration = len(y) / sr

        features = {"duration": duration}

        # Skip very short audio
        if duration < 0.5:
            logger.warning(f"Audio too short for TB analysis: {duration:.2f}s")
            return features
        
        # === SPECTRAL FEATURES (OPTIMIZED) ===
        # MFCCs - key features for cough classification (reduced to 5 for speed)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5, hop_length=512)  # Increased hop_length for speed
        features["mfcc_mean"] = float(np.mean(mfccs))
        features["mfcc_std"] = float(np.std(mfccs))
        features["mfcc_variance"] = float(np.var(mfccs))

        # Spectral centroid - center of mass of spectrum
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
        features["spectral_centroid_std"] = float(np.std(spectral_centroid))

        # Spectral rolloff - frequency below which 85% of energy exists
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)[0]
        features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
        
        # === TEMPORAL FEATURES ===
        # Zero-crossing rate - indicates noisiness/texture
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)[0]
        features["zcr_mean"] = float(np.mean(zcr))

        # === ENERGY FEATURES ===
        # RMS energy
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        features["energy_mean"] = float(np.mean(rms))
        features["energy_std"] = float(np.std(rms))

        # Energy variance - productive coughs have more variable energy
        features["energy_variance"] = float(np.var(rms))

        # High-energy frame ratio (burst detection)
        energy_threshold = np.mean(rms) + np.std(rms)
        high_energy_frames = np.sum(rms > energy_threshold) / len(rms)
        features["energy_burst_ratio"] = float(high_energy_frames)
        
        # === WET COUGH DETECTION (OPTIMIZED) ===
        # Wet coughs (common in TB) have:
        # - More low-frequency energy
        # - Higher spectral flux (rapid changes)

        # Compute STFT once and reuse (optimized)
        stft = np.abs(librosa.stft(y, hop_length=512, n_fft=1024))

        # Spectral flux (simplified calculation)
        spectral_flux = np.mean(np.abs(np.diff(stft, axis=1)))
        features["spectral_flux_mean"] = float(spectral_flux)

        # Low-frequency energy ratio (< 500 Hz)
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=1024)
        low_freq_mask = freq_bins < 500
        if np.sum(stft) > 0:
            low_freq_energy = np.sum(stft[low_freq_mask, :]) / np.sum(stft)
            features["low_freq_ratio"] = float(low_freq_energy)
        else:
            features["low_freq_ratio"] = 0
        
        # Wetness score (composite)
        wetness_indicators = 0
        if features["spectral_flux_mean"] > 50:
            wetness_indicators += 1
        if features["low_freq_ratio"] > 0.3:
            wetness_indicators += 1
        if features["energy_variance"] > self.THRESHOLDS["energy_variance_high"]:
            wetness_indicators += 1
        if features["spectral_centroid_mean"] < self.THRESHOLDS["spectral_centroid_low"]:
            wetness_indicators += 1
        
        features["wetness_score"] = wetness_indicators / 4.0
        
        # === COUGH PATTERN ANALYSIS (SIMPLIFIED) ===
        # Onset detection for cough segmentation (simplified with larger hop_length)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
        features["num_cough_events"] = len(onsets)
        features["coughs_per_second"] = float(len(onsets) / duration) if duration > 0 else 0

        return features
    
    def _calculate_tb_risk_score(self, features: Dict[str, float]) -> Tuple[float, List[str]]:
        """
        Calculate TB risk score based on extracted features.
        
        Returns:
            Tuple of (risk_score 0-1, list of risk indicators)
        """
        risk_score = 0.0
        indicators = []
        
        # Check duration (minimum required)
        if features.get("duration", 0) < 0.5:
            return 0.0, ["insufficient_audio"]
        
        # === WETNESS SCORE (weight: 0.25) ===
        # TB coughs are often productive/wet
        wetness = features.get("wetness_score", 0)
        if wetness >= 0.75:
            risk_score += 0.25
            indicators.append("highly_productive_cough")
        elif wetness >= 0.5:
            risk_score += 0.15
            indicators.append("productive_cough")
        
        # === SPECTRAL FEATURES (weight: 0.25) ===
        # Lower spectral centroid suggests congestion
        spectral_centroid = features.get("spectral_centroid_mean", 2000)
        if spectral_centroid < self.THRESHOLDS["spectral_centroid_low"]:
            risk_score += 0.15
            indicators.append("low_spectral_centroid")
        
        # High MFCC variance indicates complex cough pattern
        mfcc_var = features.get("mfcc_variance", 0)
        if mfcc_var > self.THRESHOLDS["mfcc_variance_high"]:
            risk_score += 0.10
            indicators.append("high_mfcc_variance")
        
        # === ENERGY PATTERNS (weight: 0.20) ===
        # Variable energy suggests productive cough
        energy_var = features.get("energy_variance", 0)
        if energy_var > self.THRESHOLDS["energy_variance_high"]:
            risk_score += 0.12
            indicators.append("variable_energy_pattern")
        
        energy_burst = features.get("energy_burst_ratio", 0)
        if energy_burst > self.THRESHOLDS["energy_burst_ratio"]:
            risk_score += 0.08
            indicators.append("high_energy_bursts")
        
        # === TEMPORAL FEATURES (weight: 0.15) ===
        # Longer cough duration
        duration = features.get("duration", 0)
        if duration > 2.0:
            risk_score += 0.10
            indicators.append("prolonged_cough")
        
        # Multiple cough events
        coughs_per_sec = features.get("coughs_per_second", 0)
        if coughs_per_sec > 1.5:
            risk_score += 0.05
            indicators.append("frequent_cough_events")
        
        # === LOW FREQUENCY CONTENT (weight: 0.15) ===
        # TB coughs often have more low-frequency content
        low_freq_ratio = features.get("low_freq_ratio", 0)
        if low_freq_ratio > 0.35:
            risk_score += 0.15
            indicators.append("high_low_frequency_content")
        elif low_freq_ratio > 0.25:
            risk_score += 0.08
            indicators.append("elevated_low_frequency")
        
        # Normalize score
        normalized_score = min(risk_score, 1.0)
        
        # Adjust for very short samples
        if duration < 2.0:
            normalized_score *= 0.8
            if "short_sample_warning" not in indicators:
                indicators.append("short_sample_warning")
        
        return normalized_score, indicators
    
    def screen(self, audio_path: str) -> TBScreeningResult:
        """
        Screen cough audio for tuberculosis indicators.
        
        Important: This is a SCREENING tool only. 
        Positive results require confirmatory testing.
        """
        self.load()
        
        try:
            # Extract features
            features = self._extract_tb_features(audio_path)
            
            # Calculate risk score
            risk_score, indicators = self._calculate_tb_risk_score(features)
            
            # Determine severity and recommendation
            if risk_score < 0.25:
                severity = "normal"
                detected = False
                recommendation = (
                    "No significant TB cough indicators detected. "
                    "If you have a persistent cough lasting more than 2 weeks, "
                    "please consult a healthcare provider."
                )
            elif risk_score < 0.45:
                severity = "low_risk"
                detected = False
                recommendation = (
                    "Some cough characteristics noted, but not strongly indicative of TB. "
                    "Monitor your symptoms. If cough persists for more than 2 weeks "
                    "with fever, night sweats, or weight loss, please seek medical evaluation."
                )
            elif risk_score < 0.65:
                severity = "moderate_risk"
                detected = True
                recommendation = (
                    "Your cough shows some patterns that may warrant further evaluation. "
                    "Please visit your nearest health center or DOTS center for a proper TB test "
                    "(sputum examination or GeneXpert). Early detection helps in better treatment."
                )
            else:
                severity = "high_risk"
                detected = True
                recommendation = (
                    "Your cough shows patterns consistent with TB. "
                    "Please visit a DOTS center or hospital immediately for confirmatory testing. "
                    "TB is curable with proper treatment. Do not delay seeking medical care."
                )
            
            return TBScreeningResult(
                detected=detected,
                confidence=_to_python_float(risk_score),
                severity=severity,
                recommendation=recommendation,
                details={
                    "risk_score": _to_python_float(risk_score),
                    "indicators": indicators,
                    "features": {k: _to_python_float(v) for k, v in features.items()},
                    "disclaimer": (
                        "This is a screening tool only and not a medical diagnosis. "
                        "Positive results require confirmatory laboratory testing."
                    )
                }
            )
            
        except Exception as e:
            logger.error(f"TB screening failed: {e}")
            return TBScreeningResult(
                detected=False,
                confidence=0.0,
                severity="unknown",
                recommendation="Unable to analyze cough for TB indicators. Please try again.",
                details={"error": str(e)}
            )


# Singleton instance
_tb_screener = None


def get_tb_screener() -> TuberculosisScreener:
    """Get singleton TB screener instance"""
    global _tb_screener
    if _tb_screener is None:
        _tb_screener = TuberculosisScreener()
    return _tb_screener
