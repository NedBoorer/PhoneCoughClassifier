"""
Phone Cough Classifier - Audio Quality Assessment
Evaluates recording quality and provides feedback
"""
import logging
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioQualityResult:
    """Audio quality assessment result"""
    overall_score: float  # 0-100
    snr_db: float  # Signal-to-noise ratio in dB
    has_clipping: bool
    silence_ratio: float  # Ratio of silence
    is_acceptable: bool
    issues: List[str]
    recommendations: List[str]


class AudioQualityChecker:
    """Assess audio quality for cough classification"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._librosa = None
    
    @property
    def librosa(self):
        if self._librosa is None:
            import librosa
            self._librosa = librosa
        return self._librosa
    
    def estimate_snr(self, y: np.ndarray) -> float:
        """
        Estimate signal-to-noise ratio.
        Uses percentile-based noise floor estimation.
        """
        # Compute short-time energy
        hop_length = 512
        rms = self.librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Noise floor: 10th percentile
        noise_floor = np.percentile(rms, 10)
        
        # Signal level: 90th percentile
        signal_level = np.percentile(rms, 90)
        
        # Compute SNR
        if noise_floor > 0:
            snr = 20 * np.log10(signal_level / noise_floor)
        else:
            snr = 60.0  # Very clean
        
        return float(snr)
    
    def check_clipping(self, y: np.ndarray, threshold: float = 0.99) -> bool:
        """Check for audio clipping (samples at max level)"""
        max_val = np.max(np.abs(y))
        
        # Count samples near max
        clipping_samples = np.sum(np.abs(y) > threshold)
        clipping_ratio = clipping_samples / len(y)
        
        return clipping_ratio > 0.001 or max_val >= 1.0
    
    def compute_silence_ratio(self, y: np.ndarray, threshold_db: float = -40) -> float:
        """Compute ratio of silence in audio"""
        hop_length = 512
        rms = self.librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_db = 20 * np.log10(rms + 1e-10)
        
        silence_frames = np.sum(rms_db < threshold_db)
        silence_ratio = silence_frames / len(rms)
        
        return float(silence_ratio)
    
    def compute_overall_score(
        self,
        snr_db: float,
        has_clipping: bool,
        silence_ratio: float
    ) -> float:
        """Compute overall quality score (0-100)"""
        score = 100.0
        
        # SNR penalty
        if snr_db < 10:
            score -= 40
        elif snr_db < 20:
            score -= 25
        elif snr_db < 30:
            score -= 10
        
        # Clipping penalty
        if has_clipping:
            score -= 20
        
        # Silence penalty
        if silence_ratio > 0.8:
            score -= 40
        elif silence_ratio > 0.6:
            score -= 20
        elif silence_ratio > 0.4:
            score -= 10
        
        return max(0, min(100, score))
    
    def assess_quality(self, audio_path: str) -> AudioQualityResult:
        """
        Perform full quality assessment on audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            AudioQualityResult with scores and recommendations
        """
        try:
            # Load audio
            y, sr = self.librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Run assessments
            snr_db = self.estimate_snr(y)
            has_clipping = self.check_clipping(y)
            silence_ratio = self.compute_silence_ratio(y)
            
            # Compute overall score
            overall_score = self.compute_overall_score(snr_db, has_clipping, silence_ratio)
            
            # Determine if acceptable
            is_acceptable = overall_score >= 50
            
            # Generate issues and recommendations
            issues = []
            recommendations = []
            
            if snr_db < 20:
                issues.append("High background noise")
                recommendations.append("Find a quieter location")
            
            if has_clipping:
                issues.append("Audio clipping detected")
                recommendations.append("Speak further from the phone")
            
            if silence_ratio > 0.6:
                issues.append("Too much silence")
                recommendations.append("Cough closer to the microphone")
            
            duration = len(y) / sr
            if duration < 0.5:
                issues.append("Recording too short")
                recommendations.append("Record for at least 2 seconds")
            
            if not issues:
                recommendations.append("Audio quality is good")
            
            return AudioQualityResult(
                overall_score=overall_score,
                snr_db=snr_db,
                has_clipping=has_clipping,
                silence_ratio=silence_ratio,
                is_acceptable=is_acceptable,
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return AudioQualityResult(
                overall_score=0,
                snr_db=0,
                has_clipping=False,
                silence_ratio=1.0,
                is_acceptable=False,
                issues=["Failed to analyze audio"],
                recommendations=["Please try recording again"]
            )


# Singleton instance
_checker = None


def get_quality_checker() -> AudioQualityChecker:
    """Get singleton quality checker"""
    global _checker
    if _checker is None:
        _checker = AudioQualityChecker()
    return _checker
