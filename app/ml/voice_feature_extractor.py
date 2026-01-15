"""
Voice Feature Extractor - Specialized for Neurological and Mental Health Analysis
Extracts features for Parkinson's Disease and Depression detection from voice
"""
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.ml.feature_extractor import FeatureExtractor, SAMPLE_RATE, HOP_LENGTH, N_FFT

logger = logging.getLogger(__name__)


class VoiceFeatureExtractor(FeatureExtractor):
    """
    Extended feature extractor for voice-based health analysis.
    Adds specialized features for Parkinson's Disease and Depression detection.
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        super().__init__(sample_rate)
    
    # ==================
    # Parkinson's Features
    # ==================
    
    def extract_jitter(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract jitter (frequency perturbation) features.
        Jitter measures cycle-to-cycle variation in fundamental frequency.
        Elevated jitter is associated with Parkinson's disease.
        """
        try:
            # Extract pitch using librosa's pyin
            f0, voiced_flag, voiced_probs = self.librosa.pyin(
                y,
                fmin=self.librosa.note_to_hz('C2'),
                fmax=self.librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            
            # Get voiced frames only
            f0_voiced = f0[~np.isnan(f0)]
            
            if len(f0_voiced) < 3:
                return {
                    "jitter_absolute": 0.0,
                    "jitter_relative": 0.0,
                    "jitter_rap": 0.0,
                    "jitter_ppq5": 0.0
                }
            
            # Calculate period (T = 1/f0)
            periods = 1.0 / f0_voiced
            
            # Absolute jitter (mean absolute difference between consecutive periods)
            period_diffs = np.abs(np.diff(periods))
            jitter_absolute = np.mean(period_diffs)
            
            # Relative jitter (percentage)
            jitter_relative = (jitter_absolute / np.mean(periods)) * 100
            
            # RAP (Relative Average Perturbation) - 3-point average
            rap_values = []
            for i in range(1, len(periods) - 1):
                avg_period = (periods[i-1] + periods[i] + periods[i+1]) / 3
                rap_values.append(abs(periods[i] - avg_period))
            jitter_rap = (np.mean(rap_values) / np.mean(periods)) * 100 if rap_values else 0.0
            
            # PPQ5 (5-point Period Perturbation Quotient)
            ppq5_values = []
            for i in range(2, len(periods) - 2):
                avg_period = np.mean(periods[i-2:i+3])
                ppq5_values.append(abs(periods[i] - avg_period))
            jitter_ppq5 = (np.mean(ppq5_values) / np.mean(periods)) * 100 if ppq5_values else 0.0
            
            return {
                "jitter_absolute": float(jitter_absolute),
                "jitter_relative": float(jitter_relative),
                "jitter_rap": float(jitter_rap),
                "jitter_ppq5": float(jitter_ppq5)
            }
            
        except Exception as e:
            logger.warning(f"Jitter extraction failed: {e}")
            return {
                "jitter_absolute": 0.0,
                "jitter_relative": 0.0,
                "jitter_rap": 0.0,
                "jitter_ppq5": 0.0
            }
    
    def extract_shimmer(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract shimmer (amplitude perturbation) features.
        Shimmer measures cycle-to-cycle variation in amplitude.
        Elevated shimmer is associated with Parkinson's disease.
        """
        try:
            # Get RMS energy per frame
            rms = self.librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
            
            # Filter out very low energy frames (silence)
            threshold = np.mean(rms) * 0.1
            rms_voiced = rms[rms > threshold]
            
            if len(rms_voiced) < 3:
                return {
                    "shimmer_absolute": 0.0,
                    "shimmer_relative": 0.0,
                    "shimmer_apq3": 0.0,
                    "shimmer_apq5": 0.0,
                    "shimmer_dda": 0.0
                }
            
            # Convert to dB
            amplitudes = rms_voiced
            amplitudes_db = self.librosa.amplitude_to_db(amplitudes, ref=np.max)
            
            # Absolute shimmer (mean absolute difference)
            amp_diffs = np.abs(np.diff(amplitudes))
            shimmer_absolute = np.mean(amp_diffs)
            
            # Relative shimmer (percentage)
            shimmer_relative = (shimmer_absolute / np.mean(amplitudes)) * 100
            
            # APQ3 (3-point Amplitude Perturbation Quotient)
            apq3_values = []
            for i in range(1, len(amplitudes) - 1):
                avg_amp = (amplitudes[i-1] + amplitudes[i] + amplitudes[i+1]) / 3
                apq3_values.append(abs(amplitudes[i] - avg_amp))
            shimmer_apq3 = (np.mean(apq3_values) / np.mean(amplitudes)) * 100 if apq3_values else 0.0
            
            # APQ5 (5-point Amplitude Perturbation Quotient)
            apq5_values = []
            for i in range(2, len(amplitudes) - 2):
                avg_amp = np.mean(amplitudes[i-2:i+3])
                apq5_values.append(abs(amplitudes[i] - avg_amp))
            shimmer_apq5 = (np.mean(apq5_values) / np.mean(amplitudes)) * 100 if apq5_values else 0.0
            
            # DDA (Difference of Differences of Amplitudes)
            shimmer_dda = shimmer_apq3 * 3  # DDA = 3 * APQ3
            
            return {
                "shimmer_absolute": float(shimmer_absolute),
                "shimmer_relative": float(shimmer_relative),
                "shimmer_apq3": float(shimmer_apq3),
                "shimmer_apq5": float(shimmer_apq5),
                "shimmer_dda": float(shimmer_dda)
            }
            
        except Exception as e:
            logger.warning(f"Shimmer extraction failed: {e}")
            return {
                "shimmer_absolute": 0.0,
                "shimmer_relative": 0.0,
                "shimmer_apq3": 0.0,
                "shimmer_apq5": 0.0,
                "shimmer_dda": 0.0
            }
    
    def extract_hnr(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract Harmonics-to-Noise Ratio (HNR) features.
        HNR indicates voice quality - lower values suggest breathiness/hoarseness.
        Reduced HNR is associated with Parkinson's disease.
        """
        try:
            # Use harmonic-percussive separation
            y_harmonic, y_percussive = self.librosa.effects.hpss(y)
            
            # Calculate energy ratios
            harmonic_energy = np.sum(y_harmonic ** 2)
            noise_energy = np.sum(y_percussive ** 2)
            
            # Avoid division by zero
            if noise_energy < 1e-10:
                hnr_db = 30.0  # Very clean signal
            else:
                hnr_ratio = harmonic_energy / noise_energy
                hnr_db = 10 * np.log10(hnr_ratio) if hnr_ratio > 0 else 0.0
            
            # Calculate frame-wise HNR
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            hop = frame_length // 2
            
            hnr_frames = []
            for i in range(0, len(y) - frame_length, hop):
                frame = y[i:i + frame_length]
                h_frame, p_frame = self.librosa.effects.hpss(frame)
                h_energy = np.sum(h_frame ** 2)
                p_energy = np.sum(p_frame ** 2)
                if p_energy > 1e-10:
                    hnr_frames.append(10 * np.log10(h_energy / p_energy))
            
            hnr_mean = np.mean(hnr_frames) if hnr_frames else hnr_db
            hnr_std = np.std(hnr_frames) if hnr_frames else 0.0
            
            return {
                "hnr_mean": float(hnr_mean),
                "hnr_std": float(hnr_std),
                "hnr_min": float(np.min(hnr_frames)) if hnr_frames else 0.0,
                "hnr_max": float(np.max(hnr_frames)) if hnr_frames else float(hnr_db)
            }
            
        except Exception as e:
            logger.warning(f"HNR extraction failed: {e}")
            return {
                "hnr_mean": 0.0,
                "hnr_std": 0.0,
                "hnr_min": 0.0,
                "hnr_max": 0.0
            }
    
    def extract_f0_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract fundamental frequency (F0) features.
        F0 variations are indicative of both Parkinson's and depression.
        """
        try:
            # Extract F0 using pyin
            f0, voiced_flag, voiced_probs = self.librosa.pyin(
                y,
                fmin=self.librosa.note_to_hz('C2'),
                fmax=self.librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            
            # Get voiced frames only
            f0_voiced = f0[~np.isnan(f0)]
            
            if len(f0_voiced) < 3:
                return {
                    "f0_mean": 0.0,
                    "f0_std": 0.0,
                    "f0_min": 0.0,
                    "f0_max": 0.0,
                    "f0_range": 0.0,
                    "f0_variation_coefficient": 0.0,
                    "voiced_fraction": 0.0
                }
            
            f0_mean = np.mean(f0_voiced)
            f0_std = np.std(f0_voiced)
            f0_min = np.min(f0_voiced)
            f0_max = np.max(f0_voiced)
            f0_range = f0_max - f0_min
            
            # Coefficient of variation (normalized measure)
            f0_cv = (f0_std / f0_mean) * 100 if f0_mean > 0 else 0.0
            
            # Fraction of voiced frames
            voiced_fraction = np.sum(~np.isnan(f0)) / len(f0)
            
            return {
                "f0_mean": float(f0_mean),
                "f0_std": float(f0_std),
                "f0_min": float(f0_min),
                "f0_max": float(f0_max),
                "f0_range": float(f0_range),
                "f0_variation_coefficient": float(f0_cv),
                "voiced_fraction": float(voiced_fraction)
            }
            
        except Exception as e:
            logger.warning(f"F0 extraction failed: {e}")
            return {
                "f0_mean": 0.0,
                "f0_std": 0.0,
                "f0_min": 0.0,
                "f0_max": 0.0,
                "f0_range": 0.0,
                "f0_variation_coefficient": 0.0,
                "voiced_fraction": 0.0
            }
    
    # ==================
    # Depression Features
    # ==================
    
    def extract_speech_rate(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract speech rate features.
        Slower speech rate is associated with depression.
        """
        try:
            duration = len(y) / self.sample_rate
            
            # Detect onsets (syllable approximation)
            onset_env = self.librosa.onset.onset_strength(
                y=y, sr=self.sample_rate, hop_length=HOP_LENGTH
            )
            onsets = self.librosa.onset.onset_detect(
                onset_envelope=onset_env, sr=self.sample_rate
            )
            
            # Syllables per second (approximation)
            num_syllables = len(onsets)
            syllable_rate = num_syllables / duration if duration > 0 else 0.0
            
            # Articulation rate (syllables per voiced time)
            # Get voiced segments
            rms = self.librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
            threshold = np.mean(rms) * 0.3
            voiced_frames = np.sum(rms > threshold)
            voiced_duration = (voiced_frames * HOP_LENGTH) / self.sample_rate
            
            articulation_rate = num_syllables / voiced_duration if voiced_duration > 0 else 0.0
            
            return {
                "speech_rate": float(syllable_rate),
                "articulation_rate": float(articulation_rate),
                "num_syllables": float(num_syllables),
                "total_duration": float(duration),
                "voiced_duration": float(voiced_duration)
            }
            
        except Exception as e:
            logger.warning(f"Speech rate extraction failed: {e}")
            return {
                "speech_rate": 0.0,
                "articulation_rate": 0.0,
                "num_syllables": 0.0,
                "total_duration": 0.0,
                "voiced_duration": 0.0
            }
    
    def extract_pause_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract pause-related features.
        Longer and more frequent pauses are associated with depression.
        """
        try:
            # Get RMS energy
            rms = self.librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
            duration = len(y) / self.sample_rate
            
            # Dynamic threshold based on signal
            threshold = np.mean(rms) * 0.2
            
            # Find pause segments (below threshold)
            is_pause = rms < threshold
            
            # Calculate pause statistics
            pause_durations = []
            current_pause = 0
            
            for is_p in is_pause:
                if is_p:
                    current_pause += 1
                elif current_pause > 0:
                    # Convert frames to seconds
                    pause_sec = (current_pause * HOP_LENGTH) / self.sample_rate
                    if pause_sec > 0.1:  # Only count pauses > 100ms
                        pause_durations.append(pause_sec)
                    current_pause = 0
            
            # Handle trailing pause
            if current_pause > 0:
                pause_sec = (current_pause * HOP_LENGTH) / self.sample_rate
                if pause_sec > 0.1:
                    pause_durations.append(pause_sec)
            
            num_pauses = len(pause_durations)
            total_pause_time = sum(pause_durations)
            pause_ratio = total_pause_time / duration if duration > 0 else 0.0
            
            return {
                "num_pauses": float(num_pauses),
                "total_pause_duration": float(total_pause_time),
                "mean_pause_duration": float(np.mean(pause_durations)) if pause_durations else 0.0,
                "max_pause_duration": float(max(pause_durations)) if pause_durations else 0.0,
                "pause_ratio": float(pause_ratio),
                "pause_rate": float(num_pauses / duration) if duration > 0 else 0.0
            }
            
        except Exception as e:
            logger.warning(f"Pause feature extraction failed: {e}")
            return {
                "num_pauses": 0.0,
                "total_pause_duration": 0.0,
                "mean_pause_duration": 0.0,
                "max_pause_duration": 0.0,
                "pause_ratio": 0.0,
                "pause_rate": 0.0
            }
    
    def extract_prosodic_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract prosodic features (pitch contour, energy variation).
        Reduced prosodic variation (monotonous speech) is associated with depression.
        """
        try:
            # Get F0 contour
            f0, voiced_flag, voiced_probs = self.librosa.pyin(
                y,
                fmin=self.librosa.note_to_hz('C2'),
                fmax=self.librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            
            f0_voiced = f0[~np.isnan(f0)]
            
            # RMS energy contour
            rms = self.librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
            
            # Pitch contour features
            if len(f0_voiced) > 2:
                # Pitch variation
                pitch_range_semitones = 12 * np.log2(max(f0_voiced) / min(f0_voiced)) if min(f0_voiced) > 0 else 0.0
                
                # Pitch slope (contour direction)
                pitch_slope = np.polyfit(range(len(f0_voiced)), f0_voiced, 1)[0]
                
                # Pitch inflections (direction changes)
                pitch_diff = np.diff(f0_voiced)
                inflections = np.sum(np.diff(np.sign(pitch_diff)) != 0)
                inflection_rate = inflections / len(f0_voiced)
            else:
                pitch_range_semitones = 0.0
                pitch_slope = 0.0
                inflection_rate = 0.0
            
            # Energy variation
            energy_mean = np.mean(rms)
            energy_std = np.std(rms)
            energy_cv = (energy_std / energy_mean) * 100 if energy_mean > 0 else 0.0
            
            # Energy range in dB
            rms_nonzero = rms[rms > 1e-10]
            if len(rms_nonzero) > 0:
                energy_range_db = 20 * np.log10(max(rms_nonzero) / min(rms_nonzero))
            else:
                energy_range_db = 0.0
            
            return {
                "pitch_range_semitones": float(pitch_range_semitones),
                "pitch_slope": float(pitch_slope),
                "pitch_inflection_rate": float(inflection_rate),
                "energy_mean": float(energy_mean),
                "energy_std": float(energy_std),
                "energy_variation_coefficient": float(energy_cv),
                "energy_range_db": float(energy_range_db)
            }
            
        except Exception as e:
            logger.warning(f"Prosodic feature extraction failed: {e}")
            return {
                "pitch_range_semitones": 0.0,
                "pitch_slope": 0.0,
                "pitch_inflection_rate": 0.0,
                "energy_mean": 0.0,
                "energy_std": 0.0,
                "energy_variation_coefficient": 0.0,
                "energy_range_db": 0.0
            }
    
    # ==================
    # Combined Extraction
    # ==================
    
    def extract_parkinsons_features(self, audio_path: str) -> Dict:
        """
        Extract all features relevant for Parkinson's disease detection.
        Best used with sustained vowel sounds (/a/, /o/, /e/).
        """
        try:
            y, sr = self.load_audio(audio_path)
            
            if len(y) < self.sample_rate * 0.5:  # Less than 0.5 seconds
                raise ValueError("Audio too short for Parkinson's analysis")
            
            features = {}
            
            # Core Parkinson's features
            features.update(self.extract_jitter(y))
            features.update(self.extract_shimmer(y))
            features.update(self.extract_hnr(y))
            features.update(self.extract_f0_features(y))
            
            # Additional features from base class
            features.update(self.extract_spectral(y))
            
            # Add MFCCs (useful for all voice analysis)
            mfcc_features = self.extract_mfcc(y)
            for key, value in mfcc_features.items():
                if isinstance(value, np.ndarray):
                    for i, v in enumerate(value):
                        features[f"{key}_{i}"] = float(v)
                else:
                    features[key] = float(value)
            
            return {
                "features": features,
                "sample_rate": sr,
                "duration": len(y) / sr,
                "analysis_type": "parkinsons"
            }
            
        except Exception as e:
            logger.error(f"Parkinson's feature extraction failed: {e}")
            raise
    
    def extract_depression_features(self, audio_path: str) -> Dict:
        """
        Extract all features relevant for depression detection.
        Best used with spontaneous speech or reading tasks.
        """
        try:
            y, sr = self.load_audio(audio_path)
            
            if len(y) < self.sample_rate * 1.0:  # Less than 1 second
                raise ValueError("Audio too short for depression analysis (need at least 1 second)")
            
            features = {}
            
            # Core depression features
            features.update(self.extract_f0_features(y))
            features.update(self.extract_speech_rate(y))
            features.update(self.extract_pause_features(y))
            features.update(self.extract_prosodic_features(y))
            
            # Additional features
            features.update(self.extract_spectral(y))
            features.update(self.extract_temporal(y))
            
            # Add MFCCs
            mfcc_features = self.extract_mfcc(y)
            for key, value in mfcc_features.items():
                if isinstance(value, np.ndarray):
                    for i, v in enumerate(value):
                        features[f"{key}_{i}"] = float(v)
                else:
                    features[key] = float(value)
            
            return {
                "features": features,
                "sample_rate": sr,
                "duration": len(y) / sr,
                "analysis_type": "depression"
            }
            
        except Exception as e:
            logger.error(f"Depression feature extraction failed: {e}")
            raise


# Singleton instance
_voice_extractor = None


def get_voice_feature_extractor() -> VoiceFeatureExtractor:
    """Get singleton voice feature extractor instance"""
    global _voice_extractor
    if _voice_extractor is None:
        _voice_extractor = VoiceFeatureExtractor()
    return _voice_extractor
