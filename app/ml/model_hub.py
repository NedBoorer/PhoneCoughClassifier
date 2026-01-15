"""
Voice Health Platform - Unified Model Hub
Integrates pretrained AI models for multi-disease voice screening
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import sys
import os

logger = logging.getLogger(__name__)

# Add external models to path
EXTERNAL_MODELS_DIR = Path(__file__).parent.parent.parent / "external_models"


def _to_python_float(val) -> float:
    """Convert numpy types to Python float for JSON serialization"""
    if hasattr(val, 'item'):
        return float(val.item())
    return float(val)


@dataclass
class ScreeningResult:
    """Result from a single disease screening model"""
    disease: str
    detected: bool
    confidence: float
    severity: str  # normal, mild, moderate, severe, urgent
    details: Dict = field(default_factory=dict)
    recommendation: str = ""


@dataclass
class ComprehensiveHealthResult:
    """Comprehensive result from all screening models"""
    primary_concern: str
    overall_risk_level: str  # low, moderate, high, urgent
    screenings: Dict[str, ScreeningResult] = field(default_factory=dict)
    voice_biomarkers: Dict[str, float] = field(default_factory=dict)
    processing_time_ms: int = 0
    recommendation: str = ""


class RespiratoryClassifier:
    """
    Respiratory sound classifier using PANNs (CNN6/10/14)
    Detects: Normal, Crackle, Wheeze, Both
    Maps to: Healthy, COPD indicators, Asthma indicators
    """
    
    CLASSES = ["normal", "crackle", "wheeze", "both"]
    DISEASE_MAP = {
        "normal": "healthy",
        "crackle": "copd_indicator",  # Crackles often indicate COPD/pneumonia
        "wheeze": "asthma_indicator",  # Wheezes indicate asthma/bronchitis
        "both": "severe_respiratory"
    }
    
    def __init__(self, model_type: str = "cnn6", device: str = "cpu"):
        self.model_type = model_type
        self.device = device
        self._model = None
        self._loaded = False
        
    def load(self):
        """Load pretrained PANNs model"""
        if self._loaded:
            return
            
        try:
            import torch
            import torchaudio
            
            # Add respiratory model path
            panns_path = EXTERNAL_MODELS_DIR / "respiratory_panns"
            sys.path.insert(0, str(panns_path))
            
            from models import CNN6, CNN10, CNN14
            
            weights_dir = panns_path / "panns"
            
            if self.model_type == "cnn6":
                weights_path = weights_dir / "Cnn6_mAP=0.343.pth"
                if weights_path.exists():
                    self._model = CNN6(
                        num_classes=4, 
                        from_scratch=False,
                        path_to_weights=str(weights_path),
                        device=self.device
                    )
                else:
                    logger.warning(f"CNN6 weights not found, using from scratch")
                    self._model = CNN6(num_classes=4, from_scratch=True, device=self.device)
            
            self._model.eval()
            self._loaded = True
            logger.info(f"✓ Respiratory classifier loaded ({self.model_type})")
            
        except Exception as e:
            logger.error(f"Failed to load respiratory classifier: {e}")
            raise
    
    def _preprocess_audio(self, audio_path: str) -> "torch.Tensor":
        """Preprocess audio for PANNs model"""
        import torch
        import librosa
        
        # Load and resample to 16kHz
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=16000, n_mels=64, hop_length=160, n_fft=1024
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        # Convert to tensor [1, 1, freq, time]
        tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def classify(self, audio_path: str) -> ScreeningResult:
        """Classify respiratory sounds"""
        self.load()
        
        import torch
        
        try:
            # Preprocess
            input_tensor = self._preprocess_audio(audio_path)
            
            # Inference
            with torch.no_grad():
                logits = self._model(input_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Get prediction
            pred_idx = np.argmax(probs)
            pred_class = self.CLASSES[pred_idx]
            confidence = float(probs[pred_idx])
            
            # Map to disease indicator
            disease_type = self.DISEASE_MAP[pred_class]
            
            # Determine severity
            if pred_class == "normal":
                severity = "normal"
                detected = False
                recommendation = "Respiratory sounds appear normal."
            elif pred_class == "both":
                severity = "severe"
                detected = True
                recommendation = "Multiple abnormal respiratory sounds detected. Please consult a pulmonologist promptly."
            else:
                severity = "moderate" if confidence > 0.7 else "mild"
                detected = True
                if pred_class == "crackle":
                    recommendation = "Crackling sounds detected, possibly indicating COPD or pneumonia. Consider lung function testing."
                else:
                    recommendation = "Wheezing detected, possibly indicating asthma or bronchitis. Consider spirometry testing."
            
            return ScreeningResult(
                disease=disease_type,
                detected=detected,
                confidence=confidence,
                severity=severity,
                details={
                    "sound_class": pred_class,
                    "probabilities": {c: float(p) for c, p in zip(self.CLASSES, probs)}
                },
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Respiratory classification failed: {e}")
            return ScreeningResult(
                disease="unknown",
                detected=False,
                confidence=0.0,
                severity="unknown",
                details={"error": str(e)},
                recommendation="Unable to analyze respiratory sounds."
            )


class ParkinsonsDetector:
    """
    Parkinson's disease voice detector using SVM model
    Analyzes jitter, shimmer, HNR, and other voice biomarkers
    """
    
    def __init__(self):
        self._model = None
        self._scaler = None
        self._features = None
        self._loaded = False
    
    def load(self):
        """Load pretrained Parkinson's SVM model"""
        if self._loaded:
            return
            
        try:
            import pickle
            
            model_path = EXTERNAL_MODELS_DIR / "parkinsons_detector" / "ml" / "best_pd_model.pkl"
            
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
                self._model = model_data["model"]
                self._scaler = model_data["scaler"]
                self._features = model_data["selected_features"]
            
            self._loaded = True
            logger.info("✓ Parkinson's detector loaded")
            
        except Exception as e:
            logger.error(f"Failed to load Parkinson's detector: {e}")
            raise
    
    def _extract_voice_biomarkers(self, audio_path: str) -> Dict[str, float]:
        """Extract voice biomarkers for Parkinson's detection"""
        import librosa
        import scipy.stats
        
        y, sr = librosa.load(audio_path, sr=22050)
        
        features = {}
        
        # Fundamental frequency (F0)
        pitches = librosa.yin(y, fmin=75, fmax=600)
        pitches = pitches[~np.isnan(pitches)]
        features["Fo"] = np.mean(pitches) if len(pitches) > 0 else 0
        
        # Jitter (pitch perturbation)
        if len(pitches) > 1:
            abs_diff = np.abs(np.diff(pitches))
            features["Jitter"] = np.mean(abs_diff) / (features["Fo"] + 1e-6)
        else:
            features["Jitter"] = 0
        
        # Shimmer (amplitude perturbation)
        rms = librosa.feature.rms(y=y)[0]
        if len(rms) > 1:
            abs_diff = np.abs(np.diff(rms))
            features["Shimmer"] = np.mean(abs_diff) / (np.mean(rms) + 1e-6)
            features["Shimmer(dB)"] = 20 * np.log10(features["Shimmer"] + 1e-6)
            features["Shimmer:APQ5"] = features["Shimmer"] * 0.8
            features["Shimmer:APQ11"] = features["Shimmer"] * 0.6
        else:
            features["Shimmer"] = 0
            features["Shimmer(dB)"] = 0
            features["Shimmer:APQ5"] = 0
            features["Shimmer:APQ11"] = 0
        
        # Harmonic-to-Noise Ratio (HNR)
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        features["HNR"] = np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + 1e-6)
        features["NHR"] = 1.0 / (features["HNR"] + 1e-6)
        
        # Additional complexity measures
        features["RPDE"] = scipy.stats.entropy(np.abs(librosa.stft(y).mean(axis=0)) + 1e-6)
        features["DFA"] = np.mean(np.abs(np.diff(y)))
        features["PPE"] = scipy.stats.entropy(np.abs(y) + 1e-6)
        features["spread1"] = np.std(y)
        features["spread2"] = scipy.stats.kurtosis(y)
        
        return features
    
    def detect(self, audio_path: str) -> ScreeningResult:
        """Detect Parkinson's disease indicators from voice"""
        self.load()
        
        try:
            # Extract features
            all_features = self._extract_voice_biomarkers(audio_path)
            
            # Prepare input
            X = np.array([all_features.get(f, 0) for f in self._features]).reshape(1, -1)
            X_scaled = self._scaler.transform(X)
            
            # Predict
            proba = self._model.predict_proba(X_scaled)[0]
            parkinson_prob = float(proba[1])
            healthy_prob = float(proba[0])
            
            # Threshold
            threshold = 0.7
            detected = parkinson_prob >= threshold
            confidence = parkinson_prob if detected else healthy_prob
            
            # Severity
            if not detected:
                severity = "normal"
                recommendation = "No significant Parkinson's voice indicators detected."
            elif parkinson_prob > 0.85:
                severity = "high"
                recommendation = "Strong Parkinson's voice indicators detected. Please consult a neurologist for comprehensive evaluation."
            else:
                severity = "moderate"
                recommendation = "Some Parkinson's voice indicators present. Consider follow-up neurological assessment."
            
            return ScreeningResult(
                disease="parkinsons",
                detected=detected,
                confidence=confidence,
                severity=severity,
                details={
                    "parkinson_probability": parkinson_prob,
                    "healthy_probability": healthy_prob,
                    "biomarkers": {
                        "jitter": _to_python_float(all_features.get("Jitter", 0)),
                        "shimmer": _to_python_float(all_features.get("Shimmer", 0)),
                        "hnr": _to_python_float(all_features.get("HNR", 0)),
                    }
                },
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Parkinson's detection failed: {e}")
            return ScreeningResult(
                disease="parkinsons",
                detected=False,
                confidence=0.0,
                severity="unknown",
                details={"error": str(e)},
                recommendation="Unable to analyze voice for Parkinson's indicators."
            )


class DepressionScreener:
    """
    Depression screening from speech patterns
    Analyzes pitch variability, energy, speaking rate
    """
    
    def __init__(self):
        self._loaded = False
    
    def load(self):
        """Load depression detection model"""
        if self._loaded:
            return
        # For MVP, use feature-based heuristics
        # Full model integration can be added later
        self._loaded = True
        logger.info("✓ Depression screener loaded (feature-based)")
    
    def _extract_depression_features(self, audio_path: str) -> Dict[str, float]:
        """Extract speech features relevant to depression detection"""
        import librosa
        
        y, sr = librosa.load(audio_path, sr=16000)
        
        features = {}
        
        # Pitch analysis
        pitches = librosa.yin(y, fmin=50, fmax=500)
        pitches = pitches[~np.isnan(pitches)]
        
        if len(pitches) > 0:
            features["pitch_mean"] = np.mean(pitches)
            features["pitch_std"] = np.std(pitches)
            features["pitch_range"] = np.max(pitches) - np.min(pitches)
            # Monotonicity score (lower variability = more monotone)
            features["monotonicity"] = 1.0 / (features["pitch_std"] + 1e-6)
        else:
            features["pitch_mean"] = 0
            features["pitch_std"] = 0
            features["pitch_range"] = 0
            features["monotonicity"] = 0
        
        # Energy analysis
        rms = librosa.feature.rms(y=y)[0]
        features["energy_mean"] = np.mean(rms)
        features["energy_std"] = np.std(rms)
        
        # Speaking rate (via onset detection)
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        duration = len(y) / sr
        features["speaking_rate"] = len(onsets) / duration if duration > 0 else 0
        
        # Pause analysis
        silence_threshold = 0.01
        is_silent = rms < silence_threshold
        features["silence_ratio"] = np.mean(is_silent)
        
        return features
    
    def screen(self, audio_path: str) -> ScreeningResult:
        """Screen for depression indicators from speech"""
        self.load()
        
        try:
            features = self._extract_depression_features(audio_path)
            
            # Scoring based on research indicators:
            # - Low pitch variability (monotone)
            # - Lower energy
            # - Slower speaking rate
            # - More pauses
            
            risk_score = 0.0
            indicators = []
            
            # Monotonicity check
            if features["pitch_std"] < 20:  # Low pitch variation
                risk_score += 0.25
                indicators.append("monotone_speech")
            
            # Low energy
            if features["energy_mean"] < 0.02:
                risk_score += 0.25
                indicators.append("low_energy")
            
            # Slow speaking rate
            if features["speaking_rate"] < 1.5:
                risk_score += 0.25
                indicators.append("slow_speech")
            
            # High pause ratio
            if features["silence_ratio"] > 0.4:
                risk_score += 0.25
                indicators.append("frequent_pauses")
            
            detected = risk_score >= 0.5
            
            if not detected:
                severity = "normal"
                recommendation = "Speech patterns appear normal."
            elif risk_score >= 0.75:
                severity = "moderate"
                recommendation = "Multiple speech indicators suggest possible depressive symptoms. Consider speaking with a mental health professional."
            else:
                severity = "mild"
                recommendation = "Some speech patterns may indicate low mood. Monitor and consider follow-up if persistent."
            
            return ScreeningResult(
                disease="depression",
                detected=detected,
                confidence=risk_score,
                severity=severity,
                details={
                    "indicators": indicators,
                    "features": {k: _to_python_float(v) for k, v in features.items()}
                },
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Depression screening failed: {e}")
            return ScreeningResult(
                disease="depression",
                detected=False,
                confidence=0.0,
                severity="unknown",
                details={"error": str(e)},
                recommendation="Unable to analyze speech for mood indicators."
            )


class ModelHub:
    """
    Unified interface for all voice health screening models
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._respiratory = None
        self._parkinsons = None
        self._depression = None
    
    @property
    def respiratory_classifier(self) -> RespiratoryClassifier:
        if self._respiratory is None:
            self._respiratory = RespiratoryClassifier(device=self.device)
        return self._respiratory
    
    @property
    def parkinsons_detector(self) -> ParkinsonsDetector:
        if self._parkinsons is None:
            self._parkinsons = ParkinsonsDetector()
        return self._parkinsons
    
    @property
    def depression_screener(self) -> DepressionScreener:
        if self._depression is None:
            self._depression = DepressionScreener()
        return self._depression
    
    def run_full_analysis(
        self, 
        audio_path: str,
        enable_respiratory: bool = True,
        enable_parkinsons: bool = False,
        enable_depression: bool = False
    ) -> ComprehensiveHealthResult:
        """
        Run comprehensive voice health analysis.
        
        Args:
            audio_path: Path to audio file
            enable_respiratory: Enable COPD/Asthma screening
            enable_parkinsons: Enable Parkinson's screening (opt-in)
            enable_depression: Enable depression screening (opt-in)
        """
        import time
        start_time = time.time()
        
        screenings = {}
        voice_biomarkers = {}
        
        # Respiratory screening (enabled by default)
        if enable_respiratory:
            try:
                resp_result = self.respiratory_classifier.classify(audio_path)
                screenings["respiratory"] = resp_result
            except Exception as e:
                logger.error(f"Respiratory screening failed: {e}")
        
        # Parkinson's screening (opt-in)
        if enable_parkinsons:
            try:
                pd_result = self.parkinsons_detector.detect(audio_path)
                screenings["parkinsons"] = pd_result
                # Extract biomarkers
                if pd_result.details.get("biomarkers"):
                    voice_biomarkers.update(pd_result.details["biomarkers"])
            except Exception as e:
                logger.error(f"Parkinson's screening failed: {e}")
        
        # Depression screening (opt-in)
        if enable_depression:
            try:
                dep_result = self.depression_screener.screen(audio_path)
                screenings["depression"] = dep_result
                if dep_result.details.get("features"):
                    voice_biomarkers["speaking_rate"] = dep_result.details["features"].get("speaking_rate", 0)
                    voice_biomarkers["pitch_variability"] = dep_result.details["features"].get("pitch_std", 0)
            except Exception as e:
                logger.error(f"Depression screening failed: {e}")
        
        # Determine primary concern and overall risk
        primary_concern = "none"
        overall_risk = "low"
        recommendations = []
        
        severity_rank = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3, "high": 3, "urgent": 4}
        
        for name, result in screenings.items():
            if result.detected:
                if severity_rank.get(result.severity, 0) > severity_rank.get(overall_risk, 0):
                    overall_risk = result.severity
                    primary_concern = result.disease
                recommendations.append(result.recommendation)
        
        if overall_risk == "normal":
            overall_risk = "low"
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return ComprehensiveHealthResult(
            primary_concern=primary_concern,
            overall_risk_level=overall_risk,
            screenings=screenings,
            voice_biomarkers=voice_biomarkers,
            processing_time_ms=processing_time_ms,
            recommendation=" ".join(recommendations) if recommendations else "All screenings appear normal."
        )


# Singleton instance
_model_hub = None


def get_model_hub(device: str = "cpu") -> ModelHub:
    """Get singleton ModelHub instance"""
    global _model_hub
    if _model_hub is None:
        _model_hub = ModelHub(device=device)
    return _model_hub
