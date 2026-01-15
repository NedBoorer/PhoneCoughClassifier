"""
Depression Voice Classifier
Detects vocal biomarkers associated with depression from speech samples
"""
import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# Severity levels (aligned with PHQ-9 categories)
SEVERITY_LEVELS = ["minimal", "mild", "moderate", "moderately_severe", "severe"]


@dataclass
class DepressionClassificationResult:
    """Classification result for depression screening"""
    severity_level: str
    confidence: float
    indicators: Dict[str, float]
    method: str
    processing_time_ms: int
    recommendation: str
    resources: Dict[str, str]
    disclaimer: str = (
        "This is a screening tool only and should NOT be used for diagnosis. "
        "If you are experiencing symptoms of depression, please reach out to a mental health professional. "
        "If you are in crisis, please contact a crisis helpline immediately."
    )


class DepressionClassifier:
    """
    Classifier for depression screening using voice analysis.
    
    Analyzes prosodic features, speech rate, and pause patterns.
    Best results with spontaneous speech or reading tasks of 10+ seconds.
    """
    
    # Feature thresholds based on literature
    # Note: Depression is complex - these are general patterns
    THRESHOLDS = {
        "pitch_range_semitones": {
            "normal": 6.0,      # > 6 semitones normal variation
            "mild": 4.0,        # 4-6 reduced variation
            "moderate": 2.0,    # 2-4 flat affect
            "severe": 0.0       # < 2 very monotone
        },
        "speech_rate": {
            "normal": 3.5,      # > 3.5 syllables/sec normal
            "mild": 2.5,        # 2.5-3.5 slightly slow
            "moderate": 1.5,    # 1.5-2.5 noticeably slow
            "severe": 0.0       # < 1.5 very slow
        },
        "pause_ratio": {
            "normal": 0.15,     # < 15% pause time normal
            "mild": 0.25,       # 15-25% mild increase
            "moderate": 0.35,   # 25-35% moderate
            "severe": 1.0       # > 35% excessive pausing
        },
        "energy_variation_coefficient": {
            "normal": 30.0,     # > 30% variation is normal
            "mild": 20.0,       # 20-30% reduced
            "moderate": 10.0,   # 10-20% flat
            "severe": 0.0       # < 10% very flat
        },
        "f0_variation_coefficient": {
            "normal": 15.0,     # > 15% pitch variation
            "mild": 10.0,       # 10-15% reduced
            "moderate": 5.0,    # 5-10% monotone tendency
            "severe": 0.0       # < 5% very monotone
        }
    }
    
    # Mental health resources
    RESOURCES = {
        "india": {
            "iCALL": "9152987821",
            "Vandrevala Foundation": "1860-2662-345",
            "NIMHANS": "080-46110007"
        },
        "international": {
            "Crisis Text Line": "Text HOME to 741741",
            "International Association for Suicide Prevention": "https://www.iasp.info/resources/Crisis_Centres/"
        }
    }
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.depression_model_path
        self._sklearn_model = None
        self._feature_extractor = None
        self.model_type = "rule_based"
        
    @property
    def feature_extractor(self):
        """Lazy load feature extractor"""
        if self._feature_extractor is None:
            from app.ml.voice_feature_extractor import get_voice_feature_extractor
            self._feature_extractor = get_voice_feature_extractor()
        return self._feature_extractor
    
    def load_sklearn_model(self):
        """Load trained sklearn model if available"""
        try:
            import joblib
            model_file = Path(self.model_path)
            if model_file.exists():
                self._sklearn_model = joblib.load(model_file)
                self.model_type = "sklearn"
                logger.info(f"Loaded depression sklearn model from {self.model_path}")
                return True
        except Exception as e:
            logger.warning(f"Could not load sklearn model: {e}")
        return False
    
    def load(self):
        """Attempt to load available models"""
        self.load_sklearn_model()
        logger.info(f"Depression classifier initialized: {self.model_type}")
    
    def classify_with_rules(
        self, 
        features: Dict[str, float]
    ) -> DepressionClassificationResult:
        """
        Rule-based classification using prosodic patterns.
        """
        start_time = time.time()
        
        scores = {}
        indicator_levels = {}
        
        # Pitch range (inverse - lower is worse)
        pitch_range = features.get("pitch_range_semitones", 8)
        if pitch_range > self.THRESHOLDS["pitch_range_semitones"]["normal"]:
            scores["pitch_range"] = 0
            indicator_levels["pitch_range"] = "normal"
        elif pitch_range > self.THRESHOLDS["pitch_range_semitones"]["mild"]:
            scores["pitch_range"] = 1
            indicator_levels["pitch_range"] = "mild"
        elif pitch_range > self.THRESHOLDS["pitch_range_semitones"]["moderate"]:
            scores["pitch_range"] = 2
            indicator_levels["pitch_range"] = "moderate"
        else:
            scores["pitch_range"] = 3
            indicator_levels["pitch_range"] = "severe"
        
        # Speech rate (inverse - slower is concerning)
        speech_rate = features.get("speech_rate", 4)
        if speech_rate > self.THRESHOLDS["speech_rate"]["normal"]:
            scores["speech_rate"] = 0
            indicator_levels["speech_rate"] = "normal"
        elif speech_rate > self.THRESHOLDS["speech_rate"]["mild"]:
            scores["speech_rate"] = 1
            indicator_levels["speech_rate"] = "mild"
        elif speech_rate > self.THRESHOLDS["speech_rate"]["moderate"]:
            scores["speech_rate"] = 2
            indicator_levels["speech_rate"] = "moderate"
        else:
            scores["speech_rate"] = 3
            indicator_levels["speech_rate"] = "severe"
        
        # Pause ratio (direct - more pauses is concerning)
        pause_ratio = features.get("pause_ratio", 0.1)
        if pause_ratio < self.THRESHOLDS["pause_ratio"]["normal"]:
            scores["pause_ratio"] = 0
            indicator_levels["pause_ratio"] = "normal"
        elif pause_ratio < self.THRESHOLDS["pause_ratio"]["mild"]:
            scores["pause_ratio"] = 1
            indicator_levels["pause_ratio"] = "mild"
        elif pause_ratio < self.THRESHOLDS["pause_ratio"]["moderate"]:
            scores["pause_ratio"] = 2
            indicator_levels["pause_ratio"] = "moderate"
        else:
            scores["pause_ratio"] = 3
            indicator_levels["pause_ratio"] = "severe"
        
        # Energy variation (inverse - less variation is concerning)
        energy_cv = features.get("energy_variation_coefficient", 35)
        if energy_cv > self.THRESHOLDS["energy_variation_coefficient"]["normal"]:
            scores["energy_variation"] = 0
            indicator_levels["energy_variation"] = "normal"
        elif energy_cv > self.THRESHOLDS["energy_variation_coefficient"]["mild"]:
            scores["energy_variation"] = 1
            indicator_levels["energy_variation"] = "mild"
        elif energy_cv > self.THRESHOLDS["energy_variation_coefficient"]["moderate"]:
            scores["energy_variation"] = 2
            indicator_levels["energy_variation"] = "moderate"
        else:
            scores["energy_variation"] = 3
            indicator_levels["energy_variation"] = "severe"
        
        # F0 variation (inverse - less variation is concerning)
        f0_cv = features.get("f0_variation_coefficient", 20)
        if f0_cv > self.THRESHOLDS["f0_variation_coefficient"]["normal"]:
            scores["f0_variation"] = 0
            indicator_levels["f0_variation"] = "normal"
        elif f0_cv > self.THRESHOLDS["f0_variation_coefficient"]["mild"]:
            scores["f0_variation"] = 1
            indicator_levels["f0_variation"] = "mild"
        elif f0_cv > self.THRESHOLDS["f0_variation_coefficient"]["moderate"]:
            scores["f0_variation"] = 2
            indicator_levels["f0_variation"] = "moderate"
        else:
            scores["f0_variation"] = 3
            indicator_levels["f0_variation"] = "severe"
        
        # Calculate overall severity
        total_score = sum(scores.values())
        max_score = len(scores) * 3
        score_ratio = total_score / max_score
        
        # Map to severity levels
        if score_ratio < 0.15:
            severity_level = "minimal"
            confidence = 0.8 - score_ratio
        elif score_ratio < 0.3:
            severity_level = "mild"
            confidence = 0.55 + (0.3 - score_ratio)
        elif score_ratio < 0.5:
            severity_level = "moderate"
            confidence = 0.5 + (0.5 - score_ratio) / 2
        elif score_ratio < 0.7:
            severity_level = "moderately_severe"
            confidence = 0.5 + score_ratio / 3
        else:
            severity_level = "severe"
            confidence = 0.5 + score_ratio / 3
        
        confidence = min(0.9, max(0.35, confidence))
        
        recommendation = self.get_recommendation(severity_level, indicator_levels)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return DepressionClassificationResult(
            severity_level=severity_level,
            confidence=confidence,
            indicators={
                "pitch_range_semitones": features.get("pitch_range_semitones", 0),
                "speech_rate": features.get("speech_rate", 0),
                "pause_ratio": features.get("pause_ratio", 0),
                "energy_variation_coefficient": features.get("energy_variation_coefficient", 0),
                "f0_variation_coefficient": features.get("f0_variation_coefficient", 0),
                "indicator_levels": indicator_levels
            },
            method="rule_based",
            processing_time_ms=processing_time,
            recommendation=recommendation,
            resources=self.RESOURCES
        )
    
    def classify_with_sklearn(
        self, 
        features: Dict[str, float]
    ) -> DepressionClassificationResult:
        """Classify using trained sklearn model"""
        start_time = time.time()
        
        feature_names = sorted(features.keys())
        X = np.array([[features[name] for name in feature_names]])
        
        prediction = self._sklearn_model.predict(X)[0]
        probabilities = self._sklearn_model.predict_proba(X)[0]
        
        severity_level = SEVERITY_LEVELS[min(prediction, len(SEVERITY_LEVELS) - 1)]
        confidence = float(max(probabilities))
        
        indicator_levels = self._analyze_indicators(features)
        recommendation = self.get_recommendation(severity_level, indicator_levels)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return DepressionClassificationResult(
            severity_level=severity_level,
            confidence=confidence,
            indicators={
                "pitch_range_semitones": features.get("pitch_range_semitones", 0),
                "speech_rate": features.get("speech_rate", 0),
                "pause_ratio": features.get("pause_ratio", 0),
                "energy_variation_coefficient": features.get("energy_variation_coefficient", 0),
                "f0_variation_coefficient": features.get("f0_variation_coefficient", 0),
                "indicator_levels": indicator_levels,
                "class_probabilities": dict(zip(SEVERITY_LEVELS, probabilities.tolist()))
            },
            method="sklearn",
            processing_time_ms=processing_time,
            recommendation=recommendation,
            resources=self.RESOURCES
        )
    
    def _analyze_indicators(self, features: Dict[str, float]) -> Dict[str, str]:
        """Analyze individual indicators"""
        indicator_levels = {}
        
        pitch_range = features.get("pitch_range_semitones", 8)
        if pitch_range > self.THRESHOLDS["pitch_range_semitones"]["normal"]:
            indicator_levels["pitch_range"] = "normal"
        elif pitch_range > self.THRESHOLDS["pitch_range_semitones"]["mild"]:
            indicator_levels["pitch_range"] = "mild"
        elif pitch_range > self.THRESHOLDS["pitch_range_semitones"]["moderate"]:
            indicator_levels["pitch_range"] = "moderate"
        else:
            indicator_levels["pitch_range"] = "severe"
        
        speech_rate = features.get("speech_rate", 4)
        if speech_rate > self.THRESHOLDS["speech_rate"]["normal"]:
            indicator_levels["speech_rate"] = "normal"
        elif speech_rate > self.THRESHOLDS["speech_rate"]["mild"]:
            indicator_levels["speech_rate"] = "mild"
        elif speech_rate > self.THRESHOLDS["speech_rate"]["moderate"]:
            indicator_levels["speech_rate"] = "moderate"
        else:
            indicator_levels["speech_rate"] = "severe"
        
        pause_ratio = features.get("pause_ratio", 0.1)
        if pause_ratio < self.THRESHOLDS["pause_ratio"]["normal"]:
            indicator_levels["pause_ratio"] = "normal"
        elif pause_ratio < self.THRESHOLDS["pause_ratio"]["mild"]:
            indicator_levels["pause_ratio"] = "mild"
        elif pause_ratio < self.THRESHOLDS["pause_ratio"]["moderate"]:
            indicator_levels["pause_ratio"] = "moderate"
        else:
            indicator_levels["pause_ratio"] = "severe"
        
        return indicator_levels
    
    def classify(self, audio_path: str) -> DepressionClassificationResult:
        """
        Main classification method for depression screening.
        
        Args:
            audio_path: Path to audio file (spontaneous speech preferred)
            
        Returns:
            DepressionClassificationResult with severity assessment
        """
        try:
            extraction_result = self.feature_extractor.extract_depression_features(audio_path)
            features = extraction_result["features"]
            
            if self._sklearn_model is not None:
                return self.classify_with_sklearn(features)
            else:
                return self.classify_with_rules(features)
                
        except Exception as e:
            logger.error(f"Depression classification failed: {e}")
            return DepressionClassificationResult(
                severity_level="unknown",
                confidence=0.0,
                indicators={},
                method="error",
                processing_time_ms=0,
                recommendation="Unable to analyze audio. Please try again with a clearer recording.",
                resources=self.RESOURCES
            )
    
    def get_recommendation(
        self, 
        severity_level: str, 
        indicator_levels: Dict[str, str]
    ) -> str:
        """Generate appropriate recommendation based on results"""
        
        if severity_level == "minimal":
            return (
                "Your voice patterns appear within the typical range. "
                "No concerning indicators were detected. "
                "Continue to prioritize your mental well-being with regular self-care."
            )
        
        elif severity_level == "mild":
            return (
                "Some mild variations were detected in your speech patterns. "
                "This could be temporary stress or fatigue. "
                "Consider practicing self-care, staying connected with loved ones, "
                "and monitoring how you're feeling. If concerns persist, "
                "talking to a counselor can be helpful."
            )
        
        elif severity_level == "moderate":
            return (
                "Your speech patterns show some variations that may warrant attention. "
                "We encourage you to speak with a mental health professional or counselor. "
                "Early support can make a significant difference. "
                "You are not alone, and help is available."
            )
        
        elif severity_level == "moderately_severe":
            return (
                "Your voice analysis suggests you may be experiencing significant stress. "
                "We strongly encourage you to reach out to a mental health professional soon. "
                "Please consider contacting one of the helplines provided. "
                "Your feelings are valid, and support is available."
            )
        
        else:  # severe
            return (
                "We are concerned about what your voice patterns may indicate. "
                "Please reach out to a mental health professional or crisis helpline today. "
                "Your well-being matters, and there are people who want to help. "
                "If you're having thoughts of self-harm, please contact a crisis line immediately."
            )


# Singleton instance
_depression_classifier = None


def get_depression_classifier() -> DepressionClassifier:
    """Get singleton depression classifier instance"""
    global _depression_classifier
    if _depression_classifier is None:
        _depression_classifier = DepressionClassifier()
        _depression_classifier.load()
    return _depression_classifier
