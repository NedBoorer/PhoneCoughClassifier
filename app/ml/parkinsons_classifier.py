"""
Parkinson's Disease Voice Classifier
Detects vocal biomarkers associated with Parkinson's Disease from sustained vowel sounds
"""
import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# Risk levels
RISK_LEVELS = ["normal", "low_risk", "moderate_risk", "elevated_risk"]


@dataclass
class ParkinsonsClassificationResult:
    """Classification result for Parkinson's disease screening"""
    risk_level: str
    confidence: float
    indicators: Dict[str, float]
    method: str
    processing_time_ms: int
    recommendation: str
    disclaimer: str = (
        "This is a screening tool only and should NOT be used for diagnosis. "
        "Please consult a neurologist for proper evaluation."
    )


class ParkinsonsClassifier:
    """
    Classifier for Parkinson's Disease detection using voice analysis.
    
    Uses jitter, shimmer, HNR, and F0 features to assess vocal biomarkers.
    Best results with sustained vowel sounds (/a/, /o/, /e/) of 3+ seconds.
    """
    
    # Clinical thresholds based on literature
    # Note: These are approximations; actual thresholds vary by study
    THRESHOLDS = {
        "jitter_relative": {
            "normal": 1.0,      # < 1.0% is normal
            "mild": 1.5,        # 1.0-1.5% mild elevation
            "moderate": 2.5,    # 1.5-2.5% moderate
            "elevated": float("inf")  # > 2.5% elevated
        },
        "shimmer_relative": {
            "normal": 3.0,      # < 3.0% is normal
            "mild": 5.0,        # 3.0-5.0% mild elevation
            "moderate": 7.0,    # 5.0-7.0% moderate
            "elevated": float("inf")
        },
        "hnr_mean": {
            "normal": 20.0,     # > 20 dB is normal
            "mild": 15.0,       # 15-20 dB mild reduction
            "moderate": 10.0,   # 10-15 dB moderate
            "elevated": 0.0     # < 10 dB significant reduction
        },
        "f0_variation_coefficient": {
            "normal": 2.0,      # < 2% normal variation
            "mild": 4.0,        # 2-4% mild
            "moderate": 6.0,    # 4-6% moderate
            "elevated": float("inf")
        }
    }
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.parkinsons_model_path
        self._sklearn_model = None
        self._feature_extractor = None
        self.model_type = "rule_based"  # Default
        
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
                logger.info(f"Loaded Parkinson's sklearn model from {self.model_path}")
                return True
        except Exception as e:
            logger.warning(f"Could not load sklearn model: {e}")
        return False
    
    def load(self):
        """Attempt to load available models"""
        self.load_sklearn_model()
        logger.info(f"Parkinson's classifier initialized: {self.model_type}")
    
    def classify_with_rules(
        self, 
        features: Dict[str, float]
    ) -> ParkinsonsClassificationResult:
        """
        Rule-based classification using clinical thresholds.
        """
        start_time = time.time()
        
        # Score each indicator
        scores = {}
        indicator_levels = {}
        
        # Check jitter
        jitter = features.get("jitter_relative", 0)
        if jitter < self.THRESHOLDS["jitter_relative"]["normal"]:
            scores["jitter"] = 0
            indicator_levels["jitter"] = "normal"
        elif jitter < self.THRESHOLDS["jitter_relative"]["mild"]:
            scores["jitter"] = 1
            indicator_levels["jitter"] = "mild"
        elif jitter < self.THRESHOLDS["jitter_relative"]["moderate"]:
            scores["jitter"] = 2
            indicator_levels["jitter"] = "moderate"
        else:
            scores["jitter"] = 3
            indicator_levels["jitter"] = "elevated"
        
        # Check shimmer
        shimmer = features.get("shimmer_relative", 0)
        if shimmer < self.THRESHOLDS["shimmer_relative"]["normal"]:
            scores["shimmer"] = 0
            indicator_levels["shimmer"] = "normal"
        elif shimmer < self.THRESHOLDS["shimmer_relative"]["mild"]:
            scores["shimmer"] = 1
            indicator_levels["shimmer"] = "mild"
        elif shimmer < self.THRESHOLDS["shimmer_relative"]["moderate"]:
            scores["shimmer"] = 2
            indicator_levels["shimmer"] = "moderate"
        else:
            scores["shimmer"] = 3
            indicator_levels["shimmer"] = "elevated"
        
        # Check HNR (reverse scale - lower is worse)
        hnr = features.get("hnr_mean", 25)
        if hnr > self.THRESHOLDS["hnr_mean"]["normal"]:
            scores["hnr"] = 0
            indicator_levels["hnr"] = "normal"
        elif hnr > self.THRESHOLDS["hnr_mean"]["mild"]:
            scores["hnr"] = 1
            indicator_levels["hnr"] = "mild"
        elif hnr > self.THRESHOLDS["hnr_mean"]["moderate"]:
            scores["hnr"] = 2
            indicator_levels["hnr"] = "moderate"
        else:
            scores["hnr"] = 3
            indicator_levels["hnr"] = "elevated"
        
        # Check F0 variation
        f0_cv = features.get("f0_variation_coefficient", 0)
        if f0_cv < self.THRESHOLDS["f0_variation_coefficient"]["normal"]:
            scores["f0_variation"] = 0
            indicator_levels["f0_variation"] = "normal"
        elif f0_cv < self.THRESHOLDS["f0_variation_coefficient"]["mild"]:
            scores["f0_variation"] = 1
            indicator_levels["f0_variation"] = "mild"
        elif f0_cv < self.THRESHOLDS["f0_variation_coefficient"]["moderate"]:
            scores["f0_variation"] = 2
            indicator_levels["f0_variation"] = "moderate"
        else:
            scores["f0_variation"] = 3
            indicator_levels["f0_variation"] = "elevated"
        
        # Calculate overall risk
        total_score = sum(scores.values())
        max_score = len(scores) * 3
        
        # Determine risk level
        score_ratio = total_score / max_score
        if score_ratio < 0.15:
            risk_level = "normal"
            confidence = 0.85 - score_ratio
        elif score_ratio < 0.35:
            risk_level = "low_risk"
            confidence = 0.6 + (0.35 - score_ratio)
        elif score_ratio < 0.6:
            risk_level = "moderate_risk"
            confidence = 0.5 + (0.6 - score_ratio) / 2
        else:
            risk_level = "elevated_risk"
            confidence = 0.5 + score_ratio / 2
        
        # Cap confidence
        confidence = min(0.95, max(0.4, confidence))
        
        # Get recommendation
        recommendation = self.get_recommendation(risk_level, indicator_levels)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return ParkinsonsClassificationResult(
            risk_level=risk_level,
            confidence=confidence,
            indicators={
                "jitter_relative": features.get("jitter_relative", 0),
                "shimmer_relative": features.get("shimmer_relative", 0),
                "hnr_mean": features.get("hnr_mean", 0),
                "f0_variation_coefficient": features.get("f0_variation_coefficient", 0),
                "indicator_levels": indicator_levels
            },
            method="rule_based",
            processing_time_ms=processing_time,
            recommendation=recommendation
        )
    
    def classify_with_sklearn(
        self, 
        features: Dict[str, float]
    ) -> ParkinsonsClassificationResult:
        """Classify using trained sklearn model"""
        start_time = time.time()
        
        # Prepare feature vector
        feature_names = sorted(features.keys())
        X = np.array([[features[name] for name in feature_names]])
        
        # Predict
        prediction = self._sklearn_model.predict(X)[0]
        probabilities = self._sklearn_model.predict_proba(X)[0]
        
        # Map prediction to risk level
        risk_level = RISK_LEVELS[prediction]
        confidence = float(max(probabilities))
        
        # Get indicator analysis for context
        indicator_levels = self._analyze_indicators(features)
        recommendation = self.get_recommendation(risk_level, indicator_levels)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return ParkinsonsClassificationResult(
            risk_level=risk_level,
            confidence=confidence,
            indicators={
                "jitter_relative": features.get("jitter_relative", 0),
                "shimmer_relative": features.get("shimmer_relative", 0),
                "hnr_mean": features.get("hnr_mean", 0),
                "f0_variation_coefficient": features.get("f0_variation_coefficient", 0),
                "indicator_levels": indicator_levels,
                "class_probabilities": dict(zip(RISK_LEVELS, probabilities.tolist()))
            },
            method="sklearn",
            processing_time_ms=processing_time,
            recommendation=recommendation
        )
    
    def _analyze_indicators(self, features: Dict[str, float]) -> Dict[str, str]:
        """Analyze individual indicators for context"""
        indicator_levels = {}
        
        jitter = features.get("jitter_relative", 0)
        if jitter < self.THRESHOLDS["jitter_relative"]["normal"]:
            indicator_levels["jitter"] = "normal"
        elif jitter < self.THRESHOLDS["jitter_relative"]["mild"]:
            indicator_levels["jitter"] = "mild"
        elif jitter < self.THRESHOLDS["jitter_relative"]["moderate"]:
            indicator_levels["jitter"] = "moderate"
        else:
            indicator_levels["jitter"] = "elevated"
        
        shimmer = features.get("shimmer_relative", 0)
        if shimmer < self.THRESHOLDS["shimmer_relative"]["normal"]:
            indicator_levels["shimmer"] = "normal"
        elif shimmer < self.THRESHOLDS["shimmer_relative"]["mild"]:
            indicator_levels["shimmer"] = "mild"
        elif shimmer < self.THRESHOLDS["shimmer_relative"]["moderate"]:
            indicator_levels["shimmer"] = "moderate"
        else:
            indicator_levels["shimmer"] = "elevated"
        
        hnr = features.get("hnr_mean", 25)
        if hnr > self.THRESHOLDS["hnr_mean"]["normal"]:
            indicator_levels["hnr"] = "normal"
        elif hnr > self.THRESHOLDS["hnr_mean"]["mild"]:
            indicator_levels["hnr"] = "mild"
        elif hnr > self.THRESHOLDS["hnr_mean"]["moderate"]:
            indicator_levels["hnr"] = "moderate"
        else:
            indicator_levels["hnr"] = "elevated"
        
        return indicator_levels
    
    def classify(self, audio_path: str) -> ParkinsonsClassificationResult:
        """
        Main classification method for Parkinson's disease screening.
        
        Args:
            audio_path: Path to audio file (sustained vowel sound preferred)
            
        Returns:
            ParkinsonsClassificationResult with risk assessment
        """
        try:
            # Extract features
            extraction_result = self.feature_extractor.extract_parkinsons_features(audio_path)
            features = extraction_result["features"]
            
            # Use sklearn if available, otherwise rules
            if self._sklearn_model is not None:
                return self.classify_with_sklearn(features)
            else:
                return self.classify_with_rules(features)
                
        except Exception as e:
            logger.error(f"Parkinson's classification failed: {e}")
            # Return safe default
            return ParkinsonsClassificationResult(
                risk_level="unknown",
                confidence=0.0,
                indicators={},
                method="error",
                processing_time_ms=0,
                recommendation="Unable to analyze audio. Please try again with a clearer recording."
            )
    
    def get_recommendation(
        self, 
        risk_level: str, 
        indicator_levels: Dict[str, str]
    ) -> str:
        """Generate appropriate recommendation based on results"""
        
        if risk_level == "normal":
            return (
                "Your voice analysis shows patterns within the normal range. "
                "No concerning indicators were detected. "
                "Continue with regular health check-ups."
            )
        
        elif risk_level == "low_risk":
            elevated = [k for k, v in indicator_levels.items() if v in ["mild", "moderate"]]
            return (
                f"Minor variations detected in: {', '.join(elevated)}. "
                "These may be normal variations or temporary. "
                "If you have concerns, consider mentioning this at your next doctor visit."
            )
        
        elif risk_level == "moderate_risk":
            elevated = [k for k, v in indicator_levels.items() if v in ["moderate", "elevated"]]
            return (
                f"Some voice patterns ({', '.join(elevated)}) show variations that warrant attention. "
                "We recommend discussing these results with your primary care physician. "
                "Many factors can affect voice quality, so professional evaluation is important."
            )
        
        else:  # elevated_risk
            return (
                "Several voice indicators show significant variations. "
                "We strongly recommend scheduling an appointment with a neurologist "
                "for a comprehensive evaluation. Early detection and intervention "
                "can be beneficial. Please remember this is a screening tool, not a diagnosis."
            )


# Singleton instance
_parkinsons_classifier = None


def get_parkinsons_classifier() -> ParkinsonsClassifier:
    """Get singleton Parkinson's classifier instance"""
    global _parkinsons_classifier
    if _parkinsons_classifier is None:
        _parkinsons_classifier = ParkinsonsClassifier()
        _parkinsons_classifier.load()
    return _parkinsons_classifier
