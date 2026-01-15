"""
Phone Cough Classifier - Main Classifier
Uses Google HeAR embeddings + sklearn classifier for cough classification
"""
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import json

from app.config import settings

logger = logging.getLogger(__name__)

# Classification labels
COUGH_CLASSES = ["dry", "wet", "whooping", "chronic", "normal"]


@dataclass
class ClassificationResult:
    """Classification result with probabilities and metadata"""
    classification: str
    confidence: float
    probabilities: Dict[str, float]
    method: str
    processing_time_ms: int
    severity: str
    recommendation: str


class CoughClassifier:
    """
    Main cough classifier using Google HeAR embeddings or sklearn fallback
    """
    
    def __init__(self, model_path: Optional[str] = None, use_hear: bool = True):
        self.model_path = model_path or settings.model_path
        self.use_hear = use_hear and settings.use_hear_embeddings
        self.model_type = "initializing"
        
        self._sklearn_model = None
        self._hear_model = None
        self._feature_extractor = None
        self._loaded = False
    
    @property
    def feature_extractor(self):
        """Lazy load feature extractor"""
        if self._feature_extractor is None:
            from app.ml.feature_extractor import get_feature_extractor
            self._feature_extractor = get_feature_extractor()
        return self._feature_extractor
    
    def load_sklearn_model(self) -> bool:
        """Load scikit-learn model from disk"""
        try:
            model_file = Path(self.model_path)
            if model_file.exists():
                import joblib
                self._sklearn_model = joblib.load(model_file)
                self.model_type = "sklearn_random_forest"
                logger.info(f"✓ Loaded sklearn model from {model_file}")
                return True
            else:
                logger.warning(f"No sklearn model found at {model_file}")
                return False
        except Exception as e:
            logger.error(f"Failed to load sklearn model: {e}")
            return False
    
    def load_hear_model(self) -> bool:
        """Load Google HeAR model from Hugging Face"""
        if not self.use_hear:
            return False
        
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
            
            logger.info("Loading Google HeAR model from Hugging Face...")
            
            # Load HeAR model
            self._hear_model = {
                "processor": AutoProcessor.from_pretrained("google/hear-pytorch", trust_remote_code=True),
                "model": AutoModel.from_pretrained("google/hear-pytorch", trust_remote_code=True)
            }
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._hear_model["model"].to(device)
            self._hear_model["device"] = device
            
            self.model_type = "hear_embeddings"
            logger.info(f"✓ HeAR model loaded on {device}")
            return True
            
        except Exception as e:
            logger.warning(f"HeAR model not available: {e}")
            logger.info("Falling back to sklearn classifier")
            return False
    
    def load(self):
        """Load all available models"""
        if self._loaded:
            return
        
        # Try HeAR first (best accuracy)
        if self.use_hear:
            if self.load_hear_model():
                self._loaded = True
                return
        
        # Try sklearn model
        if self.load_sklearn_model():
            self._loaded = True
            return
        
        # Fallback to rule-based
        self.model_type = "rule_based"
        self._loaded = True
        logger.info("Using rule-based classifier (no trained model)")
    
    def extract_hear_embeddings(self, audio_path: str) -> np.ndarray:
        """Extract embeddings using Google HeAR model"""
        import torch
        import librosa
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Pad or truncate to 2 seconds (HeAR expects 2s clips)
        target_length = 2 * 16000
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]
        
        # Process with HeAR
        processor = self._hear_model["processor"]
        model = self._hear_model["model"]
        device = self._hear_model["device"]
        
        # Create input
        inputs = processor(y, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy().flatten()
    
    def classify_with_hear(self, audio_path: str) -> Tuple[str, float, Dict[str, float]]:
        """Classify using HeAR embeddings + simple classifier"""
        embeddings = self.extract_hear_embeddings(audio_path)
        
        # If we have a trained sklearn model, use it
        if self._sklearn_model is not None:
            proba = self._sklearn_model.predict_proba([embeddings])[0]
            class_idx = np.argmax(proba)
            return (
                COUGH_CLASSES[class_idx],
                float(proba[class_idx]),
                {c: float(p) for c, p in zip(COUGH_CLASSES, proba)}
            )
        
        # Otherwise, use embedding-based heuristics
        return self.classify_with_rules(audio_path, embeddings)
    
    def classify_with_sklearn(self, audio_path: str) -> Tuple[str, float, Dict[str, float]]:
        """Classify using sklearn model with extracted features"""
        features, _ = self.feature_extractor.get_feature_vector(audio_path)
        
        proba = self._sklearn_model.predict_proba([features])[0]
        class_idx = np.argmax(proba)
        
        return (
            COUGH_CLASSES[class_idx],
            float(proba[class_idx]),
            {c: float(p) for c, p in zip(COUGH_CLASSES, proba)}
        )
    
    def classify_with_rules(self, audio_path: str, embeddings: Optional[np.ndarray] = None) -> Tuple[str, float, Dict[str, float]]:
        """Rule-based classification using acoustic features"""
        result = self.feature_extractor.extract_all_features(audio_path)
        features = result["features"]
        
        # Rule-based scoring
        scores = {c: 0.0 for c in COUGH_CLASSES}
        
        # Duration-based rules
        duration = features.get("duration", 1.0)
        if duration < 0.5:
            scores["dry"] += 0.3
        elif duration > 2.0:
            scores["chronic"] += 0.4
        
        # Spectral centroid (higher = drier cough)
        centroid = features.get("spectral_centroid_mean", 2000)
        if centroid > 3000:
            scores["dry"] += 0.3
        elif centroid < 1500:
            scores["wet"] += 0.3
        
        # ZCR (higher = more noise/raspiness)
        zcr = features.get("zcr_mean", 0.1)
        if zcr > 0.15:
            scores["whooping"] += 0.2
        elif zcr < 0.05:
            scores["wet"] += 0.2
        
        # RMS energy patterns
        rms_std = features.get("rms_std", 0.1)
        if rms_std > 0.2:
            scores["whooping"] += 0.3
        
        # Number of onsets (cough bursts)
        num_onsets = features.get("num_onsets", 1)
        if num_onsets > 3:
            scores["chronic"] += 0.3
            scores["whooping"] += 0.2
        elif num_onsets == 1:
            scores["normal"] += 0.3
        
        # Normalize scores
        total = sum(scores.values())
        if total == 0:
            # Default to normal
            scores["normal"] = 1.0
            total = 1.0
        
        probabilities = {c: s / total for c, s in scores.items()}
        best_class = max(probabilities, key=probabilities.get)
        
        return best_class, probabilities[best_class], probabilities
    
    def classify(self, audio_path: str) -> ClassificationResult:
        """
        Main classification method. Uses best available model.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            
        Returns:
            ClassificationResult with classification, confidence, and metadata
        """
        import time
        start_time = time.time()
        
        # Ensure model is loaded
        self.load()
        
        try:
            # Choose classification method
            if self.model_type == "hear_embeddings" and self._hear_model:
                classification, confidence, probabilities = self.classify_with_hear(audio_path)
            elif self._sklearn_model:
                classification, confidence, probabilities = self.classify_with_sklearn(audio_path)
            else:
                classification, confidence, probabilities = self.classify_with_rules(audio_path)
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Determine severity and recommendation
            severity, recommendation = self.get_recommendation(classification, confidence)
            
            return ClassificationResult(
                classification=classification,
                confidence=confidence,
                probabilities=probabilities,
                method=self.model_type,
                processing_time_ms=processing_time_ms,
                severity=severity,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            # Return unknown classification
            return ClassificationResult(
                classification="unknown",
                confidence=0.0,
                probabilities={c: 0.2 for c in COUGH_CLASSES},
                method="error",
                processing_time_ms=int((time.time() - start_time) * 1000),
                severity="unknown",
                recommendation="Unable to classify. Please try again with a clearer recording."
            )
    
    def get_recommendation(self, classification: str, confidence: float) -> Tuple[str, str]:
        """Get severity level and health recommendation based on classification"""
        recommendations = {
            "dry": (
                "mild",
                "Your cough appears to be dry and non-productive. Stay hydrated, "
                "use a humidifier, and consider honey or lozenges. If symptoms persist "
                "beyond 2 weeks, please consult a healthcare provider."
            ),
            "wet": (
                "moderate",
                "Your cough appears to be productive (wet). This often indicates your body "
                "is clearing mucus. Stay hydrated and rest. If you see blood in mucus, have "
                "difficulty breathing, or symptoms worsen, seek medical attention."
            ),
            "whooping": (
                "urgent",
                "Your cough has characteristics of whooping cough (pertussis). This is a "
                "contagious respiratory infection. Please consult a doctor promptly for "
                "testing and appropriate treatment. Avoid contact with infants and elderly."
            ),
            "chronic": (
                "moderate",
                "Your cough appears to be chronic (lasting more than 3 weeks). Chronic coughs "
                "can have many causes including asthma, GERD, or post-nasal drip. Please "
                "schedule a visit with your healthcare provider for proper evaluation."
            ),
            "normal": (
                "mild",
                "Your cough appears to be a typical acute cough, likely due to a minor "
                "respiratory infection. Rest, stay hydrated, and monitor symptoms. If fever "
                "develops or symptoms worsen, consult a healthcare provider."
            )
        }

        return recommendations.get(classification, ("unknown", "Unable to provide recommendation."))

    def adjust_for_occupational_risk(
        self,
        classification: str,
        confidence: float,
        severity: str,
        recommendation: str,
        occupational_data: Optional[Dict] = None
    ) -> Tuple[str, str]:
        """
        Adjust severity and recommendation based on occupational hazards.

        For farmers exposed to pesticides, dust, or other occupational hazards,
        respiratory symptoms may indicate more serious conditions.

        Args:
            classification: Cough classification
            confidence: Classification confidence
            severity: Initial severity level
            recommendation: Initial recommendation
            occupational_data: Dict with keys: occupation, pesticide_exposure,
                             dust_exposure, work_environment

        Returns:
            Tuple of (adjusted_severity, adjusted_recommendation)
        """
        if not occupational_data:
            return severity, recommendation

        is_farmer = occupational_data.get("occupation") in ["farmer", "farm_worker"]
        pesticide_exposure = occupational_data.get("pesticide_exposure", False)
        dust_exposure = occupational_data.get("dust_exposure", False)

        if not is_farmer:
            return severity, recommendation

        # Risk escalation matrix
        new_severity = severity
        additional_warnings = []

        # CRITICAL: Pesticide exposure + respiratory symptoms
        if pesticide_exposure:
            if classification in ["wet", "whooping"] or (classification == "chronic" and confidence > 0.6):
                # Escalate to urgent - possible organophosphate poisoning
                new_severity = "urgent"
                additional_warnings.append(
                    "⚠️ URGENT: Pesticide exposure with breathing problems can indicate "
                    "organophosphate poisoning. Stop using chemicals immediately. "
                    "See a doctor TODAY. Symptoms may worsen rapidly."
                )
            elif classification in ["dry", "normal"]:
                # Still concerning - upgrade to moderate
                if severity == "mild":
                    new_severity = "moderate"
                additional_warnings.append(
                    "⚠️ CAUTION: You use pesticides. Respiratory symptoms may indicate "
                    "chemical irritation. Wear protective masks during spraying. "
                    "Monitor symptoms closely. See doctor if symptoms worsen."
                )

        # HIGH: Dust exposure + chronic cough
        if dust_exposure:
            if classification == "chronic" or (classification in ["dry", "wet"] and confidence > 0.7):
                # Possible Farmer's Lung disease
                if severity == "mild":
                    new_severity = "moderate"
                additional_warnings.append(
                    "⚠️ WARNING: Dust exposure with chronic cough may indicate "
                    "Farmer's Lung (hypersensitivity pneumonitis). Wear a dust mask "
                    "when handling grain, hay, or crops. See a pulmonologist for evaluation."
                )

        # Combine original recommendation with occupational warnings
        if additional_warnings:
            adjusted_recommendation = "\n\n".join([recommendation] + additional_warnings)
        else:
            adjusted_recommendation = recommendation

        return new_severity, adjusted_recommendation


# Singleton instance
_classifier = None


def get_classifier() -> CoughClassifier:
    """Get singleton classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = CoughClassifier()
        _classifier.load()
    return _classifier
