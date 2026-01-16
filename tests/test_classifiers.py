"""
Tests for ML Classifiers
"""
import pytest
from pathlib import Path
import numpy as np

# Import classifiers to check types if needed, or just duck type check
from app.ml.classifier import ClassificationResult
from app.ml.parkinsons_classifier import ParkinsonsClassificationResult
from app.ml.depression_classifier import DepressionClassificationResult

class TestCoughClassifier:
    """Tests for the cough classifier"""
    
    def test_classifier_loads(self):
        """Test that classifier loads correctly"""
        from app.ml.classifier import get_classifier
        
        classifier = get_classifier()
        assert classifier is not None
        # Accepted model types based on implementation
        assert classifier.model_type in ["sklearn_random_forest", "rule_based", "hear_embeddings", "initializing"]
    
    def test_classifier_predict(self, sample_audio_path: Path):
        """Test classifier prediction on audio"""
        from app.ml.classifier import get_classifier
        
        classifier = get_classifier()
        # API uses .classify(), not .predict()
        result = classifier.classify(str(sample_audio_path))
        
        assert result is not None
        assert isinstance(result, ClassificationResult)
        assert hasattr(result, "classification")
        assert hasattr(result, "confidence")
        assert 0 <= result.confidence <= 1


class TestParkinsonsClassifier:
    """Tests for the Parkinson's classifier"""
    
    def test_classifier_loads(self):
        """Test that PD classifier loads correctly"""
        from app.ml.parkinsons_classifier import get_parkinsons_classifier
        
        classifier = get_parkinsons_classifier()
        assert classifier is not None
        assert classifier.model_type in ["sklearn", "rule_based"]
    
    def test_classifier_predict(self, sample_audio_path: Path):
        """Test PD classifier prediction"""
        from app.ml.parkinsons_classifier import get_parkinsons_classifier
        
        classifier = get_parkinsons_classifier()
        result = classifier.classify(str(sample_audio_path))
        
        assert result is not None
        assert isinstance(result, ParkinsonsClassificationResult)
        assert hasattr(result, "risk_level")
        assert hasattr(result, "confidence")


class TestDepressionClassifier:
    """Tests for the Depression classifier"""
    
    def test_classifier_loads(self):
        """Test that depression classifier loads correctly"""
        from app.ml.depression_classifier import get_depression_classifier
        
        classifier = get_depression_classifier()
        assert classifier is not None
        assert classifier.model_type in ["sklearn", "rule_based"]
    
    def test_classifier_predict(self, sample_audio_path: Path):
        """Test depression classifier prediction"""
        from app.ml.depression_classifier import get_depression_classifier
        
        classifier = get_depression_classifier()
        result = classifier.classify(str(sample_audio_path))
        
        assert result is not None
        assert isinstance(result, DepressionClassificationResult)
        assert hasattr(result, "severity_level")
        assert hasattr(result, "confidence")
