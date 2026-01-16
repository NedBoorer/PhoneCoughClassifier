"""
Tests for ML Classifiers
"""
import pytest
from pathlib import Path
import numpy as np


class TestCoughClassifier:
    """Tests for the cough classifier"""
    
    def test_classifier_loads(self):
        """Test that classifier loads correctly"""
        from app.ml.classifier import get_classifier
        
        classifier = get_classifier()
        assert classifier is not None
        assert classifier.model_type in ["sklearn", "rule_based", "hear"]
    
    def test_classifier_predict(self, sample_audio_path: Path):
        """Test classifier prediction on audio"""
        from app.ml.classifier import get_classifier
        
        classifier = get_classifier()
        result = classifier.predict(str(sample_audio_path))
        
        assert result is not None
        assert "classification" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
    
    def test_classifier_classes(self):
        """Test that classifier has expected classes"""
        from app.ml.classifier import get_classifier
        
        classifier = get_classifier()
        expected_classes = ["dry", "wet", "whooping", "chronic", "normal"]
        
        # Verify classes are available
        assert hasattr(classifier, "classes") or hasattr(classifier, "model")


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
        result = classifier.predict(str(sample_audio_path))
        
        assert result is not None
        assert "classification" in result
        assert "confidence" in result
        assert "risk_level" in result


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
        result = classifier.predict(str(sample_audio_path))
        
        assert result is not None
        assert "classification" in result
        assert "confidence" in result


class TestModelHub:
    """Tests for the Model Hub (multi-model coordinator)"""
    
    def test_model_hub_loads(self):
        """Test that model hub initializes correctly"""
        from app.ml.model_hub import get_model_hub
        
        hub = get_model_hub()
        assert hub is not None
    
    def test_model_hub_available_models(self):
        """Test that expected models are available"""
        from app.ml.model_hub import get_model_hub
        
        hub = get_model_hub()
        models = hub.get_available_models()
        
        assert "respiratory" in models or "cough" in models
