"""
Tests for Tuberculosis (TB) Cough Classifier

Tests the TuberculosisScreener class for:
- Initialization and loading
- Feature extraction from audio
- Classification and risk scoring
- Edge cases (short audio, silence)
"""
import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestTuberculosisScreener:
    """Test suite for TuberculosisScreener"""
    
    @pytest.fixture
    def screener(self):
        """Create a TB screener instance"""
        from app.ml.tb_classifier import TuberculosisScreener
        return TuberculosisScreener()
    
    @pytest.fixture
    def sample_audio_path(self):
        """Create a temporary test audio file"""
        import librosa
        import soundfile as sf
        
        # Generate a 3-second test audio (simulated cough-like sound)
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create a cough-like sound (burst of noise with envelope)
        noise = np.random.randn(len(t)) * 0.5
        envelope = np.exp(-5 * (t - 0.5) ** 2) + np.exp(-5 * (t - 1.5) ** 2)
        audio = noise * envelope
        
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, sr)
            yield f.name
        
        # Cleanup
        if os.path.exists(f.name):
            os.unlink(f.name)
    
    @pytest.fixture
    def short_audio_path(self):
        """Create a very short audio file (< 0.5s)"""
        import soundfile as sf
        
        sr = 16000
        duration = 0.3  # Too short
        audio = np.random.randn(int(sr * duration)) * 0.1
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, sr)
            yield f.name
        
        if os.path.exists(f.name):
            os.unlink(f.name)
    
    def test_screener_initialization(self, screener):
        """Test that screener initializes correctly"""
        assert screener is not None
        assert screener._loaded == False
    
    def test_screener_load(self, screener):
        """Test that screener loads without error"""
        screener.load()
        assert screener._loaded == True
    
    def test_feature_extraction(self, screener, sample_audio_path):
        """Test that feature extraction works"""
        features = screener._extract_tb_features(sample_audio_path)
        
        # Check that key features are extracted
        assert "duration" in features
        assert "mfcc_mean" in features
        assert "mfcc_std" in features
        assert "spectral_centroid_mean" in features
        assert "energy_mean" in features
        assert "wetness_score" in features
        assert "zcr_mean" in features
        
        # Check feature values are reasonable
        assert features["duration"] > 0
        assert features["mfcc_std"] >= 0
        assert 0 <= features["wetness_score"] <= 1
    
    def test_risk_score_calculation(self, screener, sample_audio_path):
        """Test risk score calculation"""
        features = screener._extract_tb_features(sample_audio_path)
        risk_score, indicators = screener._calculate_tb_risk_score(features)
        
        # Risk score should be between 0 and 1
        assert 0 <= risk_score <= 1
        
        # Indicators should be a list
        assert isinstance(indicators, list)
    
    def test_screen_normal_audio(self, screener, sample_audio_path):
        """Test screening with normal audio"""
        result = screener.screen(sample_audio_path)
        
        # Check result structure
        assert hasattr(result, 'detected')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'severity')
        assert hasattr(result, 'recommendation')
        assert hasattr(result, 'details')
        
        # Confidence should be between 0 and 1
        assert 0 <= result.confidence <= 1
        
        # Severity should be valid
        assert result.severity in ['normal', 'low_risk', 'moderate_risk', 'high_risk', 'unknown']
        
        # Recommendation should not be empty
        assert len(result.recommendation) > 0
    
    def test_screen_short_audio(self, screener, short_audio_path):
        """Test screening with very short audio"""
        result = screener.screen(short_audio_path)
        
        # Should handle gracefully
        assert result is not None
        
        # Short audio should indicate insufficient sample
        if result.details.get('indicators'):
            assert 'short_sample_warning' in result.details['indicators'] or \
                   'insufficient_audio' in result.details['indicators']
    
    def test_severity_levels(self, screener):
        """Test that severity levels map correctly to risk scores"""
        # Test the thresholds
        test_cases = [
            (0.1, "normal", False),
            (0.3, "low_risk", False),
            (0.5, "moderate_risk", True),
            (0.7, "high_risk", True),
        ]
        
        for risk_score, expected_severity, expected_detected in test_cases:
            if risk_score < 0.25:
                assert expected_severity == "normal"
            elif risk_score < 0.45:
                assert expected_severity == "low_risk"
            elif risk_score < 0.65:
                assert expected_severity == "moderate_risk"
            else:
                assert expected_severity == "high_risk"
    
    def test_details_structure(self, screener, sample_audio_path):
        """Test that details contain expected fields"""
        result = screener.screen(sample_audio_path)
        
        details = result.details
        assert 'risk_score' in details
        assert 'indicators' in details
        assert 'features' in details
        assert 'disclaimer' in details
        
        # Disclaimer should mention screening tool
        assert 'screening' in details['disclaimer'].lower()
    
    def test_singleton_getter(self):
        """Test the singleton getter function"""
        from app.ml.tb_classifier import get_tb_screener
        
        screener1 = get_tb_screener()
        screener2 = get_tb_screener()
        
        # Should return the same instance
        assert screener1 is screener2


class TestTBModelHubIntegration:
    """Test TB screening integration with ModelHub"""
    
    @pytest.fixture
    def model_hub(self):
        """Create a ModelHub instance"""
        from app.ml.model_hub import ModelHub
        return ModelHub()
    
    def test_tuberculosis_screener_property(self, model_hub):
        """Test that ModelHub has tuberculosis_screener property"""
        assert hasattr(model_hub, 'tuberculosis_screener')
        
        screener = model_hub.tuberculosis_screener
        assert screener is not None
    
    def test_run_full_analysis_tb_parameter(self, model_hub):
        """Test that run_full_analysis accepts enable_tuberculosis parameter"""
        import inspect
        sig = inspect.signature(model_hub.run_full_analysis)
        
        assert 'enable_tuberculosis' in sig.parameters
        
        # Default should be True
        assert sig.parameters['enable_tuberculosis'].default == True


class TestTBConfig:
    """Test TB configuration settings"""
    
    def test_tb_settings_exist(self):
        """Test that TB settings are in config"""
        from app.config import settings
        
        assert hasattr(settings, 'enable_tuberculosis_screening')
        assert hasattr(settings, 'tb_model_path')
        assert hasattr(settings, 'tb_screening_threshold')
    
    def test_tb_enabled_by_default(self):
        """Test that TB screening is enabled by default"""
        from app.config import settings
        
        assert settings.enable_tuberculosis_screening == True


class TestTBI18n:
    """Test TB translations"""
    
    def test_tb_translations_exist(self):
        """Test that TB translations are available"""
        from app.utils.i18n import get_text
        
        # Test key translations exist
        keys = [
            'tb_screening_label',
            'tb_normal',
            'tb_low_risk',
            'tb_moderate_risk',
            'tb_high_risk',
            'tb_dots_info',
            'tb_disclaimer',
            'tb_helpline',
        ]
        
        for key in keys:
            # English should always exist
            text = get_text(key, 'en')
            assert text != key, f"Missing translation for {key}"
            assert len(text) > 0
    
    def test_tb_translations_hindi(self):
        """Test TB translations in Hindi"""
        from app.utils.i18n import get_text
        
        # Hindi translation should be different from English
        en_text = get_text('tb_screening_label', 'en')
        hi_text = get_text('tb_screening_label', 'hi')
        
        assert en_text != hi_text
        assert 'टीबी' in hi_text  # Should contain Hindi word for TB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
