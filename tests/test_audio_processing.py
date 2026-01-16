"""
Tests for Audio Processing Utilities
"""
import pytest
from pathlib import Path
import numpy as np


class TestAudioProcessing:
    """Tests for audio processing utilities"""
    
    def test_load_audio(self, sample_audio_path: Path):
        """Test loading audio file"""
        from app.utils.audio_processing import load_audio
        
        audio, sr = load_audio(str(sample_audio_path))
        
        assert audio is not None
        assert len(audio) > 0
        assert sr == 16000  # Default sample rate
    
    def test_normalize_audio(self, sample_audio_path: Path):
        """Test audio normalization"""
        from app.utils.audio_processing import load_audio, normalize_audio
        
        audio, sr = load_audio(str(sample_audio_path))
        normalized = normalize_audio(audio)
        
        assert np.max(np.abs(normalized)) <= 1.0
    
    def test_extract_audio_duration(self, sample_audio_path: Path):
        """Test audio duration extraction"""
        from app.utils.audio_processing import get_audio_duration
        
        duration = get_audio_duration(str(sample_audio_path))
        
        assert duration is not None
        assert duration > 0
        assert duration <= 10  # Test audio is 2 seconds


class TestAudioQuality:
    """Tests for audio quality assessment"""
    
    def test_quality_score(self, sample_audio_path: Path):
        """Test audio quality scoring"""
        from app.utils.audio_quality import AudioQualityChecker
        
        checker = AudioQualityChecker()
        result = checker.check_quality(str(sample_audio_path))
        
        assert result is not None
        assert "overall_score" in result
        assert 0 <= result["overall_score"] <= 1
    
    def test_quality_metrics(self, sample_audio_path: Path):
        """Test individual quality metrics"""
        from app.utils.audio_quality import AudioQualityChecker
        
        checker = AudioQualityChecker()
        result = checker.check_quality(str(sample_audio_path))
        
        # Should have various quality metrics
        assert "snr" in result or "signal_to_noise" in result or "quality_score" in result


class TestFeatureExtractor:
    """Tests for ML feature extraction"""
    
    def test_extract_features(self, sample_audio_path: Path):
        """Test feature extraction from audio"""
        from app.ml.feature_extractor import get_feature_extractor
        
        extractor = get_feature_extractor()
        features, names = extractor.get_feature_vector(str(sample_audio_path))
        
        assert features is not None
        assert len(features) > 0
        assert len(names) == len(features)
    
    def test_mfcc_extraction(self, sample_audio_path: Path):
        """Test MFCC feature extraction"""
        from app.ml.feature_extractor import get_feature_extractor
        
        extractor = get_feature_extractor()
        features, names = extractor.get_feature_vector(str(sample_audio_path))
        
        # Should have MFCC features
        mfcc_features = [n for n in names if "mfcc" in n.lower()]
        assert len(mfcc_features) > 0


class TestVoiceBiomarkers:
    """Tests for voice biomarker extraction"""
    
    def test_biomarker_extraction(self, sample_audio_path: Path):
        """Test voice biomarker extraction"""
        from app.ml.voice_biomarkers import VoiceBiomarkerExtractor
        
        extractor = VoiceBiomarkerExtractor()
        biomarkers = extractor.extract(str(sample_audio_path))
        
        assert biomarkers is not None
        # Should have common biomarkers
        assert "energy_mean" in biomarkers or "pitch_mean" in biomarkers or "jitter" in biomarkers
