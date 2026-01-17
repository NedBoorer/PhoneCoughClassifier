"""
PyTest Configuration and Shared Fixtures
"""
import pytest
import asyncio
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, AsyncMock

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_audio_path(tmp_path: Path) -> Path:
    """Create a sample WAV file for testing"""
    import numpy as np
    import soundfile as sf
    
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate simple synthetic audio
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    
    audio_file = tmp_path / "test_audio.wav"
    sf.write(audio_file, audio, sample_rate)
    
    return audio_file


@pytest.fixture
def mock_twilio_client():
    """Mock Twilio client"""
    mock = MagicMock()
    mock.calls.create = MagicMock(return_value=MagicMock(sid="CAtest123"))
    mock.messages.create = MagicMock(return_value=MagicMock(sid="SMtest123"))
    return mock


@pytest.fixture
def mock_recording_url():
    """Mock Twilio recording URL"""
    return "https://api.twilio.com/2010-04-01/Accounts/test/Recordings/REtest.wav"


@pytest.fixture
def test_client():
    """Create FastAPI test client"""
    from fastapi.testclient import TestClient
    from app.main import app
    
    return TestClient(app)


@pytest.fixture
def mock_call_data():
    """Sample Twilio call webhook data"""
    return {
        "CallSid": "CAtest123456789",
        "From": "+919876543210",
        "To": "+911234567890",
        "CallStatus": "ringing",
        "Direction": "inbound"
    }


@pytest.fixture
def mock_recording_data():
    """Sample Twilio recording webhook data"""
    return {
        "CallSid": "CAtest123456789",
        "From": "+919876543210",
        "RecordingUrl": "https://api.twilio.com/2010-04-01/Accounts/test/Recordings/REtest",
        "RecordingDuration": "5",
        "RecordingStatus": "completed"
    }
