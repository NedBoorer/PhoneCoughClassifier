from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app.config import settings

client = TestClient(app)

def test_incoming_call_redirects_when_agent_enabled():
    """Test that incoming call redirects to voice agent when enabled"""
    with patch("app.config.settings.enable_voice_agent", True):
        response = client.post("/twilio/voice/incoming")
        
        assert response.status_code == 200
        # Check for Redirect verb in TwiML
        assert "<Redirect>" in response.text
        assert "voice-agent/start" in response.text

def test_incoming_call_legacy_flow_when_agent_disabled():
    """Test that incoming call uses legacy flow when agent disabled"""
    with patch("app.config.settings.enable_voice_agent", False):
        response = client.post("/twilio/voice/incoming")
        
        assert response.status_code == 200
        # Check for Say verb (legacy flow)
        assert "<Say" in response.text
        assert "Welcome to the Voice Health Screening" in response.text
        # Ensure NO redirect to voice agent
        assert "voice-agent/start" not in response.text
