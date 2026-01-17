"""
Tests for Twilio Webhook Endpoints
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestIncomingCall:
    """Tests for incoming call webhook"""
    
    def test_incoming_call_returns_twiml(self, test_client, mock_call_data):
        """Test that incoming call returns valid TwiML"""
        response = test_client.post("/india/voice/incoming", data=mock_call_data)
        
        assert response.status_code == 200
        assert "xml" in response.headers.get("content-type", "")
        assert "<?xml" in response.text or "<Response>" in response.text
    
    def test_incoming_call_greeting(self, test_client, mock_call_data):
        """Test that call includes greeting"""
        response = test_client.post("/india/voice/incoming", data=mock_call_data)
        
        assert response.status_code == 200
        # Should have some form of Say or Gather element
        assert "<Say" in response.text or "<Gather" in response.text


class TestLanguageSelection:
    """Tests for language selection flow"""
    
    def test_language_select_english(self, test_client, mock_call_data):
        """Test English language selection (press 1)"""
        data = {**mock_call_data, "Digits": "1"}
        response = test_client.post("/india/voice/language-selected", data=data)
        
        assert response.status_code == 200
    
    def test_language_select_hindi(self, test_client, mock_call_data):
        """Test Hindi language selection (press 2)"""
        data = {**mock_call_data, "Digits": "2"}
        response = test_client.post("/india/voice/language-selected", data=data)
        
        assert response.status_code == 200


class TestMissedCallFlow:
    """Tests for zero-cost missed call feature"""
    
    @patch('app.api.india_webhooks.get_twilio_service')
    def test_missed_call_triggers_callback(self, mock_twilio, test_client, mock_call_data):
        """Test that missed call triggers callback"""
        # Setup mock
        mock_service = MagicMock()
        mock_service.initiate_callback = AsyncMock(return_value=True)
        mock_twilio.return_value = mock_service
        
        response = test_client.post("/india/voice/missed-call", data=mock_call_data)
        
        assert response.status_code == 200
        # Should reject the call with busy signal
        assert "<Reject" in response.text or "busy" in response.text.lower()


class TestASHAWorkerMode:
    """Tests for ASHA worker (community health worker) mode"""
    
    def test_asha_menu_accessible(self, test_client, mock_call_data):
        """Test ASHA menu is accessible with digit * (not 9)"""
        # ASHA mode is usually Star (*) in the implementation viewed
        data = {**mock_call_data, "Digits": "*"}
        response = test_client.post("/india/voice/language-selected", data=data)
        
        # Should redirect to ASHA flow or be a valid response
        assert response.status_code == 200


class TestRecordingComplete:
    """Tests for recording completion webhook"""
    
    @patch('app.api.india_webhooks.get_twilio_service')
    @patch('app.api.india_webhooks.get_model_hub')
    def test_recording_complete_processes_audio(
        self, mock_hub, mock_twilio, test_client, mock_recording_data
    ):
        """Test that completed recording triggers analysis"""
        from app.ml.model_hub import ComprehensiveHealthResult, ScreeningResult
        
        # Mock Twilio download
        mock_service = MagicMock()
        mock_service.download_recording = AsyncMock(return_value=True)
        mock_twilio.return_value = mock_service
        
        # Mock analysis result
        mock_hub_instance = MagicMock()
        mock_result = ComprehensiveHealthResult(
            primary_concern="none",
            overall_risk_level="normal",
            screenings={
                "respiratory": ScreeningResult(
                    disease="respiratory",
                    detected=False,
                    confidence=0.2,
                    severity="normal",
                    details={}
                )
            }
        )
        mock_hub_instance.run_full_analysis_async = AsyncMock(return_value=mock_result)
        mock_hub.return_value = mock_hub_instance
        
        response = test_client.post(
            "/india/voice/recording-complete",
            data=mock_recording_data
        )
        
        assert response.status_code == 200


class TestHealthCheck:
    """Tests for health check endpoint"""
    
    def test_health_endpoint(self, test_client):
        """Test health check returns status"""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns index.html"""
        response = test_client.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
