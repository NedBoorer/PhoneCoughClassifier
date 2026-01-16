"""
Tests for WhatsApp Webhooks
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

class TestHealthCardGeneration:
    """Test health card generation logic"""
    
    @patch("app.utils.health_card_generator.create_image")
    def test_generate_card_calls_image_generator(self, mock_create_image):
        from app.utils.health_card_generator import generate_health_card
        from app.ml.model_hub import ComprehensiveHealthResult, ScreeningResult

        # Mock result
        result = ComprehensiveHealthResult(
            primary_concern="none",
            overall_risk_level="normal",
            screenings={
                "respiratory": ScreeningResult(
                    disease="respiratory",
                    detected=False,
                    confidence=0.9,
                    severity="normal",
                    details={}
                )
            },
            recommendation="Stay healthy"
        )
        
        path = generate_health_card(result, language="en")
        
        assert path is not None
        mock_create_image.assert_called_once()
        args = mock_create_image.call_args[1]
        assert args["risk_level"] == "normal"
        assert "Stay healthy" in args["details"]

@pytest.fixture
def mock_whatsapp_service():
    with patch("app.api.whatsapp_webhooks.get_whatsapp_service") as mock:
        service = MagicMock()
        service.send_text = MagicMock(return_value=True)
        service.send_interactive = MagicMock(return_value=True)
        service.download_media = AsyncMock(return_value=True)
        service.convert_ogg_to_wav = MagicMock(return_value="/tmp/test.wav")
        mock.return_value = service
        yield service

@pytest.fixture
def mock_channel_service():
    with patch("app.api.whatsapp_webhooks.get_channel_service") as mock:
        service = MagicMock()
        session = MagicMock()
        session.language = "en"
        session.state = "initial"
        service.get_or_create_session.return_value = session
        mock.return_value = service
        yield service

class TestWhatsAppIncoming:
    """Tests for WhatsApp incoming messages"""
    
    def test_text_help_returns_interactive(self, test_client, mock_whatsapp_service, mock_channel_service):
        """Test 'help' command returns interactive menu"""
        data = {
            "MessageSid": "SM123",
            "From": "whatsapp:+1234567890",
            "Body": "help",
            "NumMedia": "0"
        }
        
        response = test_client.post("/whatsapp/incoming", data=data)
        
        assert response.status_code == 200
        mock_whatsapp_service.send_interactive.assert_called_once()
        
    def test_language_selection(self, test_client, mock_whatsapp_service, mock_channel_service):
        """Test selecting language updates session"""
        data = {
            "MessageSid": "SM123",
            "From": "whatsapp:+1234567890",
            "Body": "1",  # English
            "NumMedia": "0"
        }
        
        response = test_client.post("/whatsapp/incoming", data=data)
        
        assert response.status_code == 200
        # Should confirm language
        args, _ = mock_whatsapp_service.send_text.call_args
        assert "English" in args[1]

    @patch("app.api.whatsapp_webhooks.get_model_hub")
    def test_voice_note_processing(self, mock_hub, test_client, mock_whatsapp_service, mock_channel_service):
        """Test handling of voice note (audio message)"""
        # Mock ML analysis
        mock_result = MagicMock()
        mock_result.recommendation = "Consult a doctor"
        mock_hub.return_value.run_full_analysis_async = AsyncMock(return_value=mock_result)
        
        data = {
            "MessageSid": "SM123",
            "From": "whatsapp:+1234567890",
            "NumMedia": "1",
            "MediaContentType0": "audio/ogg",
            "MediaUrl0": "http://example.com/voice.ogg"
        }
        
        response = test_client.post("/whatsapp/incoming", data=data)
        
        assert response.status_code == 200
        # Should send processing message
        mock_whatsapp_service.send_text.assert_called()
        # Background task processing isn't easily tested with TestClient without explicit execution
        # but we verify the endpoint accepted the request
