import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from app.api.india_webhooks import router
from app.services.kisan_mitra_service import KisanMitraService
from app.ml.model_hub import ComprehensiveHealthResult, ScreeningResult

class TestKisanManasFlow(unittest.TestCase):
    
    def setUp(self):
        self.client = TestClient(app)
        
    @patch('app.api.india_webhooks.get_twilio_service')
    @patch('app.api.india_webhooks.get_model_hub')
    def test_market_price_normal(self, mock_get_hub, mock_get_twilio):
        """Test normal market price check without intervention"""
        
        # Mock Twilio Download
        mock_twilio = MagicMock()
        mock_twilio.download_recording = AsyncMock(return_value=True)
        mock_get_twilio.return_value = mock_twilio
        
        # Mock Analysis Result (Normal)
        mock_hub = MagicMock()
        mock_result = ComprehensiveHealthResult(
            primary_concern="none",
            overall_risk_level="normal",
            screenings={
                "depression": ScreeningResult(
                    disease="depression",
                    detected=False,
                    confidence=0.1,
                    severity="normal",
                    details={"indicators": []}
                )
            }
        )
        mock_hub.run_full_analysis_async = AsyncMock(return_value=mock_result)
        mock_get_hub.return_value = mock_hub
        
        # Call Endpoint
        response = self.client.post(
            "/india/voice/market/analyze",
            data={
                "CallSid": "TEST1234",
                "RecordingUrl": "http://example.com/rec.wav"
            },
            params={"commodity": "onion"}
        )
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        xml = response.content.decode()
        
        # Should verify market price is spoken
        self.assertIn("The average price for onion", xml)
        # Should NOT redirect to handover
        self.assertNotIn("kisan-mitra/handover", xml)

    @patch('app.api.india_webhooks.get_twilio_service')
    @patch('app.api.india_webhooks.get_model_hub')
    def test_market_price_intervention(self, mock_get_hub, mock_get_twilio):
        """Test market price check WITH intervention (Depression Detected)"""
        
        # Mock Twilio
        mock_twilio = MagicMock()
        mock_twilio.download_recording = AsyncMock(return_value=True)
        mock_get_twilio.return_value = mock_twilio
        
        # Mock Analysis Result (Severe Depression)
        mock_hub = MagicMock()
        mock_result = ComprehensiveHealthResult(
            primary_concern="depression",
            overall_risk_level="severe",
            screenings={
                "depression": ScreeningResult(
                    disease="depression",
                    detected=True,
                    confidence=0.9,
                    severity="severe",
                    details={"indicators": ["monotone", "slow_speech"]}
                )
            }
        )
        mock_hub.run_full_analysis_async = AsyncMock(return_value=mock_result)
        mock_get_hub.return_value = mock_hub
        
        # Call Endpoint
        response = self.client.post(
            "/india/voice/market/analyze",
            data={
                "CallSid": "TEST5678",
                "RecordingUrl": "http://example.com/rec_sad.wav"
            },
            params={"commodity": "wheat"}
        )
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        xml = response.content.decode()
        
        # Should Redirect to Handover
        self.assertIn("<Redirect>", xml)
        self.assertIn("/india/voice/kisan-mitra/handover?reason=severe_depression_signs", xml)

if __name__ == '__main__':
    unittest.main()
