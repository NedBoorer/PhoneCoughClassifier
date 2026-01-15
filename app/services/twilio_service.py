"""
Phone Cough Classifier - Twilio Service
Handles SMS, voice response, and audio downloads
"""
import logging
from typing import Optional
from pathlib import Path
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class TwilioService:
    """Twilio helper for SMS and voice operations"""
    
    def __init__(self):
        self._client = None
    
    @property
    def client(self):
        """Lazy load Twilio client"""
        if self._client is None:
            from twilio.rest import Client
            self._client = Client(
                settings.twilio_account_sid,
                settings.twilio_auth_token
            )
        return self._client
    
    def send_sms(
        self,
        to: str,
        message: str,
        from_number: Optional[str] = None
    ) -> bool:
        """
        Send SMS message.
        
        Args:
            to: Recipient phone number (E.164 format)
            message: Message text
            from_number: Sender number (defaults to configured number)
            
        Returns:
            True if sent successfully
        """
        try:
            from_number = from_number or settings.twilio_phone_number
            
            result = self.client.messages.create(
                body=message,
                from_=from_number,
                to=to
            )
            
            logger.info(f"SMS sent to {to}: SID={result.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send SMS to {to}: {e}")
            return False
    
    async def download_recording(
        self,
        recording_url: str,
        output_path: str
    ) -> bool:
        """
        Download recording from Twilio.
        
        Args:
            recording_url: Twilio recording URL
            output_path: Local path to save recording
            
        Returns:
            True if downloaded successfully
        """
        try:
            # Twilio recordings require auth
            auth = (settings.twilio_account_sid, settings.twilio_auth_token)
            
            # Add .wav extension if not present
            if not recording_url.endswith('.wav'):
                recording_url = f"{recording_url}.wav"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    recording_url,
                    auth=auth,
                    follow_redirects=True
                )
                response.raise_for_status()
                
                # Save to file
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_bytes(response.content)
            
            logger.info(f"Downloaded recording to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download recording: {e}")
            return False
    
    def validate_webhook_signature(
        self,
        signature: str,
        url: str,
        params: dict
    ) -> bool:
        """Validate Twilio webhook signature for security"""
        try:
            from twilio.request_validator import RequestValidator
            
            validator = RequestValidator(settings.twilio_auth_token)
            return validator.validate(url, params, signature)
            
        except Exception as e:
            logger.error(f"Signature validation failed: {e}")
            return False


def format_sms_result(
    classification: str,
    confidence: float,
    recommendation: str,
    language: str = "en"
) -> str:
    """
    Format classification result as SMS message.
    
    Args:
        classification: Cough classification type
        confidence: Confidence score (0-1)
        recommendation: Health recommendation
        language: Language code
        
    Returns:
        Formatted SMS message
    """
    # Classification display names
    class_names = {
        "en": {
            "dry": "Dry (Non-productive)",
            "wet": "Wet (Productive)",
            "whooping": "Whooping/Barking",
            "chronic": "Chronic",
            "normal": "Normal (Acute)"
        },
        "hi": {
            "dry": "सूखी खांसी",
            "wet": "बलगम वाली खांसी",
            "whooping": "काली खांसी",
            "chronic": "पुरानी खांसी",
            "normal": "सामान्य खांसी"
        }
    }
    
    # Get display name
    names = class_names.get(language, class_names["en"])
    class_display = names.get(classification, classification.title())
    
    # Format message
    confidence_pct = int(confidence * 100)
    
    if language == "hi":
        message = f"""🩺 खांसी विश्लेषण परिणाम

वर्गीकरण: {class_display}
विश्वास स्तर: {confidence_pct}%

{recommendation}

यह AI विश्लेषण है। गंभीर लक्षणों के लिए डॉक्टर से मिलें।
हेल्पलाइन: 108"""
    else:
        message = f"""🩺 Cough Analysis Result

Classification: {class_display}
Confidence: {confidence_pct}%

{recommendation}

This is an AI analysis. For serious symptoms, please consult a doctor.
Health Helpline (India): 108"""
    
    return message


# Singleton instance
_service = None


def get_twilio_service() -> TwilioService:
    """Get singleton Twilio service"""
    global _service
    if _service is None:
        _service = TwilioService()
    return _service
