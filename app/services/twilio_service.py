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
    
    def send_whatsapp(
        self,
        to: str,
        message: str,
        from_number: Optional[str] = None
    ) -> bool:
        """
        Send a WhatsApp text message.
        
        Args:
            to: Recipient phone number (E.164 format)
            message: Message text
            from_number: Sender WhatsApp number (defaults to configured number)
            
        Returns:
            True if sent successfully
        """
        try:
            # Twilio WhatsApp numbers must be prefixed with 'whatsapp:'
            if not to.startswith('whatsapp:'):
                to = f"whatsapp:{to}"
            
            from_number = from_number or settings.twilio_whatsapp_from
            if not from_number.startswith('whatsapp:'):
                from_number = f"whatsapp:{from_number}"
                
            result = self.client.messages.create(
                body=message,
                from_=from_number,
                to=to
            )
            logger.info(f"WhatsApp sent to {to}: SID={result.sid}")
            return True
        except Exception as e:
            logger.error(f"Failed to send WhatsApp to {to}: {e}")
            return False

    def send_whatsapp_with_media(
        self,
        to: str,
        message: str,
        media_url: str
    ) -> bool:
        """
        Send WhatsApp message with media (image).
        Requires Twilio WhatsApp sender to be configured.
        """
        try:
            # Twilio WhatsApp numbers must be prefixed with 'whatsapp:'
            if not to.startswith('whatsapp:'):
                to = f"whatsapp:{to}"
            
            from_number = settings.twilio_whatsapp_from  # Fixed: use WhatsApp sender
            if not from_number.startswith('whatsapp:'):
                from_number = f"whatsapp:{from_number}"
                
            result = self.client.messages.create(
                body=message,
                from_=from_number,
                to=to,
                media_url=[media_url]
            )
            logger.info(f"WhatsApp sent to {to}: SID={result.sid}")
            return True
        except Exception as e:
            logger.error(f"Failed to send WhatsApp to {to}: {e}")
            return False

    def trigger_outbound_call(
        self,
        to: str,
        callback_url: str
    ) -> bool:
        """
        Initiate an outbound call (callback).
        """
        try:
            call = self.client.calls.create(
                to=to,
                from_=settings.twilio_phone_number,
                url=callback_url
            )
            logger.info(f"Outbound call initiated to {to}: SID={call.sid}")
            return True
        except Exception as e:
            logger.error(f"Failed to call {to}: {e}")
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
    language: str = "en",
    comprehensive_result: Optional['ComprehensiveHealthResult'] = None
) -> str:
    """
    Format classification result as SMS message.
    """
    if comprehensive_result:
        # Detailed multi-disease report
        
        # 1. Summary
        risk_emoji = {
            "low": "🟢", "normal": "🟢", 
            "mild": "🟡", "moderate": "🟠", 
            "high": "🔴", "severe": "🔴", "urgent": "🚨"
        }
        emoji = risk_emoji.get(comprehensive_result.overall_risk_level, "⚪")
        
        message = f"🩺 Voice Health Report {emoji}\n\n"
        
        # 2. Respiratory
        resp = comprehensive_result.screenings.get("respiratory")
        if resp and resp.detected:
            message += f"🫁 Respiratory: {resp.severity.upper()} risk\n"
            message += f"   Indication: {resp.details.get('sound_class', 'unknown')}\n"
        else:
            message += "🫁 Respiratory: Normal\n"
            
        # 3. Parkinson's (if enabled/run)
        pd = comprehensive_result.screenings.get("parkinsons")
        if pd:
            if pd.detected:
                 message += f"🧠 Voice Tremor: {pd.severity.upper()} risk\n"
            else:
                 message += "🧠 Voice Tremor: Normal\n"

        # 4. Depression (if enabled/run)
        dep = comprehensive_result.screenings.get("depression")
        if dep:
            if dep.detected:
                message += f"🎭 Mood Indicators: {dep.severity.upper()} risk\n"
            else:
                message += "🎭 Mood Indicators: Normal\n"
        
        # 5. Recommendation
        message += f"\n💡 {comprehensive_result.recommendation}\n"
        message += "\nThis is an AI screening. Consult a doctor for diagnosis."
        
        return message

    # Dynamic Multi-language SMS
    from app.utils.i18n import get_text, get_class_name
    
    # Get translated components
    title = get_text("result_title", language)
    lbl_class = get_text("label_classification", language)
    lbl_conf = get_text("label_confidence", language)
    class_name = get_class_name(classification, language)
    disclaimer = get_text("disclaimer", language)
    helpline = get_text("helpline", language)
    
    confidence_pct = int(confidence * 100)
    
    message = f"""{title}

{lbl_class}: {class_name}
{lbl_conf}: {confidence_pct}%

{recommendation}

{disclaimer}
{helpline}"""
    
    return message


def format_medical_report(
    comprehensive_result: 'ComprehensiveHealthResult',
    patient_phone: str,
    report_type: str = "brief"  # "brief" or "detailed"
) -> str:
    """
    Format a medical report for doctors.
    
    Args:
        comprehensive_result: Health screening result
        patient_phone: Patient phone number
        report_type: "brief" for SMS-friendly, "detailed" for printable
        
    Returns:
        Formatted medical report
    """
    from app.utils.medical_report import (
        generate_medical_report,
        generate_short_medical_summary,
        generate_printable_report
    )
    
    if report_type == "brief":
        return generate_medical_report(
            result=comprehensive_result,
            patient_phone=patient_phone,
            language="en",
            include_biomarkers=True
        )
    elif report_type == "summary":
        return generate_short_medical_summary(
            result=comprehensive_result,
            max_length=500
        )
    else:  # detailed
        return generate_printable_report(
            result=comprehensive_result,
            patient_phone=patient_phone
        )


# Singleton instance
_service = None


def get_twilio_service() -> TwilioService:
    """Get singleton Twilio service"""
    global _service
    if _service is None:
        _service = TwilioService()
    return _service
