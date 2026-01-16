"""
WhatsApp Service
Handles WhatsApp messaging, voice notes, and interactive messages
"""
import logging
import subprocess
from pathlib import Path
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class WhatsAppService:
    """WhatsApp messaging and media handling service"""
    
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
    
    def _normalize_whatsapp_number(self, number: str) -> str:
        """Ensure number has whatsapp: prefix"""
        if not number.startswith('whatsapp:'):
            return f"whatsapp:{number}"
        return number
    
    def send_text(
        self,
        to: str,
        message: str,
        from_number: Optional[str] = None
    ) -> bool:
        """
        Send a text message via WhatsApp.
        
        Args:
            to: Recipient phone number (E.164 format)
            message: Message text
            from_number: Sender number (defaults to configured WhatsApp number)
            
        Returns:
            True if sent successfully
        """
        try:
            to = self._normalize_whatsapp_number(to)
            from_number = from_number or settings.twilio_whatsapp_from
            from_number = self._normalize_whatsapp_number(from_number)
            
            result = self.client.messages.create(
                body=message,
                from_=from_number,
                to=to
            )
            
            logger.info(f"WhatsApp text sent to {to}: SID={result.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send WhatsApp to {to}: {e}")
            return False
    
    def send_interactive(
        self,
        to: str,
        body: str,
        buttons: list[dict],
        header: Optional[str] = None,
        footer: Optional[str] = None
    ) -> bool:
        """
        Send an interactive button message via WhatsApp.
        
        Args:
            to: Recipient phone number
            body: Main message body
            buttons: List of buttons, each with 'id' and 'title' keys
            header: Optional header text
            footer: Optional footer text
            
        Returns:
            True if sent successfully
        """
        try:
            to = self._normalize_whatsapp_number(to)
            from_number = self._normalize_whatsapp_number(settings.twilio_whatsapp_from)
            
            # Build interactive content JSON
            # Twilio uses ContentSid for templates, or we use a simple button approach
            # For sandbox, we use quick reply buttons in the body text
            
            # Format buttons as inline options for sandbox compatibility
            button_text = "\n".join([
                f"• Reply '{btn['id']}' for {btn['title']}"
                for btn in buttons[:3]  # WhatsApp max 3 buttons
            ])
            
            full_message = f"{body}\n\n{button_text}"
            if footer:
                full_message += f"\n\n{footer}"
            
            result = self.client.messages.create(
                body=full_message,
                from_=from_number,
                to=to
            )
            
            logger.info(f"WhatsApp interactive sent to {to}: SID={result.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send WhatsApp interactive to {to}: {e}")
            return False
    
    def send_media(
        self,
        to: str,
        message: str,
        media_url: str
    ) -> bool:
        """
        Send a message with media (image/audio) via WhatsApp.
        
        Args:
            to: Recipient phone number
            message: Caption text
            media_url: Public URL of the media file
            
        Returns:
            True if sent successfully
        """
        try:
            to = self._normalize_whatsapp_number(to)
            from_number = self._normalize_whatsapp_number(settings.twilio_whatsapp_from)
            
            result = self.client.messages.create(
                body=message,
                from_=from_number,
                to=to,
                media_url=[media_url]
            )
            
            logger.info(f"WhatsApp media sent to {to}: SID={result.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send WhatsApp media to {to}: {e}")
            return False
    
    async def download_media(
        self,
        media_url: str,
        output_path: str
    ) -> bool:
        """
        Download media from Twilio WhatsApp message.
        
        Args:
            media_url: Twilio media URL
            output_path: Local path to save the file
            
        Returns:
            True if downloaded successfully
        """
        try:
            auth = (settings.twilio_account_sid, settings.twilio_auth_token)
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    media_url,
                    auth=auth,
                    follow_redirects=True
                )
                response.raise_for_status()
                
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_bytes(response.content)
            
            logger.info(f"Downloaded WhatsApp media to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download WhatsApp media: {e}")
            return False
    
    def convert_ogg_to_wav(
        self,
        ogg_path: str,
        wav_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Convert OGG Opus (WhatsApp voice note format) to WAV.
        
        Args:
            ogg_path: Path to the OGG file
            wav_path: Optional output path (defaults to same name with .wav)
            
        Returns:
            Path to the WAV file, or None if conversion failed
        """
        try:
            ogg_file = Path(ogg_path)
            if not wav_path:
                wav_path = str(ogg_file.with_suffix('.wav'))
            
            # Use ffmpeg for conversion
            result = subprocess.run(
                [
                    'ffmpeg', '-y',  # Overwrite output
                    '-i', str(ogg_file),
                    '-ar', str(settings.audio_sample_rate),  # 16kHz
                    '-ac', '1',  # Mono
                    '-sample_fmt', 's16',  # 16-bit
                    wav_path
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"ffmpeg conversion failed: {result.stderr}")
                return None
            
            logger.info(f"Converted {ogg_path} to {wav_path}")
            return wav_path
            
        except FileNotFoundError:
            logger.error("ffmpeg not found. Please install ffmpeg.")
            return None
        except Exception as e:
            logger.error(f"Failed to convert OGG to WAV: {e}")
            return None
    
    def send_health_card(
        self,
        to: str,
        result: 'ComprehensiveHealthResult',
        language: str = "en"
    ) -> bool:
        """
        Generate and send a health card image via WhatsApp.
        
        Args:
            to: Recipient phone number
            result: Health screening result
            language: Language for the card
            
        Returns:
            True if sent successfully
        """
        try:
            # Generate health card image
            from app.utils.health_card_generator import generate_health_card
            
            card_path = generate_health_card(result, language)
            
            if not card_path:
                # Fallback to text message
                return self.send_text(to, result.recommendation)
            
            # Card needs to be publicly accessible
            # Use the static file server
            card_filename = Path(card_path).name
            card_url = f"{settings.base_url}/data/health_cards/{card_filename}"
            
            # Send with summary caption
            risk_emoji = {
                "low": "🟢", "normal": "🟢",
                "mild": "🟡", "moderate": "🟠",
                "high": "🔴", "severe": "🔴"
            }
            emoji = risk_emoji.get(result.overall_risk_level, "⚪")
            
            caption = f"{emoji} Your Voice Health Report\n\n{result.recommendation}"
            
            return self.send_media(to, caption, card_url)
            
        except Exception as e:
            logger.error(f"Failed to send health card: {e}")
            # Fallback to text
            return self.send_text(to, result.recommendation)


# Singleton instance
_service: Optional[WhatsAppService] = None


def get_whatsapp_service() -> WhatsAppService:
    """Get singleton WhatsApp service instance"""
    global _service
    if _service is None:
        _service = WhatsAppService()
    return _service
