"""
Market Service (Mandi Bol)
Handles market price checks and farmer wellness interventions.
"""
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

from app.config import settings
from app.services.twilio_service import get_twilio_service
from app.ml.model_hub import get_model_hub
from app.services.kisan_mitra_service import get_kisan_mitra_service

logger = logging.getLogger(__name__)

class MarketService:
    """
    Service for 'Mandi Bol' feature:
    1. Market Price Information
    2. Wellness/Mental Health Screening (Passive)
    """

    def __init__(self):
        self.kisan_service = get_kisan_mitra_service()

    def get_price(self, commodity: str) -> str:
        """Get market price info for a commodity"""
        return self.kisan_service.get_market_price(commodity)

    async def process_wellness_check(self, call_sid: str, caller_number: str, recording_url: str):
        """
        Background task: Analyze wellness check recording and intervene if needed.
        """
        try:
            # Download recording
            twilio_service = get_twilio_service()
            local_path = settings.recordings_dir / f"{call_sid}_wellness.wav"
            await twilio_service.download_recording(recording_url, str(local_path))

            # Run depression screening
            hub = get_model_hub()
            result = await hub.run_full_analysis_async(
                str(local_path),
                enable_respiratory=False,
                enable_parkinsons=False,
                enable_depression=True
            )

            # Check if intervention needed
            depression_screening = result.screenings.get("depression")
            should_intervene, reason = self.kisan_service.check_intervention_needed(depression_screening)

            if should_intervene:
                # Send SMS with counselor helpline
                intervention_msg = (
                    f"Namaste. We noticed you may be going through a difficult time. "
                    f"You are not alone. Please call Kisan Call Center at 1800-180-1551 for free support. "
                    f"Your well-being matters. - Mandi Bol Farmer Wellness"
                )
                twilio_service.send_sms(caller_number, intervention_msg)
                logger.info(f"Wellness intervention SMS sent to {caller_number}: {reason}")
            
        except Exception as e:
            logger.error(f"Wellness check processing failed for {call_sid}: {e}")

    async def analyze_and_get_price(self, call_sid: str, caller_number: str, recording_url: str, commodity: str) -> Tuple[str, bool, str]:
        """
        Analyze voice for depression (passive) and return price info.
        Returns: (price_info, should_intervene, intervention_reason)
        """
        try:
            twilio_service = get_twilio_service()
            local_path = settings.recordings_dir / f"{call_sid}_market.wav"
            await twilio_service.download_recording(recording_url, str(local_path))
            
            hub = get_model_hub()
            # Enable Depression Screening!
            result = await hub.run_full_analysis_async(
                str(local_path),
                enable_respiratory=False,
                enable_parkinsons=False,
                enable_depression=True 
            )
            
            # Check Intervention
            depression_screening = result.screenings.get("depression")
            should_intervene, reason = self.kisan_service.check_intervention_needed(depression_screening)
            
            if should_intervene:
                return "", True, reason
            
            # Normal Market Flow
            price_info = self.get_price(commodity)
            return price_info, False, ""
            
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            raise e

# Singleton
_market_service = None

def get_market_service() -> MarketService:
    global _market_service
    if _market_service is None:
        _market_service = MarketService()
    return _market_service
