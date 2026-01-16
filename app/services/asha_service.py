"""
ASHA Service
Handles logic for Accredited Social Health Activist (ASHA) worker interactions.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class AshaService:
    """
    Service for ASHA Worker features:
    1. Patient ID management (simple pass-through for now)
    2. Specific guidance/scripts for health workers
    """

    def get_screening_feedback(self, risk_level: str, referral_code: Optional[str] = None) -> str:
        """
        Get the voice response text for the ASHA worker based on the screening result.
        """
        risk_high = risk_level in ["moderate", "high", "severe", "urgent"]
        
        if risk_high:
            if referral_code:
                return (
                    f"URGENT: High risk detected. Priority ticket {referral_code} sent to patient. "
                    "Please ensure patient visits the clinic today."
                )
            else:
                return (
                    f"High risk detected. Report sent to patient. Please follow up to ensure doctor visit."
                )
        else:
            return (
                f"Screening complete. Risk level is {risk_level}. Report sent to patient."
            )

# Singleton
_asha_service = None

def get_asha_service() -> AshaService:
    global _asha_service
    if _asha_service is None:
        _asha_service = AshaService()
    return _asha_service
