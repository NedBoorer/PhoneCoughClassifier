"""
Kisan Mitra Service
Handles 'Mandi Bol' (Market Prices) and 'Kisan Manas' (Mental Health Intervention) logic.
"""
import logging
from typing import Dict, Optional, Tuple
import random

from app.ml.model_hub import ScreeningResult

logger = logging.getLogger(__name__)

class KisanMitraService:
    """
    Service for Farmer Assistance features.
    """
    
    # Mock Market Data
    MARKET_DATA = {
        "onion": {"nasik": 1200, "pune": 1150, "indore": 1100, "avg": 1150},
        "tomato": {"nasik": 800, "kolar": 850, "madanapalle": 780, "avg": 810},
        "potato": {"agra": 600, "hassan": 650, "jalpaiguri": 580, "avg": 610},
        "wheat": {"khanna": 2100, "kota": 2050, "rajkot": 2150, "avg": 2100},
        "rice": {"karnal": 3500, "tarn taran": 3400, "gondia": 3200, "avg": 3350}
    }
    
    COMMODITY_KEYWORDS = {
        "onion": ["onion", "pyaaz", "kanda"],
        "tomato": ["tomato", "tamatar"],
        "potato": ["potato", "aloo", "batata"],
        "wheat": ["wheat", "gehu", "kanak"],
        "rice": ["rice", "chawal", "dhan"]
    }

    def get_market_price(self, query_text: str) -> str:
        """
        Extract commodity from query and return mock price.
        Simple keyword matching for MVP.
        """
        query_text = query_text.lower()
        found_commodity = None
        
        for commodity, keywords in self.COMMODITY_KEYWORDS.items():
            if any(k in query_text for k in keywords):
                found_commodity = commodity
                break
        
        if found_commodity:
            prices = self.MARKET_DATA[found_commodity]
            avg_price = prices["avg"]
            return f"The average price for {found_commodity} today is {avg_price} rupees per quintal. Mandi prices satisfy: { ', '.join(f'{k.title()}: {v}' for k,v in prices.items() if k != 'avg') }."
        
        return "I could not understand which crop you are asking about. Please say Onion, Tomato, or Potato."

    def check_intervention_needed(self, screening_result: Optional[ScreeningResult]) -> Tuple[bool, str]:
        """
        Determine if mental health intervention is needed based on screening result.
        Returns: (should_intervene, reason)
        """
        if not screening_result:
            return False, ""
        
        # Trigger on Severe or High risk of Depression
        if screening_result.disease == "depression" and screening_result.detected:
            if screening_result.severity in ["severe", "high", "urgent"]:
                return True, "severe_depression_signs"
            
            # Also trigger if 'monotone' and 'low_energy' are both present (mock heuristic from features)
            details = screening_result.details.get("indicators", [])
            if "monotone_speech" in details and "low_energy" in details:
                return True, "multiple_distress_indicators"

        return False, ""

    def get_empathetic_message(self, reason: str, language: str = "en") -> str:
        """
        Get the script for the intervention.
        """
        if language == "hi":
            return (
                "माफ़ कीजिये, मैं आपको टोक रही हूँ. "  # Apologies for interrupting
                "आपकी आवाज़ में मुझे थोड़ी थकान और परेशानी लग रही है. "  # You sound tired/troubled
                "किसान मित्र सेवा आपके लिए यहाँ है. "  # Kisan Mitra is here for you
                "क्या आप हमारे काउंसलर से 2 मिनट बात करना चाहेंगे? यह मुफ्त है. "  # Talk to counselor?
                "हाँ के लिए 1 दबाएं."  # Press 1 for Yes
            )
        else:
            return (
                "I apologize for interrupting. "
                "I noticed your voice sounds a bit heavy and stressed today. "
                "Farming can be very difficult, and you don't have to carry it alone. "
                "Our Kisan Mitra counselor is available to listen. "
                "Would you like to speak with them for a few minutes? It is free. "
                "Press 1 to connect."
            )

# Singleton
_kisan_service = None

def get_kisan_mitra_service() -> KisanMitraService:
    global _kisan_service
    if _kisan_service is None:
        _kisan_service = KisanMitraService()
    return _kisan_service
