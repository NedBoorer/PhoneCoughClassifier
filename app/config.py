"""
Phone Cough Classifier - Configuration Management
Uses pydantic-settings for type-safe environment variable loading
"""
from functools import lru_cache
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Twilio
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""  # Primary number for general health screening
    twilio_market_phone_number: str = ""  # Dedicated number for Mandi Bol market service
    
    # OpenAI
    openai_api_key: str = ""
    
    # Application
    base_url: str = "http://localhost:8000"
    environment: str = "development"
    debug: bool = True
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./data/cough_classifier.db"
    
    # Audio
    max_recording_duration: int = 6  # Reduced from 10 to 6 seconds for faster experience
    audio_sample_rate: int = 16000
    analysis_timeout_seconds: int = 15  # Maximum time for analysis before timeout
    
    # ML Model
    model_path: str = "./models/cough_classifier.joblib"
    use_hear_embeddings: bool = True
    
    # Health Assessment Models
    parkinsons_model_path: str = "./models/parkinsons_classifier.joblib"
    depression_model_path: str = "./models/depression_classifier.joblib"
    
    # India Accessibility
    default_language: str = "en"
    enable_multilingual: bool = True
    
    # Enhanced Health Screening
    enable_respiratory_screening: bool = True
    enable_parkinsons_screening: bool = True  # Enabled for demo
    enable_depression_screening: bool = True  # Enabled for demo
    enable_tuberculosis_screening: bool = True  # TB screening enabled by default
    screening_model_device: str = "cpu"  # "cpu" or "cuda"
    
    # Tuberculosis Screening
    tb_model_path: str = "./models/tb_classifier.joblib"
    tb_screening_threshold: float = 0.45  # Lower threshold for screening sensitivity

    # Rural India Accessibility
    doctor_helpline_number: str = "+910000000000"  # Placeholder eSanjeevani or similar
    enable_whatsapp_reports: bool = True
    high_risk_threshold: float = 0.85
    missed_call_callback_enabled: bool = True

    # Farmer-Specific Features
    enable_farmer_screening: bool = True
    enable_family_screening: bool = True
    pesticide_risk_threshold: float = 0.7  # Lower threshold for pesticide exposure
    dust_risk_threshold: float = 0.65

    # WhatsApp Configuration
    twilio_whatsapp_from: str = "whatsapp:+14155238886"  # Twilio sandbox number
    enable_whatsapp_bot: bool = True
    whatsapp_welcome_message: str = "🩺 Welcome to Voice Health! I can check your cough for health issues.\n\nPlease reply with your preferred language:\n\n1. English\n2. Hindi"
    
    # ML Thresholds
    confidence_threshold: float = 0.7
    
    # Feature Flags (from .env)
    enable_missed_call: bool = True
    enable_asha_mode: bool = True
    enable_kisan_manas: bool = True
    
    # Voice Agent Configuration
    enable_voice_agent: bool = True
    voice_agent_model: str = "gpt-4o-mini"
    voice_agent_max_turns: int = 15  # Maximum conversation turns before forcing recording
    voice_agent_timeout: int = 10  # seconds to wait for speech input
    max_no_input_attempts: int = 5  # Maximum no-input attempts before ending call
    max_recording_attempts: int = 3  # Maximum recording retry attempts
    max_family_screenings: int = 5  # Maximum family members to screen per call
    
    # Trust & Reliability
    trusted_authority_name: str = "District Health Mission"
    mock_daily_users: int = 400
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./data/app.log"

    # Paths
    @property
    def data_dir(self) -> Path:
        path = Path("./data")
        path.mkdir(exist_ok=True)
        return path
    
    @property
    def models_dir(self) -> Path:
        path = Path("./models")
        path.mkdir(exist_ok=True)
        return path
    
    @property
    def recordings_dir(self) -> Path:
        path = Path("./recordings")
        path.mkdir(exist_ok=True)
        return path
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance"""
    return Settings()


# Convenience exports
settings = get_settings()


# ===============
# FARMING CALENDAR
# ===============
# Seasonal calendar for follow-up scheduling
# Respects farmer availability throughout the agricultural cycle
FARMING_CALENDAR = {
    "sowing": {
        "months": [6, 7],  # June-July (Kharif sowing)
        "delay_days": 14,  # 2 weeks delay
        "description": "Sowing season - farmers are moderately busy"
    },
    "growing": {
        "months": [8, 9, 10],  # August-October (Kharif growing)
        "delay_days": 7,  # Normal follow-up
        "description": "Growing season - normal availability"
    },
    "harvest": {
        "months": [11, 12, 1],  # November-January (Kharif harvest + Rabi sowing)
        "delay_days": 21,  # 3 weeks delay
        "description": "Harvest season - farmers are very busy"
    },
    "off_season": {
        "months": [2, 3, 4, 5],  # February-May (Post-Rabi, pre-Kharif)
        "delay_days": 3,  # Faster follow-up
        "description": "Off-season - farmers have more time"
    }
}


def get_current_farming_season(month: int = None) -> str:
    """
    Get the current farming season based on month.

    Args:
        month: Month number (1-12). If None, uses current month.

    Returns:
        Season name: "sowing", "growing", "harvest", or "off_season"
    """
    if month is None:
        from datetime import datetime
        month = datetime.now().month

    for season_name, season_data in FARMING_CALENDAR.items():
        if month in season_data["months"]:
            return season_name

    return "off_season"


def get_followup_delay_for_season(season: str) -> int:
    """
    Get the recommended follow-up delay (in days) for a farming season.

    Args:
        season: Season name

    Returns:
        Number of days to delay follow-up
    """
    return FARMING_CALENDAR.get(season, {}).get("delay_days", 7)
