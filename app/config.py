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
    twilio_phone_number: str = ""
    
    # OpenAI
    openai_api_key: str = ""
    
    # Application
    base_url: str = "http://localhost:8000"
    environment: str = "development"
    debug: bool = True
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./data/cough_classifier.db"
    
    # Audio
    max_recording_duration: int = 10
    audio_sample_rate: int = 16000
    
    # ML Model
    model_path: str = "./models/cough_classifier.joblib"
    use_hear_embeddings: bool = True
    
    # India Accessibility
    default_language: str = "en"
    enable_multilingual: bool = True
    
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
