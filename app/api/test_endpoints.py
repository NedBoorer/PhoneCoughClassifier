"""
Phone Cough Classifier - Test Endpoints
For testing without making actual phone calls
"""
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
import tempfile

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.ml.classifier import get_classifier
from app.utils.audio_quality import get_quality_checker

logger = logging.getLogger(__name__)

router = APIRouter()


class ClassifyResponse(BaseModel):
    """Classification response model"""
    classification: str
    confidence: float
    probabilities: dict
    method: str
    processing_time_ms: int
    severity: str
    recommendation: str


class QualityResponse(BaseModel):
    """Audio quality response model"""
    overall_score: float
    snr_db: float
    has_clipping: bool
    silence_ratio: float
    is_acceptable: bool
    issues: list
    recommendations: list


class SystemStatus(BaseModel):
    """System status response"""
    status: str
    database: str
    ml_model: str
    twilio: str
    openai: str


@router.post("/classify", response_model=ClassifyResponse)
async def test_classify(audio_file: UploadFile = File(...)):
    """
    Upload an audio file to test cough classification.
    
    This endpoint allows testing without making phone calls.
    Supports: WAV, MP3, OGG, WebM, M4A
    """
    # Validate file type
    allowed_types = [
        "audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg",
        "audio/webm", "audio/x-wav", "audio/x-m4a", "audio/mp4"
    ]
    
    content_type = audio_file.content_type or ""
    if not any(t in content_type for t in ["audio", "octet-stream"]):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Expected audio file, got {content_type}"
        )
    
    try:
        # Save to temp file
        suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Classify
        classifier = get_classifier()
        result = classifier.classify(tmp_path)
        
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)
        
        return ClassifyResponse(
            classification=result.classification,
            confidence=result.confidence,
            probabilities=result.probabilities,
            method=result.method,
            processing_time_ms=result.processing_time_ms,
            severity=result.severity,
            recommendation=result.recommendation
        )
        
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quality", response_model=QualityResponse)
async def test_quality(audio_file: UploadFile = File(...)):
    """
    Test audio quality assessment.
    
    Returns quality metrics and recommendations.
    """
    try:
        # Save to temp file
        suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Check quality
        checker = get_quality_checker()
        result = checker.assess_quality(tmp_path)
        
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)
        
        return QualityResponse(
            overall_score=result.overall_score,
            snr_db=result.snr_db,
            has_clipping=result.has_clipping,
            silence_ratio=result.silence_ratio,
            is_acceptable=result.is_acceptable,
            issues=result.issues,
            recommendations=result.recommendations
        )
        
    except Exception as e:
        logger.error(f"Quality check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=SystemStatus)
async def system_status():
    """Check system component status"""
    status = SystemStatus(
        status="healthy",
        database="unknown",
        ml_model="unknown",
        twilio="not configured",
        openai="not configured"
    )
    
    # Check database
    try:
        from app.database.database import async_session_maker
        from sqlalchemy import text
        async with async_session_maker() as session:
            await session.execute(text("SELECT 1"))
        status.database = "connected"
    except Exception as e:
        status.database = f"error: {str(e)[:50]}"
    
    # Check ML model
    try:
        classifier = get_classifier()
        status.ml_model = classifier.model_type
    except Exception as e:
        status.ml_model = f"error: {str(e)[:50]}"
    
    # Check Twilio
    status.twilio = "configured" if settings.twilio_account_sid else "not configured"
    
    # Check OpenAI
    status.openai = "configured" if settings.openai_api_key else "not configured"
    
    return status


@router.get("/model-info")
async def model_info():
    """Get information about the loaded ML model"""
    try:
        classifier = get_classifier()
        
        return {
            "model_type": classifier.model_type,
            "use_hear": classifier.use_hear,
            "model_path": classifier.model_path,
            "model_loaded": classifier._loaded,
            "sklearn_available": classifier._sklearn_model is not None,
            "hear_available": classifier._hear_model is not None,
            "classes": ["dry", "wet", "whooping", "chronic", "normal"]
        }
        
    except Exception as e:
        return {"error": str(e)}


@router.get("/languages")
async def list_languages():
    """List supported languages for India accessibility"""
    from app.utils.i18n import LANGUAGES
    
    return {
        code: {
            "name": lang.name,
            "native_name": lang.native_name
        }
        for code, lang in LANGUAGES.items()
    }
