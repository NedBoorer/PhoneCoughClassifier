"""
Phone Cough Classifier - Test Endpoints
For testing without making actual phone calls
"""
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import tempfile

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
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


class ScreeningResultModel(BaseModel):
    """Single disease screening result"""
    disease: str
    detected: bool
    confidence: float
    severity: str
    details: dict = {}
    recommendation: str = ""


class ComprehensiveHealthResponse(BaseModel):
    """Comprehensive health screening response"""
    primary_concern: str
    overall_risk_level: str
    screenings: Dict[str, ScreeningResultModel] = {}
    voice_biomarkers: Dict[str, float] = {}
    processing_time_ms: int = 0
    recommendation: str = ""


class VoiceBiomarkersResponse(BaseModel):
    """Voice biomarkers response"""
    f0_mean: float = 0.0
    f0_std: float = 0.0
    jitter: float = 0.0
    shimmer: float = 0.0
    hnr: float = 0.0
    energy_mean: float = 0.0
    speaking_rate: float = 0.0
    pause_ratio: float = 0.0




@router.post("/classify", response_model=ClassifyResponse)
async def test_classify(audio_file: UploadFile = File(...)):
    """
    Upload an audio file to test cough classification.
    
    This endpoint allows testing without making phone calls.
    Supports: WAV, MP3, OGG, WebM, M4A
    """
    print("DEBUG: Received request at /test/classify")
    # Validate file type
    allowed_types = [
        "audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg",
        "audio/webm", "audio/x-wav", "audio/x-m4a", "audio/mp4"
    ]
    
    content_type = audio_file.content_type or ""
    print(f"DEBUG: File content type: {content_type}")
    if not any(t in content_type for t in ["audio", "octet-stream"]):
        print(f"DEBUG: Invalid content type: {content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Expected audio file, got {content_type}"
        )
    
    try:
        # Save to temp file
        suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
        print(f"DEBUG: Saving upload to temporary file with suffix {suffix}...")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name
        print(f"DEBUG: Saved to {tmp_path}")
        
        # Classify
        print("DEBUG: Loading classifier...")
        classifier = get_classifier()
        print(f"DEBUG: Running classification on {tmp_path}...")
        result = classifier.classify(tmp_path)
        print(f"DEBUG: Classification result: {result.classification}, Confidence: {result.confidence}")
        
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)
        print("DEBUG: Cleaned up temp file.")
        
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
        print(f"DEBUG: Classification failed: {e}")
        import traceback
        traceback.print_exc()
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


# ==================
# Enhanced Health Screening Endpoints
# ==================

@router.post("/analyze-full", response_model=ComprehensiveHealthResponse)
async def analyze_full_health(
    audio_file: UploadFile = File(...),
    enable_respiratory: bool = Query(True, description="Enable respiratory screening (COPD/Asthma)"),
    enable_parkinsons: bool = Query(False, description="Enable Parkinson's voice screening"),
    enable_depression: bool = Query(False, description="Enable depression speech screening")
):
    """
    Run comprehensive voice health analysis using multiple AI models.
    
    - **Respiratory**: Detects crackles/wheezes indicating COPD, asthma, pneumonia
    - **Parkinson's**: Analyzes jitter, shimmer, HNR voice biomarkers
    - **Depression**: Screens speech patterns for mood indicators
    
    Supports: WAV, MP3, OGG, WebM, M4A (min 3 seconds)
    """
    try:
        # Save to temp file
        suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            from app.ml.model_hub import get_model_hub
            
            hub = get_model_hub()
            result = hub.run_full_analysis(
                tmp_path,
                enable_respiratory=enable_respiratory,
                enable_parkinsons=enable_parkinsons,
                enable_depression=enable_depression
            )
            
            # Convert to response model
            screenings_dict = {}
            for name, screening in result.screenings.items():
                screenings_dict[name] = ScreeningResultModel(
                    disease=screening.disease,
                    detected=screening.detected,
                    confidence=screening.confidence,
                    severity=screening.severity,
                    details=screening.details,
                    recommendation=screening.recommendation
                )
            
            return ComprehensiveHealthResponse(
                primary_concern=result.primary_concern,
                overall_risk_level=result.overall_risk_level,
                screenings=screenings_dict,
                voice_biomarkers=result.voice_biomarkers,
                processing_time_ms=result.processing_time_ms,
                recommendation=result.recommendation
            )
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            
    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/respiratory-screen")
async def respiratory_screen(audio_file: UploadFile = File(...)):
    """
    Screen for respiratory conditions (COPD, Asthma, Pneumonia).
    
    Uses PANNs (Pretrained Audio Neural Networks) to detect:
    - Crackles (COPD/Pneumonia indicators)
    - Wheezes (Asthma/Bronchitis indicators)
    """
    try:
        suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            from app.ml.model_hub import get_model_hub
            
            hub = get_model_hub()
            result = hub.respiratory_classifier.classify(tmp_path)
            
            return {
                "disease": result.disease,
                "detected": result.detected,
                "confidence": result.confidence,
                "severity": result.severity,
                "sound_class": result.details.get("sound_class"),
                "probabilities": result.details.get("probabilities", {}),
                "recommendation": result.recommendation
            }
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            
    except Exception as e:
        logger.error(f"Respiratory screening failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/parkinsons-screen")
async def parkinsons_screen(audio_file: UploadFile = File(...)):
    """
    Screen for Parkinson's disease voice indicators.
    
    Analyzes voice biomarkers:
    - Jitter (pitch perturbation)
    - Shimmer (amplitude perturbation)
    - Harmonics-to-noise ratio (HNR)
    
    Requires clear speech sample (min 3 seconds).
    """
    try:
        suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            from app.ml.model_hub import get_model_hub
            
            hub = get_model_hub()
            result = hub.parkinsons_detector.detect(tmp_path)
            
            return {
                "disease": result.disease,
                "detected": result.detected,
                "confidence": result.confidence,
                "severity": result.severity,
                "biomarkers": result.details.get("biomarkers", {}),
                "parkinson_probability": result.details.get("parkinson_probability"),
                "recommendation": result.recommendation
            }
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            
    except Exception as e:
        logger.error(f"Parkinson's screening failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/depression-screen")
async def depression_screen(audio_file: UploadFile = File(...)):
    """
    Screen for depression indicators from speech patterns.
    
    Analyzes:
    - Pitch variability (monotonicity)
    - Speaking rate
    - Energy levels
    - Pause patterns
    
    Requires natural speech sample (min 10 seconds recommended).
    """
    try:
        suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            from app.ml.model_hub import get_model_hub
            
            hub = get_model_hub()
            result = hub.depression_screener.screen(tmp_path)
            
            return {
                "disease": result.disease,
                "detected": result.detected,
                "risk_score": result.confidence,
                "severity": result.severity,
                "indicators": result.details.get("indicators", []),
                "features": result.details.get("features", {}),
                "recommendation": result.recommendation
            }
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            
    except Exception as e:
        logger.error(f"Depression screening failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice-biomarkers", response_model=VoiceBiomarkersResponse)
async def extract_voice_biomarkers(audio_file: UploadFile = File(...)):
    """
    Extract voice biomarkers from audio.
    
    Returns:
    - F0 (fundamental frequency)
    - Jitter/Shimmer (voice perturbation)
    - HNR (harmonics-to-noise ratio)
    - Energy and speaking rate
    """
    try:
        suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            from app.ml.voice_biomarkers import get_biomarker_extractor
            
            extractor = get_biomarker_extractor()
            biomarkers = extractor.extract(tmp_path)
            
            return VoiceBiomarkersResponse(
                f0_mean=biomarkers.f0_mean,
                f0_std=biomarkers.f0_std,
                jitter=biomarkers.jitter,
                shimmer=biomarkers.shimmer,
                hnr=biomarkers.hnr,
                energy_mean=biomarkers.energy_mean,
                speaking_rate=biomarkers.speaking_rate,
                pause_ratio=biomarkers.pause_ratio
            )
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            
    except Exception as e:
        logger.error(f"Biomarker extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/screening-models")
async def list_screening_models():
    """List available screening models and their status"""
    models = {
        "cough_classifier": {
            "name": "Cough Type Classifier",
            "description": "Classifies cough into dry, wet, whooping, chronic, normal",
            "status": "active",
            "model": "HeAR/sklearn"
        },
        "respiratory_classifier": {
            "name": "Respiratory Sound Classifier",
            "description": "Detects crackles/wheezes for COPD, Asthma, Pneumonia screening",
            "status": "active",
            "model": "PANNs CNN6"
        },
        "parkinsons_detector": {
            "name": "Parkinson's Voice Detector",
            "description": "Analyzes voice biomarkers for Parkinson's indicators",
            "status": "active",
            "model": "SVM with jitter/shimmer/HNR"
        },
        "depression_screener": {
            "name": "Depression Speech Screener",
            "description": "Screens speech patterns for depression indicators",
            "status": "active",
            "model": "Feature-based analysis"
        }
    }
    
    # Check if models are loadable
    try:
        from app.ml.model_hub import get_model_hub
        hub = get_model_hub()
        models["respiratory_classifier"]["loaded"] = hub._respiratory is not None and hub._respiratory._loaded
        models["parkinsons_detector"]["loaded"] = hub._parkinsons is not None and hub._parkinsons._loaded
        models["depression_screener"]["loaded"] = hub._depression is not None and hub._depression._loaded
    except Exception as e:
        logger.error(f"Error checking model status: {e}")
    
    return {"models": models}

@router.post("/send-whatsapp-report")
async def test_send_whatsapp_report(
    to: str = Query(..., description="Target phone number (E.164 format)"),
    language: str = Query("en", description="Language code (en/hi)"),
):
    """
    Test sending a health report via WhatsApp.
    
    This manually triggers the report sending logic used in the voice agent.
    Values are mocked for demonstration.
    """
    try:
        from app.services.whatsapp_service import get_whatsapp_service
        from app.ml.model_hub import ComprehensiveHealthResult, ScreeningResult
        
        # Create mock result
        result = ComprehensiveHealthResult(
            primary_concern="none",
            overall_risk_level="normal",
            processing_time_ms=123,
            recommendation="Your lung sounds are normal. Keep maintaining good health!"
        )
        # Add a dummy screening
        result.screenings["respiratory"] = ScreeningResult(
            disease="repiratory",
            detected=False,
            confidence=0.95,
            severity="normal",
            details={"sound_class": "normal"},
            recommendation="No respiratory issues detected."
        )
        
        service = get_whatsapp_service()
        
        # Attempt send
        success = service.send_health_card(
            to=to,
            result=result,
            language=language
        )
        
        if success:
            return {"status": "success", "message": f"WhatsApp report sent to {to}"}
        else:
            return {"status": "error", "message": "Failed to send WhatsApp report (check logs)"}
            
    except Exception as e:
        logger.error(f"Test WhatsApp output failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
