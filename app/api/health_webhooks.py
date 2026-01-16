"""
Health Assessment Voice Webhooks
Twilio webhooks for Parkinson's Disease and Depression voice screening
"""
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, Form, Request, BackgroundTasks
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse

from app.config import settings
from app.utils.twiml_helpers import twiml_response
from app.services.twilio_service import get_twilio_service

logger = logging.getLogger(__name__)

router = APIRouter()


def format_parkinsons_sms(result, language: str = "en") -> str:
    """Format Parkinson's screening result for SMS"""
    messages = {
        "en": {
            "header": "🏥 Voice Health Screening Results\n\nParkinson's Screening:",
            "risk_levels": {
                "normal": "✅ Normal - No concerning patterns detected",
                "low_risk": "🟡 Low Risk - Minor variations noted",
                "moderate_risk": "🟠 Moderate Risk - Some patterns warrant attention",
                "elevated_risk": "🔴 Elevated Risk - Please consult a neurologist",
                "unknown": "⚠️ Unable to analyze"
            },
            "disclaimer": "\n\n⚠️ DISCLAIMER: This is a screening tool only, NOT a diagnosis. Please consult a doctor for proper evaluation."
        }
    }
    
    msg = messages.get(language, messages["en"])
    risk_text = msg["risk_levels"].get(result.risk_level, msg["risk_levels"]["unknown"])
    
    sms = f"{msg['header']}\n{risk_text}"
    sms += f"\nConfidence: {result.confidence:.0%}"
    
    if result.recommendation:
        sms += f"\n\n💡 {result.recommendation[:200]}"
    
    sms += msg["disclaimer"]
    
    return sms


def format_depression_sms(result, language: str = "en") -> str:
    """Format Depression screening result for SMS"""
    messages = {
        "en": {
            "header": "💚 Mental Health Voice Screening\n\nDepression Screening:",
            "severity_levels": {
                "minimal": "✅ Minimal - Your voice patterns appear healthy",
                "mild": "🟡 Mild - Some variations detected",
                "moderate": "🟠 Moderate - Consider speaking with a counselor",
                "moderately_severe": "🟠 Moderately Severe - Please reach out for support",
                "severe": "🔴 Severe - Please contact a mental health professional",
                "unknown": "⚠️ Unable to analyze"
            },
            "helplines": "\n\n📞 HELPLINES:\n• iCALL: 9152987821\n• Vandrevala: 1860-2662-345",
            "disclaimer": "\n\n⚠️ This is a screening tool only. If you're struggling, please reach out - you are not alone. 💙"
        }
    }
    
    msg = messages.get(language, messages["en"])
    severity_text = msg["severity_levels"].get(result.severity_level, msg["severity_levels"]["unknown"])
    
    sms = f"{msg['header']}\n{severity_text}"
    sms += f"\nConfidence: {result.confidence:.0%}"
    
    if result.recommendation:
        sms += f"\n\n💡 {result.recommendation[:180]}"
    
    # Always include helplines for depression screening
    sms += msg["helplines"]
    sms += msg["disclaimer"]
    
    return sms


# ==================
# Background Processing
# ==================

async def process_parkinsons_recording(
    recording_url: str,
    caller_number: str,
    call_sid: str,
    language: str = "en"
):
    """Background task to process Parkinson's screening recording"""
    try:
        logger.info(f"Processing Parkinson's recording for call {call_sid}")
        
        # Download recording
        twilio_service = get_twilio_service()
        recordings_dir = settings.recordings_dir
        local_path = recordings_dir / f"parkinsons_{call_sid}.wav"
        
        success = await twilio_service.download_recording(
            recording_url,
            str(local_path)
        )
        
        if not success:
            logger.error(f"Failed to download recording for {call_sid}")
            return
        
        # Classify
        from app.ml.parkinsons_classifier import get_parkinsons_classifier
        classifier = get_parkinsons_classifier()
        result = classifier.classify(str(local_path))
        
        logger.info(
            f"Parkinson's screening for {call_sid}: {result.risk_level} "
            f"(confidence={result.confidence:.2f})"
        )
        
        # Send SMS
        sms_message = format_parkinsons_sms(result, language)
        twilio_service.send_sms(caller_number, sms_message)
        
        # Save to database
        try:
            from app.database.database import async_session_maker
            from app.database.models import HealthAssessment
            
            async with async_session_maker() as session:
                assessment = HealthAssessment(
                    call_sid=call_sid,
                    caller_number=caller_number,
                    assessment_type="parkinsons",
                    classification=result.risk_level,
                    confidence=result.confidence,
                    risk_level=result.risk_level,
                    indicators=result.indicators,
                    recommendation=result.recommendation,
                    method=result.method,
                    processing_time_ms=result.processing_time_ms
                )
                session.add(assessment)
                await session.commit()
                
        except Exception as e:
            logger.warning(f"Database save failed: {e}")
        
    except Exception as e:
        logger.error(f"Parkinson's processing failed for {call_sid}: {e}")


async def process_depression_recording(
    recording_url: str,
    caller_number: str,
    call_sid: str,
    language: str = "en"
):
    """Background task to process Depression screening recording"""
    try:
        logger.info(f"Processing Depression recording for call {call_sid}")
        
        # Download recording
        twilio_service = get_twilio_service()
        recordings_dir = settings.recordings_dir
        local_path = recordings_dir / f"depression_{call_sid}.wav"
        
        success = await twilio_service.download_recording(
            recording_url,
            str(local_path)
        )
        
        if not success:
            logger.error(f"Failed to download recording for {call_sid}")
            return
        
        # Classify
        from app.ml.depression_classifier import get_depression_classifier
        classifier = get_depression_classifier()
        result = classifier.classify(str(local_path))
        
        logger.info(
            f"Depression screening for {call_sid}: {result.severity_level} "
            f"(confidence={result.confidence:.2f})"
        )
        
        # Send SMS
        sms_message = format_depression_sms(result, language)
        twilio_service.send_sms(caller_number, sms_message)
        
        # Save to database
        try:
            from app.database.database import async_session_maker
            from app.database.models import HealthAssessment
            
            async with async_session_maker() as session:
                assessment = HealthAssessment(
                    call_sid=call_sid,
                    caller_number=caller_number,
                    assessment_type="depression",
                    classification=result.severity_level,
                    confidence=result.confidence,
                    risk_level=result.severity_level,
                    indicators=result.indicators,
                    recommendation=result.recommendation,
                    method=result.method,
                    processing_time_ms=result.processing_time_ms
                )
                session.add(assessment)
                await session.commit()
                
        except Exception as e:
            logger.warning(f"Database save failed: {e}")
        
    except Exception as e:
        logger.error(f"Depression processing failed for {call_sid}: {e}")


# ==================
# Parkinson's Endpoints
# ==================

@router.post("/voice/parkinsons/incoming")
async def parkinsons_incoming(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
    To: str = Form(None)
):
    """
    Handle incoming call for Parkinson's screening.
    Guides user through sustained vowel recording.
    """
    logger.info(f"Parkinson's screening call: SID={CallSid}, From={From}")
    
    response = VoiceResponse()
    
    response.say(
        "Hello! Welcome to the Voice Health Screening service. "
        "This call will help screen for vocal patterns associated with Parkinson's disease. "
        "Please note: this is a screening tool only, not a diagnosis. ",
        voice="Polly.Aditi",
        language="en-IN"
    )
    
    response.pause(length=1)
    
    response.say(
        "When you hear the beep, please say the sound 'aah' in a steady voice "
        "for about 5 seconds. Take a deep breath and begin after the beep.",
        voice="Polly.Aditi",
        language="en-IN"
    )
    
    response.pause(length=1)
    
    response.record(
        max_length=10,
        timeout=3,
        play_beep=True,
        trim="trim-silence",
        action=f"{settings.base_url}/health/voice/parkinsons/recording-complete",
        recording_status_callback=f"{settings.base_url}/health/voice/parkinsons/recording-status"
    )
    
    return twiml_response(response)


@router.post("/voice/parkinsons/recording-complete")
async def parkinsons_recording_complete(
    background_tasks: BackgroundTasks,
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
    RecordingUrl: str = Form(None),
    RecordingDuration: str = Form("0")
):
    """Handle Parkinson's recording completion"""
    logger.info(
        f"Parkinson's recording complete: SID={CallSid}, "
        f"Duration={RecordingDuration}s"
    )
    
    response = VoiceResponse()
    
    if RecordingUrl:
        response.say(
            "Thank you! I received your voice sample. "
            "I'm analyzing it now for vocal patterns. "
            "You will receive your screening results via SMS shortly. "
            "Remember, this is a screening tool - please consult a doctor "
            "for any medical concerns. Take care!",
            voice="Polly.Aditi",
            language="en-IN"
        )
        
        background_tasks.add_task(
            process_parkinsons_recording,
            recording_url=RecordingUrl,
            caller_number=From,
            call_sid=CallSid,
            language="en"
        )
    else:
        response.say(
            "I'm sorry, I didn't receive your recording. "
            "Please try calling again. Goodbye!",
            voice="Polly.Aditi",
            language="en-IN"
        )
    
    response.hangup()
    return twiml_response(response)


@router.post("/voice/parkinsons/recording-status")
async def parkinsons_recording_status(
    request: Request,
    CallSid: str = Form(...),
    RecordingStatus: str = Form(None),
    RecordingSid: str = Form(None)
):
    """Handle Parkinson's recording status callbacks"""
    logger.info(f"Parkinson's recording status: {CallSid}, {RecordingStatus}")
    return Response(status_code=200)


# ==================
# Depression Endpoints
# ==================

@router.post("/voice/depression/incoming")
async def depression_incoming(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
    To: str = Form(None)
):
    """
    Handle incoming call for Depression screening.
    Guides user through spontaneous speech recording.
    """
    logger.info(f"Depression screening call: SID={CallSid}, From={From}")
    
    response = VoiceResponse()
    
    response.say(
        "Hello! Welcome to the Mental Health Voice Screening service. "
        "This is a safe space to share how you're feeling. "
        "Please remember: this is a screening tool only, and we're here to help. ",
        voice="Polly.Aditi",
        language="en-IN"
    )
    
    response.pause(length=1)
    
    response.say(
        "After the beep, please speak freely for about 15 seconds. "
        "You can describe how you've been feeling lately, "
        "or simply talk about your day. "
        "There are no right or wrong answers. Take your time.",
        voice="Polly.Aditi",
        language="en-IN"
    )
    
    response.pause(length=1)
    
    response.record(
        max_length=30,
        timeout=5,
        play_beep=True,
        trim="trim-silence",
        action=f"{settings.base_url}/health/voice/depression/recording-complete",
        recording_status_callback=f"{settings.base_url}/health/voice/depression/recording-status"
    )
    
    return twiml_response(response)


@router.post("/voice/depression/recording-complete")
async def depression_recording_complete(
    background_tasks: BackgroundTasks,
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
    RecordingUrl: str = Form(None),
    RecordingDuration: str = Form("0")
):
    """Handle Depression recording completion"""
    logger.info(
        f"Depression recording complete: SID={CallSid}, "
        f"Duration={RecordingDuration}s"
    )
    
    response = VoiceResponse()
    
    if RecordingUrl:
        response.say(
            "Thank you for sharing. I appreciate you taking this step. "
            "I'm analyzing your voice patterns now. "
            "You will receive your results via SMS shortly, "
            "along with helpful resources. "
            "Remember, reaching out for support is a sign of strength. "
            "Take care of yourself!",
            voice="Polly.Aditi",
            language="en-IN"
        )
        
        background_tasks.add_task(
            process_depression_recording,
            recording_url=RecordingUrl,
            caller_number=From,
            call_sid=CallSid,
            language="en"
        )
    else:
        response.say(
            "I'm sorry, I didn't receive your recording. "
            "Please try calling again whenever you're ready. "
            "Take care!",
            voice="Polly.Aditi",
            language="en-IN"
        )
    
    response.hangup()
    return twiml_response(response)


@router.post("/voice/depression/recording-status")
async def depression_recording_status(
    request: Request,
    CallSid: str = Form(...),
    RecordingStatus: str = Form(None),
    RecordingSid: str = Form(None)
):
    """Handle Depression recording status callbacks"""
    logger.info(f"Depression recording status: {CallSid}, {RecordingStatus}")
    return Response(status_code=200)


# ==================
# Unified Entry Point
# ==================

@router.post("/voice/incoming")
async def health_incoming(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
    To: str = Form(None),
    Digits: str = Form(None)
):
    """
    Unified entry point for health screening.
    Presents menu to choose screening type.
    """
    logger.info(f"Health screening call: SID={CallSid}, From={From}, Digits={Digits}")

    # Track attempt count to prevent infinite loops
    query_params = dict(request.query_params)
    attempt = int(query_params.get("attempt", 0))

    response = VoiceResponse()

    if Digits == "1":
        # Redirect to Parkinson's screening
        response.redirect(f"{settings.base_url}/health/voice/parkinsons/incoming")
    elif Digits == "2":
        # Redirect to Depression screening
        response.redirect(f"{settings.base_url}/health/voice/depression/incoming")
    else:
        # Max 3 attempts before auto-hangup
        if attempt >= 3:
            response.say(
                "I did not receive a valid selection. Please call back when you are ready. Goodbye.",
                voice="Polly.Aditi",
                language="en-IN"
            )
            response.hangup()
            return twiml_response(response)

        # Present menu
        gather = response.gather(
            num_digits=1,
            action=f"{settings.base_url}/health/voice/incoming",
            method="POST",
            timeout=10
        )

        gather.say(
            "Welcome to the Voice Health Screening service. "
            "Press 1 for Parkinson's disease voice screening. "
            "Press 2 for mental health voice screening. ",
            voice="Polly.Aditi",
            language="en-IN"
        )

        # Increment attempt counter on redirect
        response.redirect(f"{settings.base_url}/health/voice/incoming?attempt={attempt + 1}")

    return twiml_response(response)
