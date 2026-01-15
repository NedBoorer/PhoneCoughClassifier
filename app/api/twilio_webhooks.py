"""
Phone Cough Classifier - Twilio Voice Webhooks
Handles incoming calls, recording, and classification flow
"""
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path
import tempfile

from fastapi import APIRouter, Form, Request, BackgroundTasks, Depends
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Gather

from app.config import settings
from app.services.twilio_service import get_twilio_service, format_sms_result
from app.ml.classifier import get_classifier

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================
# Helper Functions
# ==================

def twiml_response(twiml: VoiceResponse) -> Response:
    """Return TwiML as XML response"""
    return Response(
        content=str(twiml),
        media_type="application/xml"
    )


async def process_and_classify(
    recording_url: str,
    caller_number: str,
    call_sid: str,
    language: str = "en"
):
    """Background task to process recording and send results"""
    try:
        logger.info(f"Processing recording for call {call_sid}")
        
        # Download recording
        twilio_service = get_twilio_service()
        recordings_dir = settings.recordings_dir
        local_path = recordings_dir / f"{call_sid}.wav"
        
        success = await twilio_service.download_recording(
            recording_url,
            str(local_path)
        )
        
        if not success:
            logger.error(f"Failed to download recording for {call_sid}")
            return
        
        # Classify cough
        classifier = get_classifier()
        result = classifier.classify(str(local_path))
        
        logger.info(
            f"Classification for {call_sid}: {result.classification} "
            f"(confidence={result.confidence:.2f})"
        )
        
        # Send SMS
        sms_message = format_sms_result(
            classification=result.classification,
            confidence=result.confidence,
            recommendation=result.recommendation,
            language=language
        )
        
        twilio_service.send_sms(caller_number, sms_message)
        
        # Save to database (if available)
        try:
            from app.database.database import async_session_maker
            from app.database.models import CallRecord, ClassificationResult as DBClassification
            
            async with async_session_maker() as session:
                # Update call record
                call = CallRecord(
                    call_sid=call_sid,
                    caller_number=caller_number,
                    twilio_number=settings.twilio_phone_number,
                    call_status="completed",
                    language=language,
                    recording_url=recording_url,
                    sms_sent=True,
                    sms_delivered_at=datetime.utcnow()
                )
                session.add(call)
                await session.flush()
                
                # Add classification
                db_result = DBClassification(
                    call_id=call.id,
                    classification=result.classification,
                    confidence=result.confidence,
                    probabilities=result.probabilities,
                    method=result.method,
                    processing_time_ms=result.processing_time_ms,
                    severity=result.severity,
                    recommendation=result.recommendation
                )
                session.add(db_result)
                await session.commit()
                
        except Exception as e:
            logger.warning(f"Database save failed: {e}")
        
    except Exception as e:
        logger.error(f"Processing failed for {call_sid}: {e}")


# ==================
# Webhook Endpoints
# ==================

@router.post("/voice/incoming")
async def incoming_call(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
    To: str = Form(None)
):
    """
    Handle incoming voice call.
    Greets the user and starts the cough recording flow.
    """
    logger.info(f"Incoming call: SID={CallSid}, From={From}")
    
    response = VoiceResponse()
    
    # Warm greeting
    response.say(
        "Hello! Welcome to the Cough Classifier. "
        "This service analyzes your cough to help identify its type. "
        "When you're ready, please cough clearly into your phone after the beep. "
        "I will record for 10 seconds.",
        voice="Polly.Aditi",
        language="en-IN"
    )
    
    response.pause(length=1)
    
    response.say(
        "Please cough now.",
        voice="Polly.Aditi",
        language="en-IN"
    )
    
    # Record the cough
    response.record(
        max_length=settings.max_recording_duration,
        timeout=3,
        play_beep=True,
        trim="trim-silence",
        action=f"{settings.base_url}/twilio/voice/recording-complete",
        recording_status_callback=f"{settings.base_url}/twilio/voice/recording-status"
    )
    
    return twiml_response(response)


@router.post("/voice/recording-complete")
async def recording_complete(
    background_tasks: BackgroundTasks,
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
    RecordingUrl: str = Form(None),
    RecordingDuration: str = Form("0")
):
    """
    Handle recording completion.
    Acknowledges receipt and starts background processing.
    """
    logger.info(
        f"Recording complete: SID={CallSid}, "
        f"Duration={RecordingDuration}s, URL={RecordingUrl}"
    )
    
    response = VoiceResponse()
    
    if RecordingUrl:
        response.say(
            "Thank you! I received your cough recording. "
            "I'm analyzing it now. You will receive your results "
            "via SMS in a few moments. "
            "Please take care and stay healthy!",
            voice="Polly.Aditi",
            language="en-IN"
        )
        
        # Start background processing
        background_tasks.add_task(
            process_and_classify,
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


@router.post("/voice/recording-status")
async def recording_status(
    request: Request,
    CallSid: str = Form(...),
    RecordingStatus: str = Form(None),
    RecordingUrl: str = Form(None),
    RecordingSid: str = Form(None)
):
    """Handle recording status callbacks"""
    logger.info(
        f"Recording status: SID={CallSid}, "
        f"Status={RecordingStatus}, RecSID={RecordingSid}"
    )
    return Response(status_code=200)


@router.post("/voice/status")
async def call_status(
    request: Request,
    CallSid: str = Form(...),
    CallStatus: str = Form(None),
    CallDuration: str = Form("0")
):
    """Handle call status callbacks"""
    logger.info(
        f"Call status: SID={CallSid}, "
        f"Status={CallStatus}, Duration={CallDuration}s"
    )
    return Response(status_code=200)
