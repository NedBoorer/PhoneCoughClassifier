"""
Phone Cough Classifier - Twilio Voice Webhooks
Handles incoming calls, recording, and classification flow
"""
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path
import tempfile
import asyncio

from fastapi import APIRouter, Form, Request, BackgroundTasks, Depends
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Gather

from app.config import settings
from app.services.twilio_service import get_twilio_service, format_sms_result
from app.ml.model_hub import get_model_hub

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


async def run_comprehensive_analysis(
    recording_url: str,
    call_sid: str,
    caller_number: str,
    language: str = "en"
):
    """
    Background task:
    1. Downloads audio
    2. Runs multi-disease analysis (Respiratory, Parkinson's, Depression)
    3. Saves results to DB (partial update)
    """
    try:
        logger.info(f"Starting comprehensive analysis for call {call_sid}")
        
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
            
        # Run Analysis
        hub = get_model_hub()
        
        # We run all available screenings
        result = hub.run_full_analysis(
            str(local_path),
            enable_respiratory=settings.enable_respiratory_screening,
            enable_parkinsons=settings.enable_parkinsons_screening,
            enable_depression=settings.enable_depression_screening
        )
        
        logger.info(f"Analysis complete for {call_sid}. Risk: {result.overall_risk_level}")
        
        sms_message = format_sms_result(
             classification=result.screenings.get("respiratory", {}).details.get("sound_class", "unknown"),
             confidence=result.screenings.get("respiratory", {}).confidence or 0.0,
             recommendation=result.recommendation,
             language=language, 
             comprehensive_result=result
        )
        
        twilio_service.send_sms(caller_number, sms_message)
        
    except Exception as e:
        logger.error(f"Analysis failed for {call_sid}: {e}")


# ==================
# Webhook Endpoints
# ==================

@router.post("/voice/incoming")
async def incoming_call(request: Request):
    """
    Step 1: Welcome & Combined Recording
    """
    response = VoiceResponse()
    
    response.say(
        "Welcome to the Voice Health Screening. "
        "I will check for respiratory, vocal, and mood indicators. "
        "Please cough three times, then say 'The quick brown fox jumps over the lazy dog'. "
        "Recording starts after the beep.",
        voice="Polly.Aditi"
    )
    
    response.record(
        max_length=15,  # Longer for cough + speech
        timeout=3,
        play_beep=True,
        trim="trim-silence",
        action=f"{settings.base_url}/twilio/voice/handle-recording",
        recording_status_callback=f"{settings.base_url}/twilio/voice/recording-status"
    )
    
    return twiml_response(response)


@router.post("/voice/handle-recording")
async def handle_recording(
    background_tasks: BackgroundTasks,
    CallSid: str = Form(...),
    From: str = Form(...),
    RecordingUrl: str = Form(None)
):
    """
    Step 2: Start Background Analysis & Begin Breath Questionnaire
    """
    response = VoiceResponse()
    
    if RecordingUrl:
        # 1. Start Background Analysis (The "Other 2" happening in background)
        background_tasks.add_task(
            run_comprehensive_analysis,
            recording_url=RecordingUrl,
            call_sid=CallSid,
            caller_number=From
        )
        
        response.say(
            "I've received your audio. I am analyzing your cough and voice now. "
            "While I do that, please answer three quick questions about your breathing.",
            voice="Polly.Aditi"
        )
        
        # 2. Redirect to Questionnaire
        response.redirect(f"{settings.base_url}/twilio/voice/questionnaire/1")
        
    else:
        response.say("I didn't hear anything. Please call back to try again.", voice="Polly.Aditi")
        response.hangup()
        
    return twiml_response(response)


@router.post("/voice/questionnaire/1")
async def questionnaire_q1():
    """Q1: Shortness of Breath"""
    response = VoiceResponse()
    gather = Gather(num_digits=1, action=f"{settings.base_url}/twilio/voice/questionnaire/2")
    
    gather.say(
        "Question 1: Do you experience shortness of breath when walking? "
        "Press 1 for Yes. Press 2 for No.",
        voice="Polly.Aditi"
    )
    response.append(gather)
    
    # Timeout handler
    response.say("I didn't catch that. Moving to the next question.", voice="Polly.Aditi")
    response.redirect(f"{settings.base_url}/twilio/voice/questionnaire/2")
    
    return twiml_response(response)


@router.post("/voice/questionnaire/2")
async def questionnaire_q2(Digits: str = Form(None)):
    """Q2: Chest Pain"""
    # Log answer to Q1 (Digits)
    
    response = VoiceResponse()
    gather = Gather(num_digits=1, action=f"{settings.base_url}/twilio/voice/questionnaire/3")
    
    gather.say(
        "Question 2: Do you feel any pain or tightness in your chest? "
        "Press 1 for Yes. Press 2 for No.",
        voice="Polly.Aditi"
    )
    response.append(gather)
    
    response.say("Moving on.", voice="Polly.Aditi")
    response.redirect(f"{settings.base_url}/twilio/voice/questionnaire/3")
    
    return twiml_response(response)


@router.post("/voice/questionnaire/3")
async def questionnaire_q3(Digits: str = Form(None)):
    """Q3: Smoker"""
    response = VoiceResponse()
    gather = Gather(num_digits=1, action=f"{settings.base_url}/twilio/voice/questionnaire/finish")
    
    gather.say(
        "Last question: Do you smoke tobacco products? "
        "Press 1 for Yes. Press 2 for No.",
        voice="Polly.Aditi"
    )
    response.append(gather)
    
    response.redirect(f"{settings.base_url}/twilio/voice/questionnaire/finish")
    
    return twiml_response(response)


@router.post("/voice/questionnaire/finish")
async def questionnaire_finish(Digits: str = Form(None)):
    """Finish: Wrap up call"""
    response = VoiceResponse()
    
    response.say(
        "Thank you. I have finished analyzing your voice and your answers. "
        "You will receive a detailed health report via text message shortly. "
        "Goodbye and stay healthy.",
        voice="Polly.Aditi"
    )
    response.hangup()
    
    return twiml_response(response)


@router.post("/voice/recording-status")
async def recording_status():
    return Response(status_code=200)


@router.post("/voice/status")
async def call_status():
    return Response(status_code=200)
