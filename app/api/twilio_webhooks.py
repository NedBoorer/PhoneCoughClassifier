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
from twilio.twiml.messaging_response import MessagingResponse

from app.config import settings
from app.utils.twiml_helpers import twiml_response
from app.services.twilio_service import get_twilio_service, format_sms_result
from app.ml.model_hub import get_model_hub

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================
# Helper Functions
# ==================


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

        # Extract respiratory screening result safely
        respiratory_screening = result.screenings.get("respiratory")

        # Retrieve questionnaire data from cache
        questionnaire_data = questionnaire_cache.pop(call_sid, None)
        logger.info(f"Questionnaire data for {call_sid}: {questionnaire_data}")

        # Save results to database
        from app.database.database import async_session_maker
        from app.database.models import CallRecord, ClassificationResult

        async with async_session_maker() as db:
            # Create or update Call Record
            call_record = CallRecord(
                call_sid=call_sid,
                caller_number=caller_number,
                language=language,
                call_status="completed",
                recording_url=recording_url
            )
            db.add(call_record)
            await db.flush()  # Get ID

            # Create Classification Result
            class_result = ClassificationResult(
                call_id=call_record.id,
                classification=respiratory_screening.details.get("sound_class", "unknown") if respiratory_screening else "unknown",
                confidence=respiratory_screening.confidence if respiratory_screening else 0.0,
                probabilities=respiratory_screening.details.get("probabilities", {}) if respiratory_screening else {},
                severity=result.overall_risk_level,
                recommendation=result.recommendation,
                questionnaire_data=questionnaire_data,  # Save questionnaire answers!
                processing_time_ms=result.processing_time_ms
            )
            db.add(class_result)
            await db.commit()

            logger.info(f"Saved results to database for call {call_sid}")

        # Format and send SMS
        sms_message = format_sms_result(
             classification=respiratory_screening.details.get("sound_class", "unknown") if respiratory_screening else "unknown",
             confidence=respiratory_screening.confidence if respiratory_screening else 0.0,
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

    # Check if voice agent is enabled (Conversational Agent Mode)
    if settings.enable_voice_agent:
        logger.info("Redirecting incoming call to Voice Agent")
        response.redirect(f"{settings.base_url}/voice-agent/start")
        return twiml_response(response)
    
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

        # 2. Redirect to Questionnaire with call_sid
        response.redirect(f"{settings.base_url}/twilio/voice/questionnaire/1?call_sid={CallSid}")
        
    else:
        response.say("I didn't hear anything. Please call back to try again.", voice="Polly.Aditi")
        response.hangup()
        
    return twiml_response(response)


@router.post("/voice/questionnaire/1")
async def questionnaire_q1(
    request: Request,
    CallSid: str = Form(...),
    Digits: str = Form(None)
):
    """Q1: Shortness of Breath"""
    # Get Call SID to associate answers
    query_params = dict(request.query_params)
    call_sid = query_params.get("call_sid", CallSid)

    response = VoiceResponse()
    gather = Gather(num_digits=1, action=f"{settings.base_url}/twilio/voice/questionnaire/2?call_sid={call_sid}")

    gather.say(
        "Question 1: Do you experience shortness of breath when walking? "
        "Press 1 for Yes. Press 2 for No.",
        voice="Polly.Aditi"
    )
    response.append(gather)

    # Timeout handler - pass "0" for no answer
    response.redirect(f"{settings.base_url}/twilio/voice/questionnaire/2?call_sid={call_sid}&q1=0")

    return twiml_response(response)


@router.post("/voice/questionnaire/2")
async def questionnaire_q2(
    request: Request,
    Digits: str = Form(None)
):
    """Q2: Chest Pain"""
    query_params = dict(request.query_params)
    call_sid = query_params.get("call_sid", "unknown")
    q1 = query_params.get("q1", Digits if Digits else "0")

    response = VoiceResponse()
    gather = Gather(num_digits=1, action=f"{settings.base_url}/twilio/voice/questionnaire/3?call_sid={call_sid}&q1={q1}")

    gather.say(
        "Question 2: Do you feel any pain or tightness in your chest? "
        "Press 1 for Yes. Press 2 for No.",
        voice="Polly.Aditi"
    )
    response.append(gather)

    response.say("Moving on.", voice="Polly.Aditi")
    response.redirect(f"{settings.base_url}/twilio/voice/questionnaire/3?call_sid={call_sid}&q1={q1}&q2=0")

    return twiml_response(response)


@router.post("/voice/questionnaire/3")
async def questionnaire_q3(
    request: Request,
    Digits: str = Form(None)
):
    """Q3: Smoker"""
    query_params = dict(request.query_params)
    call_sid = query_params.get("call_sid", "unknown")
    q1 = query_params.get("q1", "0")
    q2 = query_params.get("q2", Digits if Digits else "0")

    response = VoiceResponse()
    gather = Gather(num_digits=1, action=f"{settings.base_url}/twilio/voice/questionnaire/finish?call_sid={call_sid}&q1={q1}&q2={q2}")

    gather.say(
        "Last question: Do you smoke tobacco products? "
        "Press 1 for Yes. Press 2 for No.",
        voice="Polly.Aditi"
    )
    response.append(gather)

    response.redirect(f"{settings.base_url}/twilio/voice/questionnaire/finish?call_sid={call_sid}&q1={q1}&q2={q2}&q3=0")

    return twiml_response(response)


# Global dict to store questionnaire answers temporarily (in production, use Redis or database)
questionnaire_cache = {}

@router.post("/voice/questionnaire/finish")
async def questionnaire_finish(
    request: Request,
    Digits: str = Form(None)
):
    """Finish: Wrap up call and save questionnaire data"""
    query_params = dict(request.query_params)
    call_sid = query_params.get("call_sid", "unknown")
    q1 = query_params.get("q1", "0")
    q2 = query_params.get("q2", "0")
    q3 = query_params.get("q3", Digits if Digits else "0")

    # Store questionnaire answers in cache (keyed by call_sid)
    questionnaire_cache[call_sid] = {
        "shortness_of_breath": q1 == "1",
        "chest_pain": q2 == "1",
        "smoker": q3 == "1",
        "answers_raw": {"q1": q1, "q2": q2, "q3": q3}
    }

    logger.info(f"Questionnaire complete for {call_sid}: {questionnaire_cache[call_sid]}")

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


@router.post("/sms/incoming")
async def incoming_sms(
    From: str = Form(...),
    Body: str = Form(...)
):
    """
    Handle incoming SMS to this Twilio number.
    Supports:
    - Referral Verification: "SEEN REF-1234"
    """
    logger.info(f"Incoming SMS from {From}: {Body}")
    
    # Simple command parsing
    clean_body = Body.strip().upper()
    
    # Check for Verification Command
    if "REF-" in clean_body and ("SEEN" in clean_body or "VERIFY" in clean_body):
        import re
        from app.services.referral_service import get_referral_service
        
        # Extract REF-XXXX
        match = re.search(r"REF-\d+", clean_body)
        if match:
            referral_code = match.group(0)
            
            service = get_referral_service()
            # Verify
            success = await service.verify_referral(
                referral_code=referral_code,
                verifier_phone=From,
                notes=Body
            )
            
            resp = MessagingResponse()
            if success:
                resp.message(f"✅ Verified {referral_code}. Thank you, Doctor.")
            else:
                resp.message(f"❌ Error: Code {referral_code} not found.")
                
            return Response(content=str(resp), media_type="application/xml")

    # Default: No response or generic acknowledgment
    return Response(content="<Response></Response>", media_type="application/xml")
