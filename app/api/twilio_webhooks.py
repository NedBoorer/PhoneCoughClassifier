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
        print(f"DEBUG: Starting comprehensive analysis for call {call_sid} from {caller_number}")
        logger.info(f"Starting comprehensive analysis for call {call_sid}")
        
        # Download recording
        twilio_service = get_twilio_service()
        recordings_dir = settings.recordings_dir
        local_path = recordings_dir / f"{call_sid}.wav"
        
        print(f"DEBUG: Downloading recording from {recording_url} to {local_path}")
        success = await twilio_service.download_recording(
            recording_url,
            str(local_path)
        )
        
        if not success:
            print(f"DEBUG: Download failed for {call_sid}")
            logger.error(f"Failed to download recording for {call_sid}")
            return
            
        print("DEBUG: Download successful. Loading model hub...")
        # Run Analysis
        hub = get_model_hub()
        
        print("DEBUG: Running full analysis (Respiratory, Parkinson's, Depression)...")
        # We run all available screenings
        result = hub.run_full_analysis(
            str(local_path),
            enable_respiratory=settings.enable_respiratory_screening,
            enable_parkinsons=settings.enable_parkinsons_screening,
            enable_depression=settings.enable_depression_screening
        )
        
        print(f"DEBUG: Analysis complete. Risk Level: {result.overall_risk_level}")
        print(f"DEBUG: Primary Concern: {result.primary_concern}")
        logger.info(f"Analysis complete for {call_sid}. Risk: {result.overall_risk_level}")

        # Extract respiratory screening result safely
        respiratory_screening = result.screenings.get("respiratory")

        # Retrieve questionnaire data from cache
        questionnaire_data = questionnaire_cache.pop(call_sid, None)
        print(f"DEBUG: Retrieved questionnaire data: {questionnaire_data}")
        logger.info(f"Questionnaire data for {call_sid}: {questionnaire_data}")

        # Save results to database
        from app.database.database import async_session_maker
        from app.database.models import CallRecord, ClassificationResult

        print("DEBUG: Saving results to database...")
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

            print(f"DEBUG: Database save successful. Call ID: {call_record.id}")
            logger.info(f"Saved results to database for call {call_sid}")

        # Format and send SMS
        print("DEBUG: Formatting SMS...")
        sms_message = format_sms_result(
             classification=respiratory_screening.details.get("sound_class", "unknown") if respiratory_screening else "unknown",
             confidence=respiratory_screening.confidence if respiratory_screening else 0.0,
             recommendation=result.recommendation,
             language=language,
             comprehensive_result=result
        )

        print(f"DEBUG: Sending SMS to {caller_number}: {sms_message[:50]}...")
        twilio_service.send_sms(caller_number, sms_message)
        print("DEBUG: SMS sent.")
        
        # Also send brief medical report for doctor (as second message)
        print("DEBUG: Generating medical report...")
        from app.services.twilio_service import format_medical_report
        try:
            medical_report = format_medical_report(
                comprehensive_result=result,
                patient_phone=caller_number,
                report_type="brief"
            )
            
            # Send medical report with header
            intro_msg = "📋 MEDICAL REPORT FOR DOCTOR:\n" + "="*40 + "\n"
            twilio_service.send_sms(caller_number, intro_msg + medical_report)
            print("DEBUG: Medical report sent.")
        except Exception as report_error:
            logger.warning(f"Could not send medical report: {report_error}")

    except Exception as e:
        print(f"DEBUG: CRITICAL ERROR in analysis: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"Analysis failed for {call_sid}: {e}")


# ==================
# Webhook Endpoints
# ==================

@router.post("/voice/incoming")
async def incoming_call(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
    FromCity: Optional[str] = Form(None),
    FromState: Optional[str] = Form(None),
    FromCountry: Optional[str] = Form(None),
):
    """
    Main Entry Point for Incoming Calls.
    
    Optimized to directly invoke the Voice Agent (Conversational AI) flow
    without HTTP redirects, ensuring lowest latency.
    """
    logger.info(f"Incoming call from {From}. Route: Voice Agent (Direct Invocation)")
    
    # Direct invocation of Voice Agent Service
    # This avoids the 302 Redirect latency loop
    from app.api.voice_agent_webhooks import voice_agent_start
    return await voice_agent_start(
        request=request,
        CallSid=CallSid,
        From=From,
        FromCity=FromCity,
        FromState=FromState,
        FromCountry=FromCountry
    )


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
