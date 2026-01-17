"""
Voice Agent Webhooks
Twilio webhook endpoints for conversational voice agent flow
"""
import logging
import json
from typing import Optional

from fastapi import APIRouter, Form, Request, BackgroundTasks
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Gather

from app.config import settings
from app.utils.twiml_helpers import twiml_response
from app.services.voice_agent_service import (
    get_voice_agent_service,
    ConversationStep,
)
from app.services.twilio_service import get_twilio_service, format_sms_result
from app.ml.model_hub import get_model_hub
from app.utils.i18n import get_language_config
from app.database.database import async_session_maker
from app.database.models import CallRecord, ClassificationResult

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory state cache (in production, use Redis)
_call_states: dict[str, dict] = {}
_state_lock = None  # Lazy init to avoid import issues

# Background analysis results cache
_analysis_results: dict[str, dict] = {}  # {call_sid: {"status": "processing|complete|error", "result": ...}}

# Constants for call flow control
MAX_NO_INPUT_ATTEMPTS = 5
MAX_RECORDING_ATTEMPTS = 3
MAX_FAMILY_SCREENINGS = 5
MAX_RESULTS_POLL_ATTEMPTS = 3  # Max times to check for results


def _get_state_lock():
    """Get or create thread lock for state management"""
    global _state_lock
    if _state_lock is None:
        import threading
        _state_lock = threading.Lock()
    return _state_lock


def _get_voice_config(language: str) -> tuple[str, str]:
    """Get Twilio voice and language code for a language"""
    config = get_language_config(language)
    if config:
        return config.twilio_voice, config.twilio_lang
    return "Polly.Aditi", "en-IN"


def _get_call_info(call_sid: str) -> dict:
    """Get call info with defaults"""
    return _call_states.get(call_sid, {})


def _cleanup_call(call_sid: str):
    """Clean up call state, agent state, and recordings"""
    logger.info(f"Cleaning up state for {call_sid}")

    # Clean up agent state
    agent = get_voice_agent_service()
    agent.clear_state(call_sid)

    # Thread-safe removal from call states and analysis results
    lock = _get_state_lock()
    with lock:
        _call_states.pop(call_sid, None)
        _analysis_results.pop(call_sid, None)

    # CRITICAL: Clean up audio files to prevent memory leak
    # Recordings accumulate and fill disk if not deleted
    try:
        import os
        recording_path = settings.recordings_dir / f"{call_sid}_agent.wav"
        if recording_path.exists():
            os.remove(recording_path)
            logger.info(f"Deleted recording: {recording_path}")
    except Exception as e:
        logger.warning(f"Failed to delete recording for {call_sid}: {e}")


def _get_health_tips(language: str, duration_seconds: int = 10) -> list[str]:
    """
    Get health education messages to play during analysis.

    These keep the user engaged while ML models run in background.
    Messages are timed to fill the expected analysis duration.

    Args:
        language: Language code (en/hi)
        duration_seconds: Target duration in seconds

    Returns:
        List of messages (each ~3-4 seconds when spoken)
    """
    if language == "hi":
        tips = [
            "क्या आप जानते हैं? रोज़ाना गहरी सांस लेने से फेफड़े मज़बूत होते हैं।",
            "अगर खांसी दो हफ्ते से ज़्यादा रहे, तो डॉक्टर को ज़रूर दिखाएं।",
            "टीबी का इलाज मुफ्त और पूरी तरह संभव है। DOTS सेंटर पर जाएं।",
            "धूम्रपान और धुआं से दूर रहें। आपके फेफड़े आपको धन्यवाद देंगे।",
        ]
    else:
        tips = [
            "Did you know? Deep breathing exercises daily can strengthen your lungs.",
            "If your cough lasts more than two weeks, please see a doctor.",
            "TB is completely curable with free treatment available at DOTS centers.",
            "Avoid smoking and smoke exposure. Your lungs will thank you.",
        ]

    # Return subset based on duration (each tip ~3-4 seconds)
    num_tips = min(len(tips), max(2, duration_seconds // 3))
    return tips[:num_tips]


@router.post("/start")
async def voice_agent_start(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
    FromCity: Optional[str] = Form(None),
    FromState: Optional[str] = Form(None),
    FromCountry: Optional[str] = Form(None),
):
    """
    Start a conversational voice agent call.
    
    This is the entry point for the voice agent flow.
    """
    logger.info(f"Voice agent start: SID={CallSid}, From={From}, Location={FromCity}, {FromState}")
    
    # Get query params for language override
    query_params = dict(request.query_params)
    language = query_params.get("lang", "en")
    
    # Initialize conversation state
    agent = get_voice_agent_service()
    state = agent.get_or_create_state(CallSid)
    state.language = language
    
    # Store caller info with tracking counters (thread-safe)
    lock = _get_state_lock()
    with lock:
        _call_states[CallSid] = {
            "caller_number": From,
            "language": language,
            "city": FromCity,
            "state": FromState,
            "country": FromCountry,
            "recording_attempts": 0,
            "family_screenings": 0,
        }
    
    # Get initial greeting
    greeting = agent.get_initial_greeting(language)
    voice, lang_code = _get_voice_config(language)
    
    response = VoiceResponse()
    
    # Speak greeting
    response.say(greeting, voice=voice, language=lang_code)
    
    # Gather speech input
    gather = Gather(
        input="speech dtmf",  # Accept both speech and keypad
        action=f"./process-speech?lang={language}&attempt=0",
        timeout=settings.voice_agent_timeout if hasattr(settings, 'voice_agent_timeout') else 8,
        speech_timeout="auto",
        language=lang_code,
        hints="farmer, kisan, cough, khasi, yes, no, haan, nahi",
    )
    
    response.append(gather)
    
    # If no input, try again (max 2 attempts before fallback)
    response.redirect(f"./no-input?lang={language}&attempt=1")
    
    return twiml_response(response)


@router.post("/process-speech")
async def process_speech(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
    SpeechResult: Optional[str] = Form(None),
    Digits: Optional[str] = Form(None),
    Confidence: Optional[float] = Form(None),
):
    """
    Process user speech input and generate response.
    """
    query_params = dict(request.query_params)
    language = query_params.get("lang", "en")
    attempt = int(query_params.get("attempt", 0))
    
    # Get user input (speech or DTMF)
    user_input = SpeechResult or ""
    if Digits:
        # Convert DTMF to text
        dtmf_map = {
            "1": "yes",
            "2": "no",
            "9": "asha mode",
            "*": "help",
        }
        user_input = dtmf_map.get(Digits, Digits)
    
    # Check for max turn limit - guide to recording instead of hanging up
    agent = get_voice_agent_service()
    state = agent.get_or_create_state(CallSid)
    max_turns = getattr(settings, 'voice_agent_max_turns', 15)
    if state.turn_count >= max_turns:
        logger.info(f"Max turns reached for {CallSid}, guiding to cough recording")
        voice, lang_code = _get_voice_config(language)
        response = VoiceResponse()
        response.say(
            "Let me help you right away. I'll listen to your cough now." if language == "en"
            else "मैं अभी आपकी मदद करता हूं। मैं आपकी खांसी सुनता हूं।",
            voice=voice,
            language=lang_code
        )
        # Skip to recording instead of hanging up
        state.current_step = ConversationStep.RECORDING
        response.redirect(f"./continue?lang={language}&step=recording")
        return twiml_response(response)
    
    logger.info(f"Voice agent speech: SID={CallSid}, Input='{user_input}', Confidence={Confidence}")
    
    # Process input with voice agent (already initialized above)
    try:
        agent_response = await agent.process_user_input(
            call_sid=CallSid,
            user_input=user_input,
            language=language,
        )
    except Exception as e:
        logger.error(f"Voice agent processing failed for {CallSid}: {type(e).__name__}: {e}", exc_info=True)
        # Fallback response - try to continue gracefully
        voice, lang_code = _get_voice_config(language)
        response = VoiceResponse()
        response.say(
            "I'm sorry, I had trouble understanding. Let me help you with your cough check instead." if language == "en"
            else "माफ़ कीजिए, मुझे समझने में दिक्कत हुई। आइए मैं आपकी खांसी की जांच करता हूं।",
            voice=voice,
            language=lang_code
        )
        # Fallback to recording instead of looping
        response.redirect(f"./continue?lang={language}&step=recording")
        return twiml_response(response)
    
    voice, lang_code = _get_voice_config(language)
    response = VoiceResponse()
    
    # Special handling for recording step - should_record already checks for RECORDING states
    if agent_response.should_record:
        logger.info(f"Transitioning to recording for {CallSid}, step={agent_response.next_step}")
        return await _handle_recording_request(response, CallSid, language, voice, lang_code)
    
    # Special handling for goodbye
    if agent_response.should_end:
        response.say(agent_response.message, voice=voice, language=lang_code)
        response.hangup()
        # Clean up state
        _cleanup_call(CallSid)
        return twiml_response(response)
    
    # Special handling for ASHA handoff - ask for confirmation first
    if agent_response.next_step == ConversationStep.ASHA_HANDOFF:
        response.say(
            "Would you like me to connect you to a health worker? Press 1 or say yes to connect." 
            if language == "en" else 
            "क्या आप स्वास्थ्य कार्यकर्ता से बात करना चाहते हैं? जोड़ने के लिए 1 दबाएं या हां कहें।",
            voice=voice, 
            language=lang_code
        )
        gather = Gather(
            input="speech dtmf",
            action=f"./confirm-handoff?lang={language}",
            timeout=8,
            num_digits=1,
            speech_timeout="auto",
            language=lang_code,
            hints="yes, no, haan, nahi, 1, 2",
        )
        response.append(gather)
        # If no response, continue with cough recording instead
        response.redirect(f"./continue?lang={language}&step=recording")
        return twiml_response(response)
    
    # Normal response - continue conversation
    response.say(agent_response.message, voice=voice, language=lang_code)
    
    # Gather next input (reset attempt counter on successful speech)
    gather = Gather(
        input="speech dtmf",
        action=f"./process-speech?lang={language}&attempt=0",
        timeout=settings.voice_agent_timeout if hasattr(settings, 'voice_agent_timeout') else 8,
        speech_timeout="auto",
        language=lang_code,
        hints="farmer, kisan, cough, khasi, yes, no, haan, nahi, pesticide, dust",
    )
    
    response.append(gather)
    
    # Fallback if no input - increment attempt
    response.redirect(f"./no-input?lang={language}&attempt=1")
    
    return twiml_response(response)


async def _handle_recording_request(
    response: VoiceResponse,
    call_sid: str,
    language: str,
    voice: str,
    lang_code: str
) -> Response:
    """Handle the cough recording step with encouragement"""

    # Recording instruction (dynamic, encouraging, clear)
    duration = settings.max_recording_duration
    if language == "hi":
        instruction = (
            "बहुत अच्छा! अब मुझे आपकी खांसी सुननी है। "
            "बीप के बाद, अपनी सामान्य खांसी जैसे खांसें। "
            f"बस {duration} सेकंड। कोई चिंता नहीं, आराम से करें।"
        )
    else:
        instruction = (
            "Perfect! Now I need to hear your cough. "
            "After the beep, cough naturally like you normally do. "
            f"Just {duration} seconds. Don't worry, take your time and relax."
        )

    response.say(instruction, voice=voice, language=lang_code)

    response.pause(length=1)

    # UX NOTE: During recording, we CAN'T play audio overlay as it would be
    # captured in the recording and interfere with ML analysis.
    # Instead, we give clear, encouraging instructions BEFORE recording,
    # and engage them with health tips AFTER recording while analysis runs.

    # Record cough
    # Note: trim="trim-silence" removed to prevent dropping quiet recordings
    response.record(
        max_length=settings.max_recording_duration,
        timeout=3,
        play_beep=True,
        action=f"./recording-complete?lang={language}",
        recording_status_callback=f"{settings.base_url}/twilio/voice/recording-status",
    )

    return twiml_response(response)


@router.post("/recording-complete")
async def recording_complete(
    background_tasks: BackgroundTasks,
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
    RecordingUrl: Optional[str] = Form(None),
):
    """
    Handle completed cough recording.
    Analyze and provide conversational results.
    """
    query_params = dict(request.query_params)
    language = query_params.get("lang", "en")
    
    logger.info(f"Voice agent recording complete: SID={CallSid}")
    
    voice, lang_code = _get_voice_config(language)
    response = VoiceResponse()
    
    if not RecordingUrl:
        # Track recording attempts to prevent infinite loops (thread-safe)
        lock = _get_state_lock()
        with lock:
            call_info = _call_states.get(CallSid, {})
            recording_attempts = call_info.get("recording_attempts", 0) + 1

        if recording_attempts >= MAX_RECORDING_ATTEMPTS:
            logger.warning(f"Max recording attempts ({MAX_RECORDING_ATTEMPTS}) reached for {CallSid}")
            response.say(
                "I'm having trouble hearing your cough. Please try calling back later. Goodbye!"
                if language == "en" else
                "मुझे आपकी खांसी सुनने में दिक्कत हो रही है। कृपया बाद में फिर से कॉल करें। अलविदा!",
                voice=voice,
                language=lang_code
            )
            response.hangup()
            _cleanup_call(CallSid)
            return twiml_response(response)

        # Update attempt counter (thread-safe)
        with lock:
            if CallSid in _call_states:
                _call_states[CallSid]["recording_attempts"] = recording_attempts

        logger.info(f"No recording received, attempt {recording_attempts}/{MAX_RECORDING_ATTEMPTS}")
        response.say(
            "I didn't hear anything. Let me try again." if language == "en"
            else "मुझे कुछ सुनाई नहीं दिया। मैं फिर से कोशिश करता हूं।",
            voice=voice,
            language=lang_code
        )
        response.redirect(f"./continue?lang={language}&step=recording")
        return twiml_response(response)
    
    # Thank you message immediately after recording
    response.say(
        "Thank you!" if language == "en" else "धन्यवाद!",
        voice=voice,
        language=lang_code
    )
    
    response.pause(length=1)

    # Start background analysis immediately
    # This allows us to talk to the user while processing happens
    logger.info(f"Starting background analysis for {CallSid}, RecordingUrl={RecordingUrl}")

    # Mark analysis as processing
    lock = _get_state_lock()
    with lock:
        _analysis_results[CallSid] = {"status": "processing", "result": None, "error": None}

    # Start analysis in background task
    background_tasks.add_task(
        _run_background_analysis,
        CallSid=CallSid,
        RecordingUrl=RecordingUrl,
        language=language,
    )

    # UX IMPROVEMENT: Keep talking while analysis runs!
    # Instead of awkward silence, provide health education
    response.say(
        "Thank you! Let me analyze your cough while I share some health information with you."
        if language == "en" else
        "धन्यवाद! मैं आपकी खांसी की जांच करता हूं। इस दौरान मैं आपको कुछ स्वास्थ्य जानकारी देता हूं।",
        voice=voice,
        language=lang_code
    )

    response.pause(length=1)

    # Share health tips to fill ~10 seconds while analysis runs
    health_tips = _get_health_tips(language, duration_seconds=10)
    for tip in health_tips:
        response.say(tip, voice=voice, language=lang_code)
        response.pause(length=1)

    # By now, analysis should be complete or nearly complete
    # Redirect to check results
    response.redirect(f"./check-results?lang={language}&attempt=1")

    return twiml_response(response)


async def _run_background_analysis(CallSid: str, RecordingUrl: str, language: str):
    """
    Run ML analysis in background while user hears health tips.

    This decouples analysis from TwiML response, enabling better UX.
    Results are stored in _analysis_results for retrieval.
    """
    lock = _get_state_lock()

    try:
        logger.info(f"Background analysis starting for {CallSid}")
        twilio_service = get_twilio_service()
        local_path = settings.recordings_dir / f"{CallSid}_agent.wav"

        # Download recording
        logger.debug(f"Downloading recording to {local_path}")
        await twilio_service.download_recording(RecordingUrl, str(local_path))

        # Run analysis
        logger.info(f"Running ML analysis for {CallSid}")
        hub = get_model_hub()
        result = await hub.run_full_analysis_async(
            str(local_path),
            enable_respiratory=settings.enable_respiratory_screening,
            enable_parkinsons=settings.enable_parkinsons_screening,
            enable_depression=settings.enable_depression_screening,
            enable_tuberculosis=settings.enable_tuberculosis_screening,
        )
        logger.info(f"Analysis complete for {CallSid}: risk={result.overall_risk_level}, time={result.processing_time_ms}ms")

        # Store results with RecordingUrl for database
        with lock:
            _analysis_results[CallSid] = {
                "status": "complete",
                "result": result,
                "error": None,
                "recording_url": RecordingUrl,  # Save for database
            }

    except Exception as e:
        logger.error(f"Background analysis failed for {CallSid}: {type(e).__name__}: {e}", exc_info=True)
        with lock:
            _analysis_results[CallSid] = {
                "status": "error",
                "result": None,
                "error": str(e),
            }


async def _send_report(caller_number: str, result, language: str):
    """Background task to send health report via WhatsApp/SMS"""
    try:
        twilio_service = get_twilio_service()
        
        report_text = format_sms_result(
            classification="N/A",
            confidence=0,
            recommendation=result.recommendation,
            language=language,
            comprehensive_result=result,
        )
        
        twilio_service.send_sms(caller_number, report_text)
        logger.info(f"Report sent to {caller_number}")
        
    except Exception as e:
        logger.error(f"Failed to send report: {e}")


@router.post("/family-decision")
async def family_decision(
    request: Request,
    CallSid: str = Form(...),
    SpeechResult: Optional[str] = Form(None),
    Digits: Optional[str] = Form(None),
):
    """Handle decision to screen another family member"""
    query_params = dict(request.query_params)
    language = query_params.get("lang", "en")
    
    user_input = (SpeechResult or "").lower()
    voice, lang_code = _get_voice_config(language)
    
    response = VoiceResponse()
    
    # Check for yes response
    yes_indicators = ["yes", "1", "haan", "हां", "one", "ek"]
    if Digits == "1" or any(ind in user_input for ind in yes_indicators):
        # Check family screening limit (thread-safe)
        lock = _get_state_lock()
        with lock:
            call_info = _call_states.get(CallSid, {})
            family_count = call_info.get("family_screenings", 0) + 1

        if family_count >= MAX_FAMILY_SCREENINGS:
            logger.info(f"Max family screenings ({MAX_FAMILY_SCREENINGS}) reached for {CallSid}")
            response.say(
                f"You've screened {MAX_FAMILY_SCREENINGS} people today. That's wonderful! Please call back tomorrow if you need to check more family members. Take care!"
                if language == "en" else
                f"आपने आज {MAX_FAMILY_SCREENINGS} लोगों की जांच की है। बहुत अच्छा! अगर और परिवार के सदस्यों की जांच करनी हो तो कल फिर कॉल करें। ख्याल रखें!",
                voice=voice,
                language=lang_code
            )
            response.redirect(f"./goodbye?lang={language}")
            return twiml_response(response)

        # Update family screening counter (thread-safe)
        with lock:
            if CallSid in _call_states:
                _call_states[CallSid]["family_screenings"] = family_count
                _call_states[CallSid]["recording_attempts"] = 0  # Reset recording attempts

        # Reset for next family member
        agent = get_voice_agent_service()
        state = agent.get_state(CallSid)
        if state:
            state.current_step = ConversationStep.RECORDING_INTRO
            state.turn_count = 0
        
        response.say(
            "Great! Let's check the next person. Please hand the phone to them."
            if language == "en" else
            "अच्छा! अगले व्यक्ति की जांच करते हैं। कृपया फोन उन्हें दें।",
            voice=voice,
            language=lang_code
        )
        
        response.pause(length=2)
        
        # Go to recording for next person
        response.redirect(f"./continue?lang={language}&step=recording")
    else:
        # End call
        response.redirect(f"./goodbye?lang={language}")
    
    return twiml_response(response)


@router.post("/confirm-handoff")
async def confirm_handoff(
    request: Request,
    CallSid: str = Form(...),
    SpeechResult: Optional[str] = Form(None),
    Digits: Optional[str] = Form(None),
):
    """Handle confirmation for doctor/ASHA handoff"""
    query_params = dict(request.query_params)
    language = query_params.get("lang", "en")
    
    user_input = (SpeechResult or "").lower()
    voice, lang_code = _get_voice_config(language)
    
    response = VoiceResponse()
    
    # Check for yes response
    yes_indicators = ["yes", "1", "haan", "हां", "one", "ek", "doctor", "connect"]
    if Digits == "1" or any(ind in user_input for ind in yes_indicators):
        response.say(
            "Connecting you to a health worker now. Please hold."
            if language == "en" else
            "आपको अभी स्वास्थ्य कार्यकर्ता से जोड़ रहा हूं। कृपया रुकें।",
            voice=voice,
            language=lang_code
        )
        # Redirect to ASHA menu
        response.redirect(f"{settings.base_url}/india/voice/asha/menu")
    else:
        # User said no - continue with cough check
        response.say(
            "No problem! Let's continue with your health check."
            if language == "en" else
            "कोई बात नहीं! आइए आपकी सेहत जांच जारी रखें।",
            voice=voice,
            language=lang_code
        )
        response.redirect(f"./continue?lang={language}&step=recording")
    
    return twiml_response(response)


@router.post("/continue")
async def continue_conversation(
    request: Request,
    CallSid: str = Form(...),
):
    """Continue the conversation from a specific step"""
    query_params = dict(request.query_params)
    language = query_params.get("lang", "en")
    step = query_params.get("step", "greeting")
    
    voice, lang_code = _get_voice_config(language)
    response = VoiceResponse()
    
    if step == "recording":
        return await _handle_recording_request(response, CallSid, language, voice, lang_code)
    
    # Default: continue with conversation
    gather = Gather(
        input="speech dtmf",
        action=f"./process-speech?lang={language}&attempt=0",
        timeout=10,
        speech_timeout="auto",
        language=lang_code,
    )
    
    gather.say(
        "Please continue." if language == "en" else "कृपया जारी रखें।",
        voice=voice,
        language=lang_code
    )
    
    response.append(gather)
    response.redirect(f"./no-input?lang={language}&attempt=1")

    return twiml_response(response)


@router.post("/no-input")
async def handle_no_input(
    request: Request,
    CallSid: str = Form(...),
):
    """Handle case when user doesn't provide input - progressive fallback"""
    query_params = dict(request.query_params)
    language = query_params.get("lang", "en")
    attempt = int(query_params.get("attempt", 1))
    
    voice, lang_code = _get_voice_config(language)
    response = VoiceResponse()
    
    logger.info(f"No-input handler: SID={CallSid}, attempt={attempt}, language={language}")

    # Safety: Prevent infinite loops by capping at max attempts (from config or constant)
    max_attempts = getattr(settings, 'max_no_input_attempts', MAX_NO_INPUT_ATTEMPTS)
    if attempt >= max_attempts:
        logger.warning(f"Max no-input attempts reached for {CallSid}, ending call")
        response.say(
            "I'm sorry, I couldn't hear you. Please call back when you're ready. Goodbye!"
            if language == "en" else
            "माफ़ कीजिए, मुझे सुनाई नहीं दिया। कृपया जब तैयार हों तब फिर से कॉल करें। अलविदा!",
            voice=voice,
            language=lang_code
        )
        response.hangup()
        _cleanup_call(CallSid)
        return twiml_response(response)

    if attempt >= 3:
        # After 3 attempts, guide to cough recording instead of hanging up
        response.say(
            "No problem! Let me just listen to your cough. Please cough after the beep."
            if language == "en" else
            "कोई बात नहीं! मैं बस आपकी खांसी सुनता हूं। बीप के बाद खांसें।",
            voice=voice,
            language=lang_code
        )
        # Guide to recording instead of hanging up
        response.redirect(f"./continue?lang={language}&step=recording")
        return twiml_response(response)
    
    # Prompt message varies by attempt to avoid repetition
    if attempt == 1:
        prompt = "I didn't catch that. Could you say that again?" if language == "en" else "मैंने सुना नहीं। क्या आप फिर से बोल सकते हैं?"
    else:
        prompt = "Please speak clearly or press 1 for yes, 2 for no." if language == "en" else "कृपया स्पष्ट बोलें या हां के लिए 1, ना के लिए 2 दबाएं।"
    
    # Say prompt first, then gather
    response.say(prompt, voice=voice, language=lang_code)
    
    # Gather with shorter timeout on retries
    gather = Gather(
        input="speech dtmf",
        action=f"./process-speech?lang={language}&attempt={attempt}",
        timeout=6,  # Shorter timeout on retries
        speech_timeout="auto",
        language=lang_code,
        hints="yes, no, haan, nahi, 1, 2",
    )
    
    response.append(gather)
    response.redirect(f"./no-input?lang={language}&attempt={attempt + 1}")
    
    return twiml_response(response)


@router.post("/check-results")
async def check_results(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
):
    """
    Check if background analysis is complete and present results.

    If still processing, say "almost ready" and poll again.
    This creates a smooth UX without awkward silence.
    """
    query_params = dict(request.query_params)
    language = query_params.get("lang", "en")
    attempt = int(query_params.get("attempt", 1))

    voice, lang_code = _get_voice_config(language)
    response = VoiceResponse()

    # Check analysis status
    lock = _get_state_lock()
    with lock:
        analysis_data = _analysis_results.get(CallSid, {})

    status = analysis_data.get("status", "unknown")
    logger.info(f"Check results for {CallSid}: status={status}, attempt={attempt}")

    if status == "complete":
        # Analysis done! Present results
        result = analysis_data.get("result")

        if not result:
            logger.error(f"Analysis marked complete but no result for {CallSid}")
            response.say(
                "I had trouble processing your results. Please try again."
                if language == "en" else
                "परिणाम की प्रक्रिया में दिक्कत हुई। कृपया फिर से कोशिश करें।",
                voice=voice,
                language=lang_code
            )
            response.redirect(f"./goodbye?lang={language}")
            return twiml_response(response)

        # Handle timeout case
        if result.primary_concern == "timeout":
            response.say(
                "The analysis took too long. Let me try with a shorter recording."
                if language == "en" else
                "विश्लेषण में बहुत समय लगा। छोटी रिकॉर्डिंग से कोशिश करें।",
                voice=voice,
                language=lang_code
            )
            response.redirect(f"./continue?lang={language}&step=recording")
            return twiml_response(response)

        # Present results (reusing existing logic)
        return await _present_results(response, CallSid, From, result, language, voice, lang_code)

    elif status == "error":
        # Analysis failed
        error = analysis_data.get("error", "Unknown error")
        logger.error(f"Analysis error for {CallSid}: {error}")

        error_msg_lower = str(error).lower()
        if "download" in error_msg_lower or "url" in error_msg_lower:
            response.say(
                "I couldn't access your recording. Please try again."
                if language == "en" else
                "मैं रिकॉर्डिंग तक नहीं पहुंच सका। कृपया फिर से कोशिश करें।",
                voice=voice,
                language=lang_code
            )
            response.redirect(f"./continue?lang={language}&step=recording")
        else:
            response.say(
                "I had trouble analyzing your cough. Please try calling back later."
                if language == "en" else
                "खांसी की जांच में दिक्कत हुई। कृपया बाद में फिर से कॉल करें।",
                voice=voice,
                language=lang_code
            )
            response.redirect(f"./goodbye?lang={language}")

        return twiml_response(response)

    else:
        # Still processing
        if attempt >= MAX_RESULTS_POLL_ATTEMPTS:
            # Taking too long, apologize and retry
            logger.warning(f"Analysis timeout for {CallSid} after {attempt} poll attempts")
            response.say(
                "The analysis is taking longer than expected. Let me try with a fresh recording."
                if language == "en" else
                "विश्लेषण में अधिक समय लग रहा है। नई रिकॉर्डिंग से कोशिश करते हैं।",
                voice=voice,
                language=lang_code
            )
            response.redirect(f"./continue?lang={language}&step=recording")
            return twiml_response(response)

        # Still processing, give positive feedback and check again
        response.say(
            "Almost ready, just one more moment..." if language == "en"
            else "बस हो गया, एक पल और...",
            voice=voice,
            language=lang_code
        )
        response.pause(length=2)
        response.redirect(f"./check-results?lang={language}&attempt={attempt + 1}")
        return twiml_response(response)


async def _present_results(response, CallSid, From, result, language, voice, lang_code):
    """Present analysis results to user (extracted for reuse)"""
    # --- SAVE TO DATABASE ---
    try:
        call_info = _call_states.get(CallSid, {})
        # Get recording URL from analysis results
        lock = _get_state_lock()
        with lock:
            analysis_data = _analysis_results.get(CallSid, {})
        recording_url = analysis_data.get("recording_url", "")

        async with async_session_maker() as db:
            # Create Call Record
            call_record = CallRecord(
                call_sid=CallSid,
                caller_number=call_info.get("caller_number", From),
                language=language,
                call_status="completed",
                recording_url=recording_url,
                city=call_info.get("city"),
                state=call_info.get("state"),
                country=call_info.get("country")
            )
            db.add(call_record)
            await db.flush()

            # Create Classification Result
            classification_str = result.primary_concern if result.primary_concern != "none" else "Normal"

            class_result = ClassificationResult(
                call_id=call_record.id,
                classification=classification_str,
                confidence=0.85 if result.overall_risk_level in ["high", "urgent"] else 0.95,
                severity=result.overall_risk_level,
                recommendation=result.recommendation,
                processing_time_ms=result.processing_time_ms
            )
            db.add(class_result)
            await db.commit()
            logger.info(f"Saved DB record for {CallSid}")
    except Exception as db_err:
        logger.error(f"Database save failed: {db_err}")
    # ------------------------

    # Get agent state for personalized response
    agent = get_voice_agent_service()
    state = agent.get_state(CallSid)

    # Generate personalized results message
    if state and hasattr(state, 'collected_info'):
        results_message = agent.get_results_message(
            state,
            result.overall_risk_level,
            result.recommendation,
        )
        state.current_step = ConversationStep.RESULTS
    else:
        # Fallback if state not found
        results_message = f"Your health check is complete. {result.recommendation}"

    response.say(results_message, voice=voice, language=lang_code)

    # Send WhatsApp/SMS report
    caller_number = call_info.get("caller_number", From)
    if settings.enable_whatsapp_reports:
        # Don't use background_tasks here, use a new task
        import asyncio
        asyncio.create_task(_send_report(caller_number, result, language))

    # Offer family screening
    response.pause(length=1)

    gather = Gather(
        input="speech dtmf",
        action=f"./family-decision?lang={language}",
        timeout=8,
        num_digits=1,
        speech_timeout="auto",
        language=lang_code,
    )

    gather.say(
        "Is there anyone else in your family who has a cough? Say yes or press 1 to check them too."
        if language == "en" else
        "क्या आपके परिवार में कोई और खांस रहा है? हां कहें या 1 दबाएं उनकी जांच के लिए।",
        voice=voice,
        language=lang_code
    )

    response.append(gather)

    # Default to goodbye if no response
    response.redirect(f"./goodbye?lang={language}")

    return twiml_response(response)


@router.post("/goodbye")
async def goodbye(
    request: Request,
    CallSid: str = Form(...),
):
    """End the call with a warm goodbye"""
    query_params = dict(request.query_params)
    language = query_params.get("lang", "en")

    voice, lang_code = _get_voice_config(language)

    response = VoiceResponse()

    response.say(
        "Thank you for calling! Take care of yourself and stay healthy. Goodbye!"
        if language == "en" else
        "कॉल करने के लिए धन्यवाद! अपना ख्याल रखें और स्वस्थ रहें। अलविदा!",
        voice=voice,
        language=lang_code
    )

    response.hangup()

    # Clean up state
    _cleanup_call(CallSid)

    return twiml_response(response)