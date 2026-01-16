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

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory state cache (in production, use Redis)
_call_states: dict[str, dict] = {}


def _get_voice_config(language: str) -> tuple[str, str]:
    """Get Twilio voice and language code for a language"""
    config = get_language_config(language)
    if config:
        return config.twilio_voice, config.twilio_lang
    return "Polly.Aditi", "en-IN"


@router.post("/start")
async def voice_agent_start(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
):
    """
    Start a conversational voice agent call.
    
    This is the entry point for the voice agent flow.
    """
    logger.info(f"Voice agent start: SID={CallSid}, From={From}")
    
    # Get query params for language override
    query_params = dict(request.query_params)
    language = query_params.get("lang", "en")
    
    # Initialize conversation state
    agent = get_voice_agent_service()
    state = agent.get_or_create_state(CallSid)
    state.language = language
    
    # Store caller info
    _call_states[CallSid] = {
        "caller_number": From,
        "language": language,
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
        action=f"{settings.base_url}/voice-agent/process-speech?lang={language}",
        timeout=settings.voice_agent_timeout if hasattr(settings, 'voice_agent_timeout') else 10,
        speech_timeout="auto",
        language=lang_code,
        hints="farmer, kisan, cough, khasi, yes, no, haan, nahi",
    )
    
    # Fallback prompt if no input
    gather.say(
        "I'm listening..." if language == "en" else "मैं सुन रहा हूं...",
        voice=voice,
        language=lang_code
    )
    
    response.append(gather)
    
    # If no input, try again
    response.redirect(f"{settings.base_url}/voice-agent/no-input?lang={language}")
    
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
    
    logger.info(f"Voice agent speech: SID={CallSid}, Input='{user_input}', Confidence={Confidence}")
    
    # Process input with voice agent
    agent = get_voice_agent_service()
    
    try:
        agent_response = await agent.process_user_input(
            call_sid=CallSid,
            user_input=user_input,
            language=language,
        )
    except Exception as e:
        logger.error(f"Voice agent processing failed: {e}")
        # Fallback response
        voice, lang_code = _get_voice_config(language)
        response = VoiceResponse()
        response.say(
            "I'm sorry, I had trouble understanding. Let me try again." if language == "en" 
            else "माफ़ कीजिए, मुझे समझने में दिक्कत हुई। मैं फिर से कोशिश करता हूं।",
            voice=voice,
            language=lang_code
        )
        response.redirect(f"{settings.base_url}/voice-agent/continue?lang={language}")
        return twiml_response(response)
    
    voice, lang_code = _get_voice_config(language)
    response = VoiceResponse()
    
    # Special handling for recording step
    if agent_response.should_record:
        return await _handle_recording_request(response, CallSid, language, voice, lang_code)
    
    # Special handling for goodbye
    if agent_response.should_end:
        response.say(agent_response.message, voice=voice, language=lang_code)
        response.hangup()
        # Clean up state
        agent.clear_state(CallSid)
        _call_states.pop(CallSid, None)
        return twiml_response(response)
    
    # Normal response - continue conversation
    response.say(agent_response.message, voice=voice, language=lang_code)
    
    # Gather next input
    gather = Gather(
        input="speech dtmf",
        action=f"{settings.base_url}/voice-agent/process-speech?lang={language}",
        timeout=settings.voice_agent_timeout if hasattr(settings, 'voice_agent_timeout') else 10,
        speech_timeout="auto",
        language=lang_code,
        hints="farmer, kisan, cough, khasi, yes, no, haan, nahi, pesticide, dust",
    )
    
    gather.say(
        "..." if language == "en" else "...",  # Brief pause indicator
        voice=voice,
        language=lang_code
    )
    
    response.append(gather)
    
    # Fallback if no input
    response.redirect(f"{settings.base_url}/voice-agent/no-input?lang={language}")
    
    return twiml_response(response)


async def _handle_recording_request(
    response: VoiceResponse,
    call_sid: str,
    language: str,
    voice: str,
    lang_code: str
) -> Response:
    """Handle the cough recording step"""
    
    # Recording instruction
    if language == "hi":
        instruction = (
            "अब मुझे आपकी खांसी सुननी है। "
            "बीप के बाद, कृपया जोर से खांसें। "
            "मैं 10 सेकंड के लिए सुनूंगा।"
        )
    else:
        instruction = (
            "Now I need to hear your cough. "
            "After the beep, please cough clearly. "
            "I'll listen for about 10 seconds."
        )
    
    response.say(instruction, voice=voice, language=lang_code)
    
    response.pause(length=1)
    
    # Record cough
    response.record(
        max_length=settings.max_recording_duration,
        timeout=3,
        play_beep=True,
        trim="trim-silence",
        action=f"{settings.base_url}/voice-agent/recording-complete?lang={language}",
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
        response.say(
            "I didn't hear anything. Let me try again." if language == "en"
            else "मुझे कुछ सुनाई नहीं दिया। मैं फिर से कोशिश करता हूं।",
            voice=voice,
            language=lang_code
        )
        response.redirect(f"{settings.base_url}/voice-agent/continue?lang={language}&step=recording")
        return twiml_response(response)
    
    # Processing message
    response.say(
        "Thank you. Let me analyze your cough. Just a moment..." if language == "en"
        else "धन्यवाद। मैं आपकी खांसी की जांच करता हूं। एक पल रुकें...",
        voice=voice,
        language=lang_code
    )
    
    # Analyze recording
    try:
        twilio_service = get_twilio_service()
        local_path = settings.recordings_dir / f"{CallSid}_agent.wav"
        await twilio_service.download_recording(RecordingUrl, str(local_path))
        
        hub = get_model_hub()
        result = await hub.run_full_analysis_async(
            str(local_path),
            enable_respiratory=settings.enable_respiratory_screening,
            enable_parkinsons=settings.enable_parkinsons_screening,
            enable_depression=settings.enable_depression_screening,
        )
        
        # Get agent state for personalized response
        agent = get_voice_agent_service()
        state = agent.get_state(CallSid)
        
        # Generate personalized results message
        if state:
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
        caller_number = _call_states.get(CallSid, {}).get("caller_number", From)
        if settings.enable_whatsapp_reports:
            background_tasks.add_task(
                _send_report,
                caller_number,
                result,
                language,
            )
        
        # Offer family screening
        response.pause(length=1)
        
        gather = Gather(
            input="speech dtmf",
            action=f"{settings.base_url}/voice-agent/family-decision?lang={language}",
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
        response.redirect(f"{settings.base_url}/voice-agent/goodbye?lang={language}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        response.say(
            "I'm sorry, I had trouble analyzing your cough. Please try calling again later."
            if language == "en" else
            "माफ़ कीजिए, मुझे खांसी की जांच में दिक्कत हुई। कृपया बाद में फिर से कॉल करें।",
            voice=voice,
            language=lang_code
        )
        response.redirect(f"{settings.base_url}/voice-agent/goodbye?lang={language}")
    
    return twiml_response(response)


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
        response.redirect(f"{settings.base_url}/voice-agent/continue?lang={language}&step=recording")
    else:
        # End call
        response.redirect(f"{settings.base_url}/voice-agent/goodbye?lang={language}")
    
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
        action=f"{settings.base_url}/voice-agent/process-speech?lang={language}",
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
    response.redirect(f"{settings.base_url}/voice-agent/no-input?lang={language}")
    
    return twiml_response(response)


@router.post("/no-input")
async def handle_no_input(
    request: Request,
    CallSid: str = Form(...),
):
    """Handle case when user doesn't provide input"""
    query_params = dict(request.query_params)
    language = query_params.get("lang", "en")
    attempt = int(query_params.get("attempt", 0))
    
    voice, lang_code = _get_voice_config(language)
    response = VoiceResponse()
    
    if attempt >= 2:
        # Too many retries, fallback to DTMF or end
        response.say(
            "I'm having trouble hearing you. Let me transfer you to our other system."
            if language == "en" else
            "मुझे आपकी आवाज सुनने में दिक्कत हो रही है। मैं आपको दूसरी प्रणाली से जोड़ता हूं।",
            voice=voice,
            language=lang_code
        )
        # Fallback to regular DTMF flow
        response.redirect(f"{settings.base_url}/india/voice/incoming?lang={language}")
        return twiml_response(response)
    
    # Try again
    gather = Gather(
        input="speech dtmf",
        action=f"{settings.base_url}/voice-agent/process-speech?lang={language}",
        timeout=10,
        speech_timeout="auto",
        language=lang_code,
    )
    
    gather.say(
        "I didn't catch that. Could you please repeat?" if language == "en"
        else "मैंने सुना नहीं। क्या आप दोहरा सकते हैं?",
        voice=voice,
        language=lang_code
    )
    
    response.append(gather)
    response.redirect(f"{settings.base_url}/voice-agent/no-input?lang={language}&attempt={attempt + 1}")
    
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
    agent = get_voice_agent_service()
    agent.clear_state(CallSid)
    _call_states.pop(CallSid, None)
    
    return twiml_response(response)
