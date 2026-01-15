"""
Phone Cough Classifier - India Accessibility Webhooks
Multi-language IVR flow for rural India accessibility
"""
import logging
from fastapi import APIRouter, Form, Request, BackgroundTasks
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Gather

from app.config import settings
from app.utils.i18n import get_text, get_language_config, LANGUAGES
from app.services.twilio_service import get_twilio_service
from app.api.twilio_webhooks import process_and_classify

logger = logging.getLogger(__name__)

router = APIRouter()


def twiml_response(twiml: VoiceResponse) -> Response:
    """Return TwiML as XML response"""
    return Response(content=str(twiml), media_type="application/xml")


# Map DTMF digits to language codes
LANGUAGE_MAP = {
    "1": "en",
    "2": "hi",
    "3": "ta",
    "4": "te",
    "5": "bn",
    "6": "mr",
    "7": "gu",
    "8": "kn",
    "9": "ml",
}


@router.post("/voice/incoming")
async def incoming_call_india(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
):
    """
    India-specific incoming call handler with language selection.
    Offers greeting in English and Hindi, then language selection.
    """
    logger.info(f"India incoming call: SID={CallSid}, From={From}")
    
    response = VoiceResponse()
    
    # Bilingual greeting (English + Hindi)
    response.say(
        "Welcome to the Cough Classifier. "
        "खांसी वर्गीकरण सेवा में आपका स्वागत है।",
        voice="Polly.Aditi",
        language="en-IN"
    )
    
    response.pause(length=1)
    
    # Language selection with Gather
    gather = Gather(
        num_digits=1,
        action=f"{settings.base_url}/india/voice/language-selected",
        timeout=10,
        input="dtmf"
    )
    
    gather.say(
        "Press 1 for English. "
        "हिंदी के लिए 2 दबाएं। "
        "தமிழுக்கு 3 அழுத்தவும்। "
        "తెలుగు కోసం 4 నొక్కండి। "
        "বাংলার জন্য 5 টিপুন।",
        voice="Polly.Aditi",
        language="en-IN"
    )
    
    response.append(gather)
    
    # Default to English if no input
    response.redirect(f"{settings.base_url}/india/voice/start-recording?lang=en")
    
    return twiml_response(response)


@router.post("/voice/language-selected")
async def language_selected(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
    Digits: str = Form(None),
):
    """Handle language selection"""
    logger.info(f"Language selected: SID={CallSid}, Digits={Digits}")
    
    # Get language from digit
    language = LANGUAGE_MAP.get(Digits, "en")
    lang_config = get_language_config(language)
    
    response = VoiceResponse()
    
    # Confirm language and proceed
    greeting = get_text("greeting", language)
    response.say(
        greeting,
        voice=lang_config.twilio_voice if lang_config else "Polly.Aditi",
        language=lang_config.twilio_lang if lang_config else "en-IN"
    )
    
    response.redirect(f"{settings.base_url}/india/voice/start-recording?lang={language}")
    
    return twiml_response(response)


@router.post("/voice/start-recording")
async def start_recording(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
    lang: str = "en"
):
    """Start the cough recording in selected language"""
    # Get lang from query params
    query_params = dict(request.query_params)
    language = query_params.get("lang", "en")
    
    logger.info(f"Start recording: SID={CallSid}, Lang={language}")
    
    lang_config = get_language_config(language)
    voice = lang_config.twilio_voice if lang_config else "Polly.Aditi"
    lang_code = lang_config.twilio_lang if lang_config else "en-IN"
    
    response = VoiceResponse()
    
    # Instruction in selected language
    instruction = get_text("record_instruction", language)
    response.say(instruction, voice=voice, language=lang_code)
    
    response.pause(length=1)
    
    cough_now = get_text("please_cough", language)
    response.say(cough_now, voice=voice, language=lang_code)
    
    # Record
    response.record(
        max_length=settings.max_recording_duration,
        timeout=3,
        play_beep=True,
        trim="trim-silence",
        action=f"{settings.base_url}/india/voice/recording-complete?lang={language}",
        recording_status_callback=f"{settings.base_url}/twilio/voice/recording-status"
    )
    
    return twiml_response(response)


@router.post("/voice/recording-complete")
async def recording_complete_india(
    background_tasks: BackgroundTasks,
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
    RecordingUrl: str = Form(None),
):
    """Handle recording completion in selected language"""
    query_params = dict(request.query_params)
    language = query_params.get("lang", "en")
    
    logger.info(f"Recording complete (India): SID={CallSid}, Lang={language}")
    
    lang_config = get_language_config(language)
    voice = lang_config.twilio_voice if lang_config else "Polly.Aditi"
    lang_code = lang_config.twilio_lang if lang_config else "en-IN"
    
    response = VoiceResponse()
    
    if RecordingUrl:
        # Thank them and explain next steps
        processing = get_text("processing", language)
        sms_coming = get_text("sms_coming", language)
        goodbye = get_text("goodbye", language)
        
        response.say(
            f"{processing} {sms_coming} {goodbye}",
            voice=voice,
            language=lang_code
        )
        
        # Start background processing
        background_tasks.add_task(
            process_and_classify,
            recording_url=RecordingUrl,
            caller_number=From,
            call_sid=CallSid,
            language=language
        )
    else:
        response.say(
            "I'm sorry, I didn't receive your recording. Please try again.",
            voice=voice,
            language=lang_code
        )
    
    response.hangup()
    
    return twiml_response(response)
