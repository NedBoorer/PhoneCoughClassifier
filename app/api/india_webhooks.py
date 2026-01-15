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
from app.services.twilio_service import get_twilio_service, format_sms_result
from app.api.twilio_webhooks import run_comprehensive_analysis
from app.ml.model_hub import get_model_hub

logger = logging.getLogger(__name__)

router = APIRouter()


def twiml_response(twiml: VoiceResponse) -> Response:
    """Return TwiML as XML response"""
    return Response(content=str(twiml), media_type="application/xml")


# Map DTMF digits to language codes
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

@router.post("/voice/missed-call")
async def missed_call_handler(
    CallSid: str = Form(...),
    From: str = Form(...),
    CallStatus: str = Form(...),
):
    """
    Handle 'Missed Call' to trigger callback.
    Twilio webhook should be configured to this endpoint.
    If call is 'ringing' or very short, we reject and call back.
    """
    response = VoiceResponse()
    
    # If it's a new incoming call (ringing), reject it to save cost to user
    if CallStatus == 'ringing':
        response.reject(reason='busy')
        
        # Trigger callback in background
        twilio = get_twilio_service()
        callback_url = f"{settings.base_url}/india/voice/incoming?is_callback=true"
        twilio.trigger_outbound_call(to=From, callback_url=callback_url)
        
        return twiml_response(response)
        
    return Response(status_code=200)


@router.post("/voice/incoming")
async def incoming_call_india(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
    is_callback: bool = False
):
    """
    India-specific incoming call handler with language selection.
    Offers greeting in English and Hindi, then language selection.
    """
    logger.info(f"India incoming call: SID={CallSid}, From={From}, Callback={is_callback}")
    
    response = VoiceResponse()
    
    # Storytelling Persona Greeting
    # "Namaste. I am your health friend. Do not worry about the cost, this call is for you."
    greeting_text = (
        "Namaste. I am your health companion. "
        "नमस्ते. मैं आपकी स्वास्थ्य सहेली हूँ. "
    )
    
    response.say(
        greeting_text,
        voice="Polly.Aditi",
        language="en-IN"
    )
    
    response.pause(length=1)
    
    # Language selection with Gather
    # Added Option 0 for Health Workers (ASHA)
    gather = Gather(
        num_digits=1,
        action=f"{settings.base_url}/india/voice/language-selected",
        timeout=10,
        input="dtmf"
    )
    
    gather.say(
        "For English, press 1. "
        "हिंदी के लिए 2 दबाएं. "
        "Health Workers, press 9.",
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
    if Digits == '9':
        # ASHA Mode
        return twiml_response(VoiceResponse().redirect(f"{settings.base_url}/india/voice/asha/menu"))

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
    """Handle recording completion: Analyze -> DB Save -> Golden Ticket -> Family Loop"""
    query_params = dict(request.query_params)
    language = query_params.get("lang", "en")
    is_asha = query_params.get("is_asha") == "true"
    patient_id = query_params.get("patient_id")
    
    logger.info(f"Recording complete (India): SID={CallSid}, Lang={language}, ASHA={is_asha}")
    
    lang_config = get_language_config(language)
    voice = lang_config.twilio_voice if lang_config else "Polly.Aditi"
    lang_code = lang_config.twilio_lang if lang_config else "en-IN"
    
    response = VoiceResponse()
    
    if RecordingUrl:
        # User feedback
        wait_msg = "Please wait a moment while I check your health." if language == 'en' else "कृपया अपनी जांच के लिए प्रतीक्षा करें"
        response.say(wait_msg, voice=voice, language=lang_code)
        
        try:
            # 1. Download
            twilio_service = get_twilio_service()
            local_path = settings.recordings_dir / f"{CallSid}.wav"
            await twilio_service.download_recording(RecordingUrl, str(local_path))
            
            # 2. Analyze
            hub = get_model_hub()
            result = await hub.run_full_analysis_async(
                str(local_path),
                enable_respiratory=True,
                enable_parkinsons=settings.enable_parkinsons_screening,
                enable_depression=settings.enable_depression_screening
            )
            
            # 3. Decision Logic & Golden Ticket
            risk_high = result.overall_risk_level in ["moderate", "high", "severe", "urgent"]
            referral_code = None
            
            import random
            if risk_high:
                referral_code = f"REF-{random.randint(1000, 9999)}"
                
            # 4. Save to DB (Using manual session management for webhook context)
            from app.database.database import async_session_maker
            from app.database.models import CallRecord, ClassificationResult
            
            async with async_session_maker() as db:
                # Create Call Record
                call_record = CallRecord(
                    call_sid=CallSid,
                    caller_number=From,
                    language=language,
                    call_status="completed",
                    recording_url=RecordingUrl
                )
                db.add(call_record)
                await db.flush() # get ID
                
                # Create Classification Result
                class_result = ClassificationResult(
                    call_id=call_record.id,
                    classification=result.screenings.get("respiratory", {}).details.get("sound_class", "unknown"),
                    confidence=result.screenings.get("respiratory", {}).confidence,
                    probabilities=result.screenings.get("respiratory", {}).probs,
                    severity=result.overall_risk_level,
                    recommendation=result.recommendation,
                    referral_code=referral_code
                )
                db.add(class_result)
                await db.commit()
            
            # 5. Reporting (WhatsApp/SMS)
            recipient_number = patient_id if is_asha and patient_id else From
            
            if settings.enable_whatsapp_reports and recipient_number:
                try:
                    # Generate Image
                    from app.utils.image_generator import generate_health_card
                    card_path = settings.data_dir / f"card_{CallSid}.png"
                    generate_health_card(result.overall_risk_level, result.recommendation, language, card_path)
                    
                    media_url = f"{settings.base_url}/data/card_{CallSid}.png"
                    
                    # Format Text
                    report_text = format_sms_result(
                        classification="N/A", confidence=0, recommendation=result.recommendation,
                        language=language, comprehensive_result=result
                    )
                    
                    if referral_code:
                        if language == 'en':
                            report_text += f"\n\n🎫 PRIORITY TICKET: {referral_code}\nShow this to the doctor at PHC."
                        else:
                            report_text += f"\n\n🎫 टिकट नंबर: {referral_code}\nडॉक्टर को यह दिखाएं."
                    
                    twilio_service.send_whatsapp_with_media(recipient_number, report_text, media_url)
                except Exception as e:
                    logger.error(f"WhatsApp generation failed: {e}")

            # 6. Audio Feedback
            if risk_high and not is_asha:
                if referral_code:
                    ticket_msg = (
                        f"I have sent a special priority ticket {referral_code} to your phone. "
                        "Please show this to the doctor."
                    )
                    response.say(ticket_msg, voice=voice, language=lang_code)
                
                # Tele-Triage Bridge
                alert_text = "Connecting you to a doctor now."
                response.say(alert_text, voice=voice, language=lang_code)
                response.dial(settings.doctor_helpline_number)
                return twiml_response(response)
                
            elif is_asha:
                response.say(
                    f"Screening complete. Risk is {result.overall_risk_level}. Report sent.",
                    voice="Polly.Aditi", language="en-IN"
                )
                # Loop back for ASHA
                response.redirect(f"{settings.base_url}/india/voice/asha/menu")
                return twiml_response(response)
            
            else:
                # Normal Result - Start Family Loop
                response.say(
                    "Your results are normal. Stay healthy.",
                    voice=voice, language=lang_code
                )
                
                response.pause(length=1)
                
                # Family Loop Gather
                gather = Gather(
                    num_digits=1,
                    action=f"{settings.base_url}/india/voice/family-decision?lang={language}",
                    timeout=5
                )
                
                family_msg = (
                    "Is anyone else in your family coughing? "
                    "Press 1 to check them now. " if language == 'en' else
                    "क्या आपके घर में कोई और खांस रहा है? उनकी जांच के लिए 1 दबाएं."
                )
                gather.say(family_msg, voice=voice, language=lang_code)
                response.append(gather)
                
                # If no input, hangup
                response.say("Goodbye.", voice=voice, language=lang_code)
                
        except Exception as e:
            logger.error(f"Sync analysis failed: {e}")
            response.say("Error in processing. We will text you.", voice=voice, language=lang_code)
    else:
        response.say("No recording received.", voice=voice, language=lang_code)
    
    response.hangup()
    return twiml_response(response)


@router.post("/voice/family-decision")
async def family_decision(
    request: Request,
    Digits: str = Form(None)
):
    """Handle decision to screen another family member"""
    query_params = dict(request.query_params)
    language = query_params.get("lang", "en")
    
    response = VoiceResponse()
    
    if Digits == '1':
        # Loop back to recording
        response.redirect(f"{settings.base_url}/india/voice/start-recording?lang={language}")
    else:
        response.say("Namaste. Goodbye.", voice="Polly.Aditi", language="en-IN")
        response.hangup()
        
    return twiml_response(response)


@router.post("/voice/asha/menu")
async def asha_menu(request: Request):
    """ASHA Worker Menu: Enter Patient ID"""
    response = VoiceResponse()
    
    gather = Gather(
        num_digits=10, 
        action=f"{settings.base_url}/india/voice/asha/patient-entry",
        timeout=10
    )
    gather.say(
        "ASHA Mode. Please enter the 10 digit mobile number of the patient.",
        voice="Polly.Aditi",
        language="en-IN"
    )
    response.append(gather)
    
    # Loop if no input
    response.redirect(f"{settings.base_url}/india/voice/asha/menu")
    return twiml_response(response)


@router.post("/voice/asha/patient-entry")
async def asha_patient_entry(
    Digits: str = Form(...),
):
    """Confirm patient and start screening"""
    response = VoiceResponse()
    
    response.say(f"Patient number {Digits}. Starting screening.", voice="Polly.Aditi")
    
    response.redirect(f"{settings.base_url}/india/voice/start-recording?lang=en&patient_id={Digits}&is_asha=true")
    
    return twiml_response(response)
