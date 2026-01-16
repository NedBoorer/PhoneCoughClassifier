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
from app.utils.twiml_helpers import twiml_response
from app.services.twilio_service import get_twilio_service, format_sms_result
from app.api.twilio_webhooks import run_comprehensive_analysis
from app.ml.model_hub import get_model_hub
from app.services.kisan_mitra_service import get_kisan_mitra_service

logger = logging.getLogger(__name__)

router = APIRouter()


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

@router.post("/voice/router")
async def incoming_call_router(
    CallSid: str = Form(...),
    From: str = Form(...),
    To: str = Form(...)
):
    """
    Smart router that directs calls to appropriate service based on number called.

    Use this as the main webhook endpoint in Twilio configuration.
    It will route to:
    - Health screening: if called on TWILIO_PHONE_NUMBER
    - Market service: if called on TWILIO_MARKET_PHONE_NUMBER
    """
    logger.info(f"Routing call: SID={CallSid}, From={From}, To={To}")

    response = VoiceResponse()

    # Normalize phone numbers for comparison (remove +, spaces, etc.)
    def normalize_number(num: str) -> str:
        return ''.join(filter(str.isdigit, num))

    to_normalized = normalize_number(To)
    health_number = normalize_number(settings.twilio_phone_number) if settings.twilio_phone_number else ""
    market_number = normalize_number(settings.twilio_market_phone_number) if settings.twilio_market_phone_number else ""

    # Route based on which number was called
    if market_number and to_normalized.endswith(market_number[-10:]):
        # Called the market service number → Mandi Bol
        logger.info(f"Routing to Mandi Bol market service")
        response.redirect(f"{settings.base_url}/india/voice/market/menu")
    else:
        # Default to health screening service
        logger.info(f"Routing to health screening service")
        response.redirect(f"{settings.base_url}/india/voice/incoming")

    return twiml_response(response)

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
    query_params = dict(request.query_params)
    attempt = int(query_params.get("lang_attempt", 0))

    logger.info(f"India incoming call: SID={CallSid}, From={From}, Callback={is_callback}, LangAttempt={attempt}")

    response = VoiceResponse()

    # First call: Full greeting
    if attempt == 0:
        greeting_text = (
            "Namaste. I am your health companion. "
            "नमस्ते. मैं आपकी स्वास्थ्य सहेली हूँ. "
        )
        response.say(greeting_text, voice="Polly.Aditi", language="en-IN")
        response.pause(length=1)

    # Timeout once: Give more language options
    elif attempt == 1:
        response.say(
            "Please select your language. कृपया अपनी भाषा चुनें.",
            voice="Polly.Aditi",
            language="en-IN"
        )

    # Timeout twice: Default to Hindi with explanation
    elif attempt >= 2:
        response.say(
            "No selection received. Continuing in Hindi. "
            "कोई चयन नहीं मिला। हिंदी में जारी रखते हैं।",
            voice="Polly.Aditi",
            language="en-IN"
        )
        response.redirect(f"{settings.base_url}/india/voice/start-recording?lang=hi")
        return twiml_response(response)

    # Language selection with Gather
    gather = Gather(
        num_digits=1,
        action=f"{settings.base_url}/india/voice/language-selected",
        timeout=10,
        input="dtmf"
    )

    # Bilingual language options
    if attempt == 0:
        gather.say(
            "For English, press 1. "
            "हिंदी के लिए 2 दबाएं. "
            "Health Workers, press 9.",
            voice="Polly.Aditi",
            language="en-IN"
        )
    else:
        # Second attempt: More concise
        gather.say(
            "Press 1 for English. 2 for Hindi. 9 for ASHA mode. "
            "1 अंग्रेजी के लिए। 2 हिंदी के लिए। 9 आशा मोड के लिए।",
            voice="Polly.Aditi",
            language="en-IN"
        )

    response.append(gather)

    # Increment attempt and redirect
    response.redirect(f"{settings.base_url}/india/voice/incoming?lang_attempt={attempt + 1}")

    return twiml_response(response)


@router.post("/voice/market/menu")
async def market_menu(request: Request):
    """
    Mandi Bol: Market Price Check Menu
    Entry point for checking prices with optional wellness check.
    """
    response = VoiceResponse()

    # Check if user has already consented
    query_params = dict(request.query_params)
    consented = query_params.get("consent") == "yes"

    if not consented:
        # First time: Get consent for wellness check
        consent_gather = Gather(
            num_digits=1,
            action=f"{settings.base_url}/india/voice/market/consent",
            timeout=10
        )
        consent_gather.say(
            "Namaste. Welcome to Mandi Bol, your market price service. "
            "This service is free and can also offer wellness support for farmers. "
            "Press 1 to continue and get today's prices.",
            voice="Polly.Aditi",
            language="en-IN"
        )
        response.append(consent_gather)
        response.say("I did not receive an input. Goodbye.", voice="Polly.Aditi")
        response.hangup()
        return twiml_response(response)

    # After consent: Show commodity menu
    gather = Gather(
        num_digits=1,
        action=f"{settings.base_url}/india/voice/market/price",
        timeout=10
    )

    gather.say(
        "For Onion prices, press 1. "
        "For Tomato prices, press 2. "
        "For Potato prices, press 3. "
        "For Wheat prices, press 4. "
        "For Rice prices, press 5. ",
        voice="Polly.Aditi",
        language="en-IN"
    )

    response.append(gather)
    response.say("I did not receive an input. Goodbye.", voice="Polly.Aditi")
    response.hangup()

    return twiml_response(response)


@router.post("/voice/market/consent")
async def market_consent(Digits: str = Form(None)):
    """Handle consent for Mandi Bol service"""
    response = VoiceResponse()

    if Digits == '1':
        # User consented, redirect to commodity menu
        response.redirect(f"{settings.base_url}/india/voice/market/menu?consent=yes")
    else:
        response.say("Thank you. Goodbye.", voice="Polly.Aditi", language="en-IN")
        response.hangup()

    return twiml_response(response)


@router.post("/voice/market/price")
async def market_price_immediate(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
    Digits: str = Form(None)
):
    """Provide market price immediately, no recording needed"""
    commodity_map = {
        '1': 'onion', '2': 'tomato', '3': 'potato', '4': 'wheat', '5': 'rice'
    }

    commodity = commodity_map.get(Digits, 'onion')
    response = VoiceResponse()

    # Get market price from service
    kisan_service = get_kisan_mitra_service()
    price_info = kisan_service.get_market_price(commodity)

    # Provide price immediately
    response.say(price_info, voice="Polly.Aditi", language="en-IN")
    response.pause(length=1)

    # Now optionally collect wellness sample
    response.say(
        "Thank you for using Mandi Bol. "
        "As part of our farmer wellness program, may I ask how you are feeling today? "
        "After the beep, please share in a few words. This is optional and helps us support farmers.",
        voice="Polly.Aditi",
        language="en-IN"
    )

    response.record(
        max_length=15,
        timeout=3,
        play_beep=True,
        trim="trim-silence",
        action=f"{settings.base_url}/india/voice/market/wellness-complete",
        recording_status_callback=f"{settings.base_url}/twilio/voice/recording-status"
    )

    return twiml_response(response)


@router.post("/voice/market/wellness-complete")
async def market_wellness_complete(
    background_tasks: BackgroundTasks,
    CallSid: str = Form(...),
    From: str = Form(...),
    RecordingUrl: str = Form(None)
):
    """Handle wellness check completion - run analysis in background"""
    response = VoiceResponse()

    response.say("Thank you for sharing. Stay strong. Jai Hind.", voice="Polly.Aditi", language="en-IN")
    response.hangup()

    # Run depression screening in background if recording provided
    if RecordingUrl:
        background_tasks.add_task(
            process_wellness_check,
            call_sid=CallSid,
            caller_number=From,
            recording_url=RecordingUrl
        )

    return twiml_response(response)


async def process_wellness_check(call_sid: str, caller_number: str, recording_url: str):
    """Background task: Analyze wellness check and send intervention if needed"""
    try:
        # Download recording
        twilio_service = get_twilio_service()
        local_path = settings.recordings_dir / f"{call_sid}_wellness.wav"
        await twilio_service.download_recording(recording_url, str(local_path))

        # Run depression screening
        hub = get_model_hub()
        result = await hub.run_full_analysis_async(
            str(local_path),
            enable_respiratory=False,
            enable_parkinsons=False,
            enable_depression=True
        )

        # Check if intervention needed
        kisan_service = get_kisan_mitra_service()
        depression_screening = result.screenings.get("depression")
        should_intervene, reason = kisan_service.check_intervention_needed(depression_screening)

        if should_intervene:
            # Send SMS with counselor helpline
            intervention_msg = (
                f"Namaste. We noticed you may be going through a difficult time. "
                f"You are not alone. Please call Kisan Call Center at 1800-180-1551 for free support. "
                f"Your well-being matters. - Mandi Bol Farmer Wellness"
            )
            twilio_service.send_sms(caller_number, intervention_msg)
            logger.info(f"Wellness intervention SMS sent to {caller_number}: {reason}")

    except Exception as e:
        logger.error(f"Wellness check processing failed for {call_sid}: {e}")


@router.post("/voice/market/selection")
async def market_selection(Digits: str = Form(None)):
    """Handle commodity selection and prompt for voice sample (Passive Screen)"""
    response = VoiceResponse()
    
    commodity_map = {
        '1': 'onion', '2': 'tomato', '3': 'potato', '4': 'wheat', '5': 'rice'
    }
    
    commodity = commodity_map.get(Digits, 'onion')
    
    # "To get the best price for [Crop], please describe your crop quality after the beep."
    # This ensures we get a speech sample for depression screening.
    response.say(
        f"You selected {commodity}. "
        "To give you the accurate price, please describe your crop quality in a few sentences after the beep. "
        "Speak for at least 10 seconds.",
        voice="Polly.Aditi",
        language="en-IN"
    )
    
    response.record(
        max_length=15,
        timeout=3,
        play_beep=True,
        trim="trim-silence",
        action=f"{settings.base_url}/india/voice/market/analyze?commodity={commodity}",
        recording_status_callback=f"{settings.base_url}/twilio/voice/recording-status"
    )
    
    return twiml_response(response)


@router.post("/voice/market/analyze")
async def market_analyze(
    request: Request,
    background_tasks: BackgroundTasks,
    CallSid: str = Form(...),
    RecordingUrl: str = Form(None)
):
    """
    Analyze voice for Depression (Passive) -> Market Price (Active)
    """
    commodity = request.query_params.get("commodity", "onion")
    
    response = VoiceResponse()
    
    if RecordingUrl:
        response.say("Please wait while we check the market data.", voice="Polly.Aditi")
        
        # 1. Download & Analyze (Synchronous for MVP flow decision, though usually async is better)
        # We need the result NOW to decide on intervention.
        try:
            twilio_service = get_twilio_service()
            local_path = settings.recordings_dir / f"{CallSid}_market.wav"
            await twilio_service.download_recording(RecordingUrl, str(local_path))
            
            hub = get_model_hub()
            # Enable Depression Screening!
            result = await hub.run_full_analysis_async(
                str(local_path),
                enable_respiratory=False,
                enable_parkinsons=False,
                enable_depression=True 
            )
            
            # 2. Check Intervention
            kisan_service = get_kisan_mitra_service()
            should_intervene, reason = kisan_service.check_intervention_needed(result.screenings.get("depression"))
            
            if should_intervene:
                # Redirect to Intervention Flow
                response.redirect(f"{settings.base_url}/india/voice/kisan-mitra/handover?reason={reason}")
                return twiml_response(response)
            
            # 3. Normal Market Flow
            price_info = kisan_service.get_market_price(commodity)
            response.say(price_info, voice="Polly.Aditi", language="en-IN")
            
            response.say("Thank you for using Mandi Bol. Jai Hind.", voice="Polly.Aditi")
            response.hangup()
            
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            response.say("Error checking prices. Please try again later.", voice="Polly.Aditi")
            response.hangup()
    else:
        response.say("No audio received.", voice="Polly.Aditi")
        response.hangup()

    return twiml_response(response)


@router.post("/voice/kisan-mitra/handover")
async def kisan_mitra_handover(request: Request):
    """
    Kisan Mitra Intervention Flow
    """
    reason = request.query_params.get("reason", "")
    kisan_service = get_kisan_mitra_service()
    
    # Get empathetic script
    script = kisan_service.get_empathetic_message(reason, language="en")
    
    response = VoiceResponse()
    
    gather = Gather(
        num_digits=1,
        action=f"{settings.base_url}/india/voice/kisan-mitra/connect",
        timeout=10
    )
    gather.say(script, voice="Polly.Aditi", language="en-IN")
    
    response.append(gather)
    
    # If no input, just hangup softly or go back to market? 
    # Let's say goodbye softly.
    response.say("Take care of yourself. Goodbye.", voice="Polly.Aditi")
    response.hangup()
    
    return twiml_response(response)


@router.post("/voice/kisan-mitra/connect")
async def kisan_mitra_connect(Digits: str = Form(None)):
    """Connect to Counselor"""
    response = VoiceResponse()
    
    if Digits == '1':
        response.say("Connecting you to a Kisan Mitra counselor now...", voice="Polly.Aditi")
        # Mock number for counselor
        response.dial("1800-180-1551") # Kisan Call Center
    else:
        response.say("No problem. Stay strong. Goodbye.", voice="Polly.Aditi")
    
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
                respiratory_screening = result.screenings.get("respiratory")
                class_result = ClassificationResult(
                    call_id=call_record.id,
                    classification=respiratory_screening.details.get("sound_class", "unknown") if respiratory_screening else "unknown",
                    confidence=respiratory_screening.confidence if respiratory_screening else 0.0,
                    probabilities=respiratory_screening.details.get("probabilities", {}) if respiratory_screening else {},
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
                        from datetime import datetime, timedelta
                        expiry_date = (datetime.now() + timedelta(days=7)).strftime("%d %b %Y")

                        if language == 'en':
                            report_text += (
                                f"\n\n🎫 PRIORITY REFERRAL TICKET\n"
                                f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                f"Ticket #: {referral_code}\n"
                                f"Valid Until: {expiry_date}\n"
                                f"Action Required: Visit your nearest Primary Health Center (PHC) within 24 hours\n"
                                f"Instructions: Show this ticket to the doctor for priority consultation\n"
                                f"\n⚕️ Your health matters. Please don't delay."
                            )
                        else:
                            report_text += (
                                f"\n\n🎫 प्राथमिकता रेफरल टिकट\n"
                                f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                f"टिकट नंबर: {referral_code}\n"
                                f"मान्यता: {expiry_date} तक\n"
                                f"आवश्यक कार्रवाई: 24 घंटे के भीतर अपने निकटतम स्वास्थ्य केंद्र जाएं\n"
                                f"निर्देश: प्राथमिकता परामर्श के लिए यह टिकट डॉक्टर को दिखाएं\n"
                                f"\n⚕️ आपका स्वास्थ्य महत्वपूर्ण है। कृपया देरी न करें।"
                            )
                    
                    twilio_service.send_whatsapp_with_media(recipient_number, report_text, media_url)
                except Exception as e:
                    logger.error(f"WhatsApp generation failed: {e}")

            # 6. Audio Feedback
            if is_asha:
                # ASHA mode: Give specific guidance based on risk level
                if risk_high:
                    if referral_code:
                        response.say(
                            f"URGENT: High risk detected. Priority ticket {referral_code} sent to patient. "
                            "Please ensure patient visits the clinic today.",
                            voice="Polly.Aditi", language="en-IN"
                        )
                    else:
                        response.say(
                            f"High risk detected. Report sent to patient. Please follow up to ensure doctor visit.",
                            voice="Polly.Aditi", language="en-IN"
                        )
                else:
                    response.say(
                        f"Screening complete. Risk level is {result.overall_risk_level}. Report sent to patient.",
                        voice="Polly.Aditi", language="en-IN"
                    )
                # Loop back for next patient
                response.redirect(f"{settings.base_url}/india/voice/asha/menu")
                return twiml_response(response)

            elif risk_high:
                # Non-ASHA high risk: Connect to doctor
                if referral_code:
                    if language == 'en':
                        ticket_msg = (
                            f"Important: I have detected health risks that need medical attention. "
                            f"I am sending you a priority referral ticket {referral_code} via WhatsApp. "
                            f"Please visit your nearest Primary Health Center within 24 hours and show this ticket to the doctor. "
                            f"This ticket is valid for 7 days."
                        )
                    else:
                        ticket_msg = (
                            f"महत्वपूर्ण: मुझे स्वास्थ्य जोखिम मिले हैं जिनके लिए चिकित्सा की आवश्यकता है। "
                            f"मैं आपको प्राथमिकता टिकट {referral_code} व्हाट्सएप पर भेज रहा हूं। "
                            f"कृपया 24 घंटे के भीतर अपने निकटतम स्वास्थ्य केंद्र जाएं और यह टिकट डॉक्टर को दिखाएं। "
                            f"यह टिकट 7 दिनों के लिए मान्य है।"
                        )
                    response.say(ticket_msg, voice=voice, language=lang_code)

                # Tele-Triage Bridge
                if language == 'en':
                    alert_text = "I am now connecting you to a health counselor. Please hold."
                else:
                    alert_text = "मैं अब आपको स्वास्थ्य परामर्शदाता से जोड़ रहा हूं। कृपया प्रतीक्षा करें।"
                response.say(alert_text, voice=voice, language=lang_code)
                response.dial(settings.doctor_helpline_number)
                return twiml_response(response)
            
            else:
                # Normal Result - Start Family Loop
                response.say(
                    "Your results are normal. Stay healthy.",
                    voice=voice, language=lang_code
                )

                response.pause(length=1)

                # Get current loop count
                loop_count = int(query_params.get("family_loop", 0))

                # Maximum 3 family members
                if loop_count >= 3:
                    response.say(
                        "You have screened multiple family members. That is very good. Stay healthy." if language == 'en' else
                        "आपने कई परिवार के सदस्यों की जांच करवाई। बहुत अच्छा। स्वस्थ रहें.",
                        voice=voice, language=lang_code
                    )
                    response.hangup()
                    return twiml_response(response)

                # Family Loop Gather with progressive messaging
                gather = Gather(
                    num_digits=1,
                    action=f"{settings.base_url}/india/voice/family-decision?lang={language}&family_loop={loop_count}",
                    timeout=5
                )

                # Progressive messaging based on loop count
                if loop_count == 0:
                    family_msg = (
                        "Is anyone else in your family coughing? Press 1 to check them now." if language == 'en' else
                        "क्या आपके घर में कोई और खांस रहा है? उनकी जांच के लिए 1 दबाएं."
                    )
                elif loop_count == 1:
                    family_msg = (
                        "Anyone else? Press 1 for one more person." if language == 'en' else
                        "कोई और? एक और व्यक्ति के लिए 1 दबाएं."
                    )
                else:  # loop_count == 2
                    family_msg = (
                        "Last one? Press 1 to screen one more family member." if language == 'en' else
                        "आखिरी? एक और सदस्य की जांच के लिए 1 दबाएं."
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
    loop_count = int(query_params.get("family_loop", 0))

    response = VoiceResponse()

    if Digits == '1':
        # Increment loop counter and redirect back to recording
        next_loop = loop_count + 1
        response.redirect(f"{settings.base_url}/india/voice/start-recording?lang={language}&family_loop={next_loop}")
    else:
        response.say("Namaste. Goodbye.", voice="Polly.Aditi", language="en-IN")
        response.hangup()

    return twiml_response(response)


@router.post("/voice/asha/menu")
async def asha_menu(request: Request):
    """ASHA Worker Menu: Enter Patient ID"""
    response = VoiceResponse()

    # Track attempt count to prevent infinite loops
    query_params = dict(request.query_params)
    attempt = int(query_params.get("attempt", 0))

    # Max 5 attempts before auto-exit
    if attempt >= 5:
        response.say(
            "Maximum attempts reached. Exiting ASHA mode. Thank you for your service.",
            voice="Polly.Aditi",
            language="en-IN"
        )
        response.hangup()
        return twiml_response(response)

    gather = Gather(
        num_digits=10,
        finish_on_key="#",
        action=f"{settings.base_url}/india/voice/asha/patient-entry",
        timeout=10
    )
    gather.say(
        "ASHA Mode. Please enter the 10 digit mobile number of the patient, followed by the hash key. "
        "Or press star star to exit ASHA mode.",
        voice="Polly.Aditi",
        language="en-IN"
    )
    response.append(gather)

    # Increment attempt counter on redirect
    response.redirect(f"{settings.base_url}/india/voice/asha/menu?attempt={attempt + 1}")
    return twiml_response(response)


@router.post("/voice/asha/patient-entry")
async def asha_patient_entry(
    request: Request,
    Digits: str = Form(None),
):
    """Confirm patient and start screening"""
    response = VoiceResponse()

    # Check for exit signal (* or **)
    if Digits and ('*' in Digits):
        response.say(
            "Exiting ASHA mode. Thank you for your service. Namaste.",
            voice="Polly.Aditi",
            language="en-IN"
        )
        response.hangup()
        return twiml_response(response)

    # Validate patient number
    if not Digits or len(Digits) < 10:
        response.say(
            "Invalid number. Please try again.",
            voice="Polly.Aditi",
            language="en-IN"
        )
        response.redirect(f"{settings.base_url}/india/voice/asha/menu")
        return twiml_response(response)

    # Get language preference (default to Hindi for rural areas, allow override via query param)
    query_params = dict(request.query_params)
    patient_lang = query_params.get("lang", "hi")

    response.say(f"Patient number {Digits}. Starting screening.", voice="Polly.Aditi")

    response.redirect(f"{settings.base_url}/india/voice/start-recording?lang={patient_lang}&patient_id={Digits}&is_asha=true")

    return twiml_response(response)
