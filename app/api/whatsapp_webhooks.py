"""
WhatsApp Webhooks
Twilio webhook endpoints for WhatsApp messaging flow
"""
import logging
from typing import Optional

from fastapi import APIRouter, Form, Request, BackgroundTasks
from fastapi.responses import Response, FileResponse

from app.config import settings
from app.services.whatsapp_service import get_whatsapp_service
from app.services.channel_service import (
    get_channel_service,
    Channel,
    ConversationState,
    MessageType
)
from app.ml.model_hub import get_model_hub
from app.utils.i18n import get_text

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/incoming")
async def handle_incoming_whatsapp(
    background_tasks: BackgroundTasks,
    request: Request,
    MessageSid: str = Form(...),
    From: str = Form(...),
    Body: Optional[str] = Form(None),
    MediaUrl0: Optional[str] = Form(None),
    MediaContentType0: Optional[str] = Form(None),
    NumMedia: int = Form(0),
):
    """
    Handle incoming WhatsApp message.
    Supports text commands and audio/voice notes.
    """
    # Normalize phone number (remove whatsapp: prefix for internal logic)
    phone_number = From.replace("whatsapp:", "")
    
    logger.info(f"WhatsApp incoming: SID={MessageSid}, From={From}, Body={Body}, Media={NumMedia}")
    
    whatsapp_service = get_whatsapp_service()
    channel_service = get_channel_service()
    
    # Get or create session
    session = channel_service.get_or_create_session(
        session_id=phone_number,  # Use phone number as persistent session ID for WhatsApp
        channel=Channel.WHATSAPP,
        phone_number=phone_number
    )
    
    # Determine message type
    if NumMedia > 0 and MediaContentType0:
        if "audio" in MediaContentType0:
            message_type = MessageType.AUDIO
        elif "image" in MediaContentType0:
            message_type = MessageType.IMAGE
        else:
            message_type = MessageType.TEXT  # Fallback
    else:
        message_type = MessageType.TEXT
    
    # ---------------------------------------------------------
    # ROUTING LOGIC
    # ---------------------------------------------------------
    
    # 1. Handle Voice Notes (Audio)
    if message_type == MessageType.AUDIO and MediaUrl0:
        return await _handle_audio_message(
            background_tasks, session, From, MediaUrl0
        )
    
    # 2. Handle Text Commands
    body_text = (Body or "").strip().lower()
    
    # 2a. Language Selection
    if body_text in ["1", "english", "en"]:
        session.language = "en"
        session.update_state(ConversationState.AWAITING_AUDIO)
        
        msg = (
            "Great! I've set your language to English. 🇬🇧\n\n"
            "To screen your health, simply **record a voice note** of you coughing. "
            "I'll analyze it for respiratory issues."
        )
        whatsapp_service.send_text(From, msg)
        return Response(status_code=200)
        
    elif body_text in ["2", "hindi", "hi", "हिंदी"]:
        session.language = "hi"
        session.update_state(ConversationState.AWAITING_AUDIO)
        
        msg = (
            "धन्यवाद! मैंने आपकी भाषा हिंदी सेट कर दी है। 🇮🇳\n\n"
            "जांच करने के लिए, कृपया **खांसते हुए एक वॉयस नोट** भेजें। "
            "मैं इसकी जांच करूंगा।"
        )
        whatsapp_service.send_text(From, msg)
        return Response(status_code=200)

    # 2b. Global Commands (Restart/Help/Menu)
    if body_text in ["hi", "hello", "help", "start", "restart", "menu", "नमस्ते", "main menu"]:
        session.update_state(ConversationState.LANGUAGE_SELECT)
        
        welcome_msg = settings.whatsapp_welcome_message
        
        buttons = [
            {"id": "english", "title": "English"},
            {"id": "hindi", "title": "हिंदी (Hindi)"}
        ]
        
        whatsapp_service.send_interactive(
            to=From,
            body=welcome_msg,
            buttons=buttons,
            footer="Select Language / भाषा चुनें"
        )
        return Response(status_code=200)
    
    # 2c. Default/Unknown
    if session.state == ConversationState.AWAITING_AUDIO:
        # User sent text instead of audio
        if session.language == "hi":
            msg = "⚠️ कृपया टेक्स्ट न भेजें।\nजांच करने के लिए, **माइक्रोफोन बटन** दबाएं और अपनी खांसी रिकॉर्ड करें। 🎙️"
        else:
            msg = "⚠️ Please do not send text.\nTo screen your health, press the **microphone button** and record your cough. 🎙️"
            
        whatsapp_service.send_text(From, msg)
    else:
        # Send menu help for other states
        whatsapp_service.send_text(
            From, 
            "I didn't understand that command. \n\nReply:\n• **'Menu'** to restart\n• **'Help'** for instructions"
        )
            
    return Response(status_code=200)


@router.post("/status")
async def handle_whatsapp_status(
    MessageSid: str = Form(...),
    MessageStatus: str = Form(...),
):
    """Handle message delivery status callbacks"""
    logger.debug(f"Message {MessageSid} status: {MessageStatus}")
    return Response(status_code=200)


async def _handle_audio_message(
    background_tasks: BackgroundTasks,
    session,
    user_number: str,
    media_url: str
):
    """
    Process incoming voice note:
    1. Download OGG
    2. Convert to WAV
    3. Run ML Analysis
    4. Send Results
    """
    whatsapp_service = get_whatsapp_service()
    
    # Notify processing
    if session.language == "hi":
        processing_msg = "प्राप्त हुआ! आपकी खांसी का विश्लेषण कर रहा हूँ... ⏳"
    else:
        processing_msg = "Received! Analyzing your cough... ⏳"
        
    whatsapp_service.send_text(user_number, processing_msg)
    
    # Process in background
    background_tasks.add_task(
        _process_audio_background,
        session,
        user_number,
        media_url
    )
    
    return Response(status_code=200)


async def _process_audio_background(session, user_number: str, media_url: str):
    """Background task for audio processing"""
    whatsapp_service = get_whatsapp_service()
    
    try:
        # 1. Setup paths
        timestamp = session.session_id + "_" + str(int(None or 0)) # simplified
        local_ogg = settings.recordings_dir / f"wa_{session.session_id}_{timestamp}.ogg"
        local_wav = settings.recordings_dir / f"wa_{session.session_id}_{timestamp}.wav"
        
        # 2. Download
        await whatsapp_service.download_media(media_url, str(local_ogg))
        
        # 3. Convert
        wav_path = whatsapp_service.convert_ogg_to_wav(str(local_ogg), str(local_wav))
        
        if not wav_path:
            raise ValueError("Audio conversion failed")
            
        # 4. Run Analysis
        hub = get_model_hub()
        result = await hub.run_full_analysis_async(
            wav_path,
            enable_respiratory=True,
            enable_parkinsons=True,  # Can enable/disable based on user preference
            enable_depression=True
        )
        
        # 5. Send Results (Health Card)
        whatsapp_service.send_health_card(
            user_number,
            result,
            language=session.language
        )
        
        # Update state
        session.update_state(ConversationState.RESULTS_DELIVERED)
        
    except Exception as e:
        logger.error(f"Error processing WhatsApp audio: {e}")
        
        if session.language == "hi":
            err_msg = "क्षमा करें, ऑडियो प्रक्रिया में त्रुटि हुई। कृपया पुन: प्रयास करें।"
        else:
            err_msg = "Sorry, I couldn't process that audio. Please try sending it again."
            
        whatsapp_service.send_text(user_number, err_msg)
