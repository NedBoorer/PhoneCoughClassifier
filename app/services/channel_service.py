"""
Channel Service
Unified abstraction layer for Voice Calls, WhatsApp, and SMS
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class Channel(Enum):
    """Communication channel types"""
    VOICE_CALL = "voice"
    WHATSAPP = "whatsapp"
    SMS = "sms"


class MessageType(Enum):
    """Types of incoming messages"""
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    BUTTON_REPLY = "button_reply"
    DTMF = "dtmf"  # Phone keypad input


class ConversationState(Enum):
    """User conversation states"""
    INITIAL = "initial"
    LANGUAGE_SELECT = "language_select"
    AWAITING_AUDIO = "awaiting_audio"
    PROCESSING = "processing"
    RESULTS_DELIVERED = "results_delivered"
    FAMILY_PROMPT = "family_prompt"
    ENDED = "ended"


@dataclass
class ChannelMessage:
    """
    Unified message format across all channels.
    
    Attributes:
        channel: The communication channel (voice, whatsapp, sms)
        sender: Sender's phone number (E.164 format)
        message_type: Type of content (text, audio, etc.)
        content: Text content (for text messages)
        audio_path: Local path to audio file (after download/conversion)
        media_url: Original media URL (for audio/image messages)
        language: User's preferred language
        session_id: Unique session/call identifier
        timestamp: When the message was received
        raw_data: Original webhook payload for debugging
    """
    channel: Channel
    sender: str
    message_type: MessageType
    content: Optional[str] = None
    audio_path: Optional[str] = None
    media_url: Optional[str] = None
    language: str = "en"
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    raw_data: Optional[dict] = None


@dataclass
class ChannelSession:
    """
    Track user session state across channels.
    
    Attributes:
        session_id: Unique identifier (CallSid or MessageSid)
        channel: Communication channel
        phone_number: User's phone number
        language: Selected language
        state: Current conversation state
        screening_type: Type of screening requested
        audio_paths: List of audio files collected
        results: Screening results if completed
        created_at: Session start time
        updated_at: Last activity time
    """
    session_id: str
    channel: Channel
    phone_number: str
    language: str = "en"
    state: ConversationState = ConversationState.INITIAL
    screening_type: Optional[str] = None
    audio_paths: list[str] = field(default_factory=list)
    results: Optional[dict] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def update_state(self, new_state: ConversationState):
        """Update conversation state with timestamp"""
        self.state = new_state
        self.updated_at = datetime.now()


class ChannelService:
    """
    Unified channel management service.
    
    Handles session tracking and provides channel-agnostic methods
    for health screening flows.
    """
    
    def __init__(self):
        # In-memory session storage (use Redis in production)
        self._sessions: dict[str, ChannelSession] = {}
    
    def get_or_create_session(
        self,
        session_id: str,
        channel: Channel,
        phone_number: str,
        language: str = "en"
    ) -> ChannelSession:
        """
        Get existing session or create a new one.
        
        Args:
            session_id: Unique identifier (CallSid, MessageSid)
            channel: Communication channel
            phone_number: User's phone number
            language: Preferred language
            
        Returns:
            ChannelSession instance
        """
        # Look up by phone number for WhatsApp (conversations span messages)
        if channel == Channel.WHATSAPP:
            for sid, session in self._sessions.items():
                if (session.channel == Channel.WHATSAPP and 
                    session.phone_number == phone_number and
                    session.state != ConversationState.ENDED):
                    session.updated_at = datetime.now()
                    return session
        
        # For voice calls, always use CallSid
        if session_id in self._sessions:
            return self._sessions[session_id]
        
        # Create new session
        session = ChannelSession(
            session_id=session_id,
            channel=channel,
            phone_number=phone_number,
            language=language
        )
        self._sessions[session_id] = session
        logger.info(f"Created new {channel.value} session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ChannelSession]:
        """Get session by ID"""
        return self._sessions.get(session_id)
    
    def get_session_by_phone(
        self,
        phone_number: str,
        channel: Optional[Channel] = None
    ) -> Optional[ChannelSession]:
        """
        Find active session by phone number.
        
        Args:
            phone_number: User's phone number
            channel: Optional channel filter
            
        Returns:
            Most recent active session, or None
        """
        matching = []
        for session in self._sessions.values():
            if session.phone_number == phone_number:
                if channel is None or session.channel == channel:
                    if session.state != ConversationState.ENDED:
                        matching.append(session)
        
        if not matching:
            return None
        
        # Return most recently updated
        return max(matching, key=lambda s: s.updated_at)
    
    def end_session(self, session_id: str):
        """Mark session as ended"""
        if session_id in self._sessions:
            self._sessions[session_id].update_state(ConversationState.ENDED)
            logger.info(f"Ended session: {session_id}")
    
    def clear_session(self, session_id: str):
        """Remove session from memory"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Cleared session: {session_id}")
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """
        Remove sessions older than max_age_hours.
        Should be called periodically (e.g., via background task).
        """
        now = datetime.now()
        expired = []
        
        for sid, session in self._sessions.items():
            age = (now - session.updated_at).total_seconds() / 3600
            if age > max_age_hours:
                expired.append(sid)
        
        for sid in expired:
            del self._sessions[sid]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")


# Singleton instance
_service: Optional[ChannelService] = None


def get_channel_service() -> ChannelService:
    """Get singleton channel service instance"""
    global _service
    if _service is None:
        _service = ChannelService()
    return _service
