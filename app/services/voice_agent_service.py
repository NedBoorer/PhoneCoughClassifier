"""
Voice Agent Service
Conversational AI agent for personalized health screening interactions
"""
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from openai import OpenAI
import logging

from app.config import settings
from app.services.rag_service import get_rag_service
from app.utils.i18n import get_text

logger = logging.getLogger(__name__)


class ConversationStep(str, Enum):
    """Steps in the conversation flow"""
    GREETING = "greeting"
    LANGUAGE = "language"
    OCCUPATION = "occupation"
    SYMPTOMS = "symptoms"
    PESTICIDE_CHECK = "pesticide_check"
    DUST_CHECK = "dust_check"
    RECORDING_INTRO = "recording_intro"
    RECORDING = "recording"
    PROCESSING = "processing"
    RESULTS = "results"
    FAMILY_OFFER = "family_offer"
    ASHA_HANDOFF = "asha_handoff"
    GOODBYE = "goodbye"


@dataclass
class ConversationState:
    """State of an ongoing conversation"""
    call_sid: str
    language: str = "en"
    current_step: ConversationStep = ConversationStep.GREETING
    collected_info: dict = field(default_factory=dict)
    message_history: list = field(default_factory=list)
    turn_count: int = 0
    
    def to_dict(self) -> dict:
        """Serialize state for storage"""
        return {
            "call_sid": self.call_sid,
            "language": self.language,
            "current_step": self.current_step.value,
            "collected_info": self.collected_info,
            "message_history": self.message_history[-10:],  # Keep last 10 messages
            "turn_count": self.turn_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConversationState":
        """Deserialize state from storage"""
        return cls(
            call_sid=data["call_sid"],
            language=data.get("language", "en"),
            current_step=ConversationStep(data.get("current_step", "greeting")),
            collected_info=data.get("collected_info", {}),
            message_history=data.get("message_history", []),
            turn_count=data.get("turn_count", 0),
        )


@dataclass
class AgentResponse:
    """Response from the voice agent"""
    message: str  # Text to speak
    next_step: ConversationStep  # Next conversation step
    should_record: bool = False  # Whether to start recording
    should_end: bool = False  # Whether to end the call
    gathered_info: dict = field(default_factory=dict)  # Info extracted from user input


class VoiceAgentService:
    """
    Conversational voice agent for health screening.
    Uses OpenAI for natural language understanding and generation.
    """
    
    def __init__(self):
        self._client: Optional[OpenAI] = None
        self._conversations: dict[str, ConversationState] = {}
        self._response_cache: dict[str, str] = {}  # Cache for common responses
        self._cache_max_size = 50
        self._executor = ThreadPoolExecutor(max_workers=10)  # For concurrent OpenAI calls
    
    @property
    def client(self) -> OpenAI:
        """Lazy load OpenAI client"""
        if self._client is None:
            if not settings.openai_api_key:
                # If no key, return a dummy client or raise clearer error
                # For now assume mostly mocked or provided
                raise ValueError("OPENAI_API_KEY not configured")
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def get_or_create_state(self, call_sid: str) -> ConversationState:
        """Get existing conversation state or create new one"""
        if call_sid not in self._conversations:
            self._conversations[call_sid] = ConversationState(call_sid=call_sid)
        return self._conversations[call_sid]
    
    def get_state(self, call_sid: str) -> Optional[ConversationState]:
        """Get existing conversation state"""
        return self._conversations.get(call_sid)
    
    def clear_state(self, call_sid: str) -> None:
        """Clear conversation state after call ends"""
        self._conversations.pop(call_sid, None)
    
    def _get_system_prompt(self, state: ConversationState) -> str:
        """Get the system prompt for the LLM"""
        # Get RAG context based on current step
        rag = get_rag_service()
        
        topic_map = {
            ConversationStep.OCCUPATION: "farming",
            ConversationStep.PESTICIDE_CHECK: "pesticide",
            ConversationStep.DUST_CHECK: "dust",
            ConversationStep.SYMPTOMS: "respiratory",
            ConversationStep.RESULTS: "screening",
        }
        
        topic = topic_map.get(state.current_step, "general")
        context = rag.get_context_for_topic(topic, state.language, max_chunks=3)
        
        return f"""You are Swasth Saathi, a warm health companion on a phone call.
        You work with {settings.trusted_authority_name} to help people.
        Over {settings.mock_daily_users} people have used this service today.

LANGUAGE: {state.language}
STEP: {state.current_step.value}
INFO: {state.collected_info}

CRITICAL RULES:
- MAXIMUM 1-2 SHORT sentences (phone call, not text!)
- Ask ONE question at a time
- Warm but BRIEF (use "Ji" if Hindi)
- Never say AI/Robot - say "health service"
- Emergency → call 108
- If TB is mentioned, emphasize it's curable with free treatment (DOTS)

CONTEXT:
{context[:300]}

Reply with ONLY your short spoken response, no explanation."""

    def _determine_next_step(
        self,
        current_step: ConversationStep,
        user_input: str,
        extracted_info: dict
    ) -> ConversationStep:
        """Determine the next conversation step - always progresses toward recording"""
        
        # Check for Handoff Intents FIRST (Safety Net - always active)
        human_keywords = ["human", "person", "doctor", "talk to", "real person", "fake", "robot", "insaan", "baat karni", "asli", "asha"]
        if any(kw in user_input.lower() for kw in human_keywords):
            return ConversationStep.ASHA_HANDOFF
        
        # Fast-track to recording if user mentions cough - go directly to RECORDING
        cough_keywords = ["cough", "khansi", "खांसी", "record", "check", "coughing", "khaans", "khasi", "test", "screen"]
        if any(kw in user_input.lower() for kw in cough_keywords) and current_step not in [ConversationStep.RECORDING, ConversationStep.PROCESSING, ConversationStep.RESULTS]:
            return ConversationStep.RECORDING
        
        # Streamlined flow - always progress toward recording
        transitions = {
            ConversationStep.GREETING: ConversationStep.SYMPTOMS,  # Skip occupation, get to health faster
            ConversationStep.LANGUAGE: ConversationStep.SYMPTOMS,
            ConversationStep.OCCUPATION: ConversationStep.SYMPTOMS,
            ConversationStep.SYMPTOMS: ConversationStep.RECORDING_INTRO,  # Go straight to recording
            ConversationStep.PESTICIDE_CHECK: ConversationStep.RECORDING_INTRO,
            ConversationStep.DUST_CHECK: ConversationStep.RECORDING_INTRO,
            ConversationStep.RECORDING_INTRO: ConversationStep.RECORDING,
            ConversationStep.RECORDING: ConversationStep.PROCESSING,
            ConversationStep.PROCESSING: ConversationStep.RESULTS,
            ConversationStep.RESULTS: ConversationStep.FAMILY_OFFER,
            ConversationStep.FAMILY_OFFER: ConversationStep.RECORDING_INTRO,  # Loop back for more coughs!
            ConversationStep.GOODBYE: ConversationStep.FAMILY_OFFER,  # Never truly goodbye, offer more
        }

        return transitions.get(current_step, ConversationStep.RECORDING_INTRO)  # Default to recording
    
    def _extract_info(self, user_input: str, current_step: ConversationStep) -> dict:
        """Extract structured information from user input using LLM"""
        user_lower = user_input.lower()
        extracted = {}
        
        # Simple keyword extraction (fast path)
        if current_step == ConversationStep.OCCUPATION:
            farmer_keywords = ["farmer", "farming", "kisan", "खेती", "किसान", "farm", "agriculture"]
            if any(kw in user_lower for kw in farmer_keywords):
                extracted["is_farmer"] = True
            elif any(kw in user_lower for kw in ["no", "नहीं", "nahi", "not"]):
                extracted["is_farmer"] = False
        
        elif current_step == ConversationStep.PESTICIDE_CHECK:
            yes_keywords = ["yes", "हां", "haan", "use", "spray", "छिड़काव"]
            no_keywords = ["no", "नहीं", "nahi", "don't", "not"]
            if any(kw in user_lower for kw in yes_keywords):
                extracted["pesticide_exposure"] = True
            elif any(kw in user_lower for kw in no_keywords):
                extracted["pesticide_exposure"] = False
        
        elif current_step == ConversationStep.DUST_CHECK:
            yes_keywords = ["yes", "हां", "haan", "dust", "grain", "dhool", "धूल"]
            no_keywords = ["no", "नहीं", "nahi", "don't", "not"]
            if any(kw in user_lower for kw in yes_keywords):
                extracted["dust_exposure"] = True
            elif any(kw in user_lower for kw in no_keywords):
                extracted["dust_exposure"] = False
        
        elif current_step == ConversationStep.SYMPTOMS:
            # Check for symptom mentions
            if any(kw in user_lower for kw in ["chest", "pain", "dard", "दर्द", "seene"]):
                extracted["chest_pain"] = True
            if any(kw in user_lower for kw in ["breath", "saans", "सांस", "breathless"]):
                extracted["shortness_of_breath"] = True
            if any(kw in user_lower for kw in ["cough", "khansi", "खांसी"]):
                extracted["has_cough"] = True
        
        return extracted
    
    async def process_user_input(
        self,
        call_sid: str,
        user_input: str,
        language: Optional[str] = None
    ) -> AgentResponse:
        """
        Process user speech input and generate response.

        Args:
            call_sid: Twilio call SID
            user_input: Transcribed user speech
            language: Override language if detected

        Returns:
            AgentResponse with message and next step
        """
        # Validate inputs
        if not call_sid or not isinstance(call_sid, str):
            logger.error(f"Invalid call_sid: {call_sid}")
            raise ValueError("Invalid call_sid")

        # Handle empty or whitespace-only input
        if not user_input or not user_input.strip():
            logger.warning(f"Empty user input for {call_sid}, treating as silence")
            user_input = "[silence]"

        state = self.get_or_create_state(call_sid)

        # Update language if provided
        if language:
            state.language = language
        
        # Extract info from user input
        extracted = self._extract_info(user_input, state.current_step)
        state.collected_info.update(extracted)
        
        # Add user message to history
        state.message_history.append({"role": "user", "content": user_input})
        state.turn_count += 1
        
        # Get RAG context for current topic
        rag = get_rag_service()
        relevant_chunks = rag.query(user_input, top_k=2, language=state.language)
        
        # Build context from relevant chunks
        rag_context = "\n".join([c.content for c in relevant_chunks])
        
        # Generate response using LLM
        system_prompt = self._get_system_prompt(state)
        
        # Add context to system prompt if we have relevant chunks
        if rag_context:
            system_prompt += f"\n\nADDITIONAL CONTEXT FOR THIS RESPONSE:\n{rag_context}"
        
        try:
            # Check cache for similar inputs (reduces API calls and latency)
            cache_key = f"{state.current_step.value}:{state.language}:{user_input[:50]}"
            if cache_key in self._response_cache:
                message = self._response_cache[cache_key]
                logger.debug(f"Cache hit for: {cache_key[:30]}")
            else:
                # Build messages with error handling
                messages = [{"role": "system", "content": system_prompt}]
                # Only include valid message history
                for msg in state.message_history[-4:]:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        messages.append(msg)

                # Run OpenAI call in thread pool to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    self._executor,
                    lambda: self.client.chat.completions.create(
                        model=settings.voice_agent_model if hasattr(settings, 'voice_agent_model') else "gpt-4o-mini",
                        messages=messages,
                        max_tokens=60,
                        temperature=0.6,
                        timeout=10.0,
                    )
                )

                message = response.choices[0].message.content.strip()

                # Validate message isn't empty
                if not message:
                    logger.warning(f"Empty LLM response for {call_sid}, using fallback")
                    message = self._get_fallback_message(state)

                # Cache response with size limit
                if len(self._response_cache) >= self._cache_max_size:
                    self._response_cache.pop(next(iter(self._response_cache)))
                self._response_cache[cache_key] = message

        except Exception as e:
            logger.error(f"LLM call failed for {call_sid}: {type(e).__name__}: {e}")
            # Fallback message
            message = self._get_fallback_message(state)
        
        # Add assistant message to history
        state.message_history.append({"role": "assistant", "content": message})
        
        # Determine next step
        next_step = self._determine_next_step(state.current_step, user_input, state.collected_info)
        state.current_step = next_step
        
        # Check if we should record or end - trigger recording for both RECORDING and RECORDING_INTRO
        should_record = next_step in [ConversationStep.RECORDING, ConversationStep.RECORDING_INTRO]
        should_end = next_step == ConversationStep.GOODBYE
        
        return AgentResponse(
            message=message,
            next_step=next_step,
            should_record=should_record,
            should_end=should_end,
            gathered_info=extracted,
        )
    
    def _get_fallback_message(self, state: ConversationState) -> str:
        """Get fallback message if LLM fails (multilingual)"""
        # Map conversation steps to i18n keys
        key_map = {
            ConversationStep.GREETING: "va_fallback_greeting",
            ConversationStep.OCCUPATION: "va_fallback_occupation",
            ConversationStep.SYMPTOMS: "va_fallback_symptoms",
            ConversationStep.RECORDING_INTRO: "va_fallback_recording",
            ConversationStep.GOODBYE: "va_fallback_goodbye",
            ConversationStep.ASHA_HANDOFF: "va_fallback_handoff",
        }
        
        i18n_key = key_map.get(state.current_step, "va_fallback_greeting")
        return get_text(i18n_key, state.language)
    
    def get_initial_greeting(self, language: str = "en") -> str:
        """Get the initial greeting message (multilingual)"""
        return get_text("va_greeting", language)
    
    def get_results_message(
        self,
        state: ConversationState,
        risk_level: str,
        recommendation: str
    ) -> str:
        """Generate personalized results message (multilingual)"""
        collected = state.collected_info
        
        # Normalize risk strings
        if risk_level in ["high_risk", "severe", "urgent"]:
            risk_level = "high"
        elif risk_level in ["moderate_risk", "moderate"]:
            risk_level = "moderate"
        elif risk_level in ["low_risk", "low", "normal"]:
            risk_level = "normal"
            
        # Escalate risk if pesticide exposure
        if collected.get("pesticide_exposure") and risk_level == "normal":
            risk_level = "moderate"
            
        # Select message key based on risk
        if risk_level == "high":
            key = "va_result_high"
        elif risk_level == "moderate":
            key = "va_result_moderate"
        else:
            key = "va_result_normal"
            
        # Get template text
        template = get_text(key, state.language)
        
        # Format with recommendation
        return template.replace("{rec}", recommendation)


# Singleton instance
_voice_agent: Optional[VoiceAgentService] = None


def get_voice_agent_service() -> VoiceAgentService:
    """Get or create the voice agent singleton"""
    global _voice_agent
    if _voice_agent is None:
        _voice_agent = VoiceAgentService()
    return _voice_agent
