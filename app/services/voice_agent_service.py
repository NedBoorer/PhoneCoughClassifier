"""
Voice Agent Service
Conversational AI agent for personalized health screening interactions
"""
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from openai import OpenAI

from app.config import settings
from app.services.rag_service import get_rag_service

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
    
    @property
    def client(self) -> OpenAI:
        """Lazy load OpenAI client"""
        if self._client is None:
            if not settings.openai_api_key:
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
        
        return f"""You are Swasth Saathi, a warm and trusted health companion.
        
IDENTITY:
- Name: Swasth Saathi (Health Friend)
- Affiliation: {settings.trusted_authority_name}
- Role: Caring digital assistant for local doctors.

SOCIAL PROOF (Mention casually if appropriate):
- "I have already helped {settings.mock_daily_users} people in your area today."

LANGUAGE: {state.language} - Always respond in this language.
STEP: {state.current_step.value}
INFO: {state.collected_info}

RULES:
- 2-3 sentences MAX (it's a phone call)
- Warm, respectful tone (use "Ji", "Beta" as appropriate)
- Be influential but patient.
- Never mention being AI/Robot directly; say you are a "digital service".
- Emergency: suggest calling 108

KNOWLEDGE:
{context[:500]}

Respond with ONLY the spoken message."""

    def _determine_next_step(
        self,
        current_step: ConversationStep,
        user_input: str,
        extracted_info: dict
    ) -> ConversationStep:
        """Determine the next conversation step based on current state"""
        
        # Check for Handoff Intents FIRST (Safety Net - always active)
        human_keywords = ["human", "person", "doctor", "talk to", "real person", "fake", "robot", "insaan", "baat karni", "asli", "asha"]
        if any(kw in user_input.lower() for kw in human_keywords):
            return ConversationStep.ASHA_HANDOFF
        
        transitions = {
            ConversationStep.GREETING: ConversationStep.OCCUPATION,
            ConversationStep.LANGUAGE: ConversationStep.OCCUPATION,
            ConversationStep.OCCUPATION: ConversationStep.SYMPTOMS,
            ConversationStep.SYMPTOMS: ConversationStep.PESTICIDE_CHECK,
            ConversationStep.PESTICIDE_CHECK: ConversationStep.DUST_CHECK,
            ConversationStep.DUST_CHECK: ConversationStep.RECORDING_INTRO,
            ConversationStep.RECORDING_INTRO: ConversationStep.RECORDING,
            ConversationStep.RECORDING: ConversationStep.PROCESSING,
            ConversationStep.PROCESSING: ConversationStep.RESULTS,
            ConversationStep.RESULTS: ConversationStep.FAMILY_OFFER,
            ConversationStep.FAMILY_OFFER: ConversationStep.GOODBYE,
            ConversationStep.GOODBYE: ConversationStep.GOODBYE,
        }
        
        # Skip pesticide/dust check if not a farmer
        if current_step == ConversationStep.SYMPTOMS:
            if not extracted_info.get("is_farmer", False):
                return ConversationStep.RECORDING_INTRO
        
        if current_step == ConversationStep.PESTICIDE_CHECK:
            if not extracted_info.get("is_farmer", False):
                return ConversationStep.RECORDING_INTRO

        return transitions.get(current_step, ConversationStep.GOODBYE)
    
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
                response = self.client.chat.completions.create(
                    model=settings.voice_agent_model if hasattr(settings, 'voice_agent_model') else "gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *state.message_history[-4:],  # Reduced from 6 to 4 for lower cost
                    ],
                    max_tokens=100,  # Reduced from 150 - voice responses should be short
                    temperature=0.7,
                )
                
                message = response.choices[0].message.content.strip()
                
                # Cache response with size limit
                if len(self._response_cache) >= self._cache_max_size:
                    self._response_cache.pop(next(iter(self._response_cache)))
                self._response_cache[cache_key] = message
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # Fallback message
            message = self._get_fallback_message(state)
        
        # Add assistant message to history
        state.message_history.append({"role": "assistant", "content": message})
        
        # Determine next step
        next_step = self._determine_next_step(state.current_step, user_input, state.collected_info)
        state.current_step = next_step
        
        # Check if we should record or end
        should_record = next_step == ConversationStep.RECORDING
        should_end = next_step == ConversationStep.GOODBYE
        
        return AgentResponse(
            message=message,
            next_step=next_step,
            should_record=should_record,
            should_end=should_end,
            gathered_info=extracted,
        )
    
    def _get_fallback_message(self, state: ConversationState) -> str:
        """Get fallback message if LLM fails"""
        fallbacks = {
            ConversationStep.GREETING: {
                "en": "Hello! I'm your health helper. How are you feeling today?",
                "hi": "नमस्ते! मैं आपका स्वास्थ्य सहायक हूं। आज आप कैसा महसूस कर रहे हैं?",
            },
            ConversationStep.OCCUPATION: {
                "en": "Are you a farmer or farm worker?",
                "hi": "क्या आप किसान या खेत मजदूर हैं?",
            },
            ConversationStep.SYMPTOMS: {
                "en": "Do you have any cough, chest pain, or breathing problems?",
                "hi": "क्या आपको खांसी, सीने में दर्द या सांस लेने में कोई समस्या है?",
            },
            ConversationStep.RECORDING_INTRO: {
                "en": "I'll need to hear your cough. Please cough after the beep.",
                "hi": "मुझे आपकी खांसी सुननी होगी। बीप के बाद कृपया खांसें।",
            },
            ConversationStep.GOODBYE: {
                "en": "Thank you for calling. Take care and stay healthy!",
                "hi": "कॉल करने के लिए धन्यवाद। अपना ख्याल रखें और स्वस्थ रहें!",
            },
            ConversationStep.ASHA_HANDOFF: {
                "en": "I understand. Let me connect you to a health worker.",
                "hi": "मैं समझता हूँ। मैं आपको एक स्वास्थ्य कार्यकर्ता से जोड़ता हूँ।",
            },
        }
        
        step_fallbacks = fallbacks.get(state.current_step, fallbacks[ConversationStep.GREETING])
        return step_fallbacks.get(state.language, step_fallbacks.get("en", "Please continue."))
    
    def get_initial_greeting(self, language: str = "en") -> str:
        """Get the initial greeting message"""
        greetings = {
            "en": (
                "Hello and welcome! I'm Swasth Saathi, your health friend. "
                "I'm here to help check on your health today. "
                "Tell me, how are you feeling?"
            ),
            "hi": (
                "नमस्ते और स्वागत है! मैं स्वास्थ साथी हूं, आपका स्वास्थ्य मित्र। "
                "मैं आज आपकी सेहत की जांच में मदद करने आया हूं। "
                "बताइए, आप कैसा महसूस कर रहे हैं?"
            ),
        }
        return greetings.get(language, greetings["en"])
    
    def get_results_message(
        self,
        state: ConversationState,
        risk_level: str,
        recommendation: str
    ) -> str:
        """Generate personalized results message"""
        collected = state.collected_info
        
        # Escalate risk if pesticide exposure
        if collected.get("pesticide_exposure") and risk_level in ["low", "normal"]:
            risk_level = "moderate"
        
        if state.language == "hi":
            if risk_level in ["high", "severe", "urgent"]:
                return (
                    f"बेटा, मुझे आपकी सेहत की चिंता है। {recommendation} "
                    "कृपया आज ही डॉक्टर से मिलें। मैं आपको रिपोर्ट भेज रहा हूं।"
                )
            elif risk_level in ["moderate"]:
                return (
                    f"कुछ ध्यान देने की जरूरत है। {recommendation} "
                    "अगर तकलीफ बढ़े तो डॉक्टर को जरूर दिखाएं।"
                )
            else:
                return (
                    f"अच्छी बात है, आपकी सेहत ठीक लग रही है। {recommendation} "
                    "अपना ख्याल रखते रहें।"
                )
        else:
            if risk_level in ["high", "severe", "urgent"]:
                return (
                    f"I'm concerned about what I'm hearing. {recommendation} "
                    "Please see a doctor today. I'm sending you a detailed report."
                )
            elif risk_level in ["moderate"]:
                return (
                    f"There are some things to pay attention to. {recommendation} "
                    "Please see a doctor if symptoms worsen."
                )
            else:
                return (
                    f"That's good news, you seem healthy. {recommendation} "
                    "Keep taking care of yourself."
                )


# Singleton instance
_voice_agent: Optional[VoiceAgentService] = None


def get_voice_agent_service() -> VoiceAgentService:
    """Get or create the voice agent singleton"""
    global _voice_agent
    if _voice_agent is None:
        _voice_agent = VoiceAgentService()
    return _voice_agent
