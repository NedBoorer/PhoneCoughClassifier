"""
Tests for Voice Agent Trust Features

Verifies that:
- ASHA handoff is triggered on human request keywords
- System prompt includes authority and social proof
"""
import pytest

from app.services.voice_agent_service import (
    VoiceAgentService,
    ConversationStep,
    ConversationState,
)
from app.config import settings


class TestTrustFeatures:
    """Test trust-building features in the voice agent"""

    def test_asha_handoff_step_exists(self):
        """Verify ASHA_HANDOFF is a valid conversation step"""
        assert hasattr(ConversationStep, "ASHA_HANDOFF")
        assert ConversationStep.ASHA_HANDOFF.value == "asha_handoff"

    def test_handoff_triggered_on_human_request(self):
        """Verify handoff is triggered when user asks for a human"""
        agent = VoiceAgentService()

        # Test various human request keywords
        human_keywords = [
            "I want to talk to a human",
            "Can I speak to a person?",
            "This is fake, let me talk to a doctor",
            "robot se nahi",
            "ye asli nahi hai",
        ]

        for user_input in human_keywords:
            next_step = agent._determine_next_step(
                current_step=ConversationStep.SYMPTOMS,
                user_input=user_input,
                extracted_info={}
            )
            assert next_step == ConversationStep.ASHA_HANDOFF, f"Failed for: {user_input}"

    def test_handoff_not_triggered_on_normal_input(self):
        """Verify normal conversation does not trigger handoff"""
        agent = VoiceAgentService()

        normal_inputs = [
            "I am a farmer",
            "Yes, I have a cough",
            "Haan, main kisan hoon",
        ]

        for user_input in normal_inputs:
            next_step = agent._determine_next_step(
                current_step=ConversationStep.OCCUPATION,
                user_input=user_input,
                extracted_info={}
            )
            assert next_step != ConversationStep.ASHA_HANDOFF, f"Unexpected handoff for: {user_input}"

    def test_system_prompt_contains_authority(self):
        """Verify system prompt includes trusted authority name"""
        agent = VoiceAgentService()
        state = ConversationState(call_sid="test123", language="en")

        # Call the method directly without mocking the property
        prompt = agent._get_system_prompt(state)

        assert settings.trusted_authority_name in prompt
        assert "Swasth Saathi" in prompt

    def test_system_prompt_contains_social_proof(self):
        """Verify system prompt includes social proof stats"""
        agent = VoiceAgentService()
        state = ConversationState(call_sid="test456", language="hi")

        prompt = agent._get_system_prompt(state)

        assert str(settings.mock_daily_users) in prompt

    def test_fallback_for_asha_handoff(self):
        """Verify fallback message exists for ASHA_HANDOFF step"""
        agent = VoiceAgentService()
        state = ConversationState(
            call_sid="test789",
            language="en",
            current_step=ConversationStep.ASHA_HANDOFF
        )

        fallback = agent._get_fallback_message(state)
        assert "health worker" in fallback.lower()

    def test_fallback_hindi_for_asha_handoff(self):
        """Verify Hindi fallback message exists for ASHA_HANDOFF step"""
        agent = VoiceAgentService()
        state = ConversationState(
            call_sid="test789",
            language="hi",
            current_step=ConversationStep.ASHA_HANDOFF
        )

        fallback = agent._get_fallback_message(state)
        assert "स्वास्थ्य" in fallback  # Health in Hindi
