"""
Tests for Voice Agent Service
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestConversationState:
    """Tests for conversation state management"""
    
    def test_state_to_dict(self):
        """Test serialization of conversation state"""
        from app.services.voice_agent_service import ConversationState, ConversationStep
        
        state = ConversationState(
            call_sid="CA123",
            language="hi",
            current_step=ConversationStep.OCCUPATION,
            collected_info={"is_farmer": True},
        )
        
        data = state.to_dict()
        
        assert data["call_sid"] == "CA123"
        assert data["language"] == "hi"
        assert data["current_step"] == "occupation"
        assert data["collected_info"]["is_farmer"] == True
    
    def test_state_from_dict(self):
        """Test deserialization of conversation state"""
        from app.services.voice_agent_service import ConversationState, ConversationStep
        
        data = {
            "call_sid": "CA456",
            "language": "en",
            "current_step": "symptoms",
            "collected_info": {"has_cough": True},
            "message_history": [],
            "turn_count": 3,
        }
        
        state = ConversationState.from_dict(data)
        
        assert state.call_sid == "CA456"
        assert state.language == "en"
        assert state.current_step == ConversationStep.SYMPTOMS
        assert state.collected_info["has_cough"] == True
        assert state.turn_count == 3


class TestVoiceAgentService:
    """Tests for voice agent service"""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client"""
        with patch('app.services.voice_agent_service.OpenAI') as mock:
            client = MagicMock()
            mock.return_value = client
            yield client
    
    @pytest.fixture
    def mock_rag_service(self):
        """Mock RAG service"""
        with patch('app.services.voice_agent_service.get_rag_service') as mock:
            rag = MagicMock()
            rag.query.return_value = []
            rag.get_context_for_topic.return_value = ""
            mock.return_value = rag
            yield rag
    
    def test_get_or_create_state(self):
        """Test state creation and retrieval"""
        from app.services.voice_agent_service import VoiceAgentService, ConversationStep
        
        agent = VoiceAgentService()
        
        # First call should create state
        state1 = agent.get_or_create_state("CA123")
        assert state1.call_sid == "CA123"
        assert state1.current_step == ConversationStep.GREETING
        
        # Second call should return same state
        state2 = agent.get_or_create_state("CA123")
        assert state1 is state2
    
    def test_clear_state(self):
        """Test state cleanup"""
        from app.services.voice_agent_service import VoiceAgentService
        
        agent = VoiceAgentService()
        
        # Create state
        agent.get_or_create_state("CA123")
        assert agent.get_state("CA123") is not None
        
        # Clear state
        agent.clear_state("CA123")
        assert agent.get_state("CA123") is None
    
    def test_extract_info_farmer(self):
        """Test farmer detection from input"""
        from app.services.voice_agent_service import VoiceAgentService, ConversationStep
        
        agent = VoiceAgentService()
        
        # English farmer
        info = agent._extract_info("Yes I am a farmer", ConversationStep.OCCUPATION)
        assert info.get("is_farmer") == True
        
        # Hindi farmer
        info = agent._extract_info("हां मैं किसान हूं", ConversationStep.OCCUPATION)
        assert info.get("is_farmer") == True
        
        # Not a farmer
        info = agent._extract_info("No I work in office", ConversationStep.OCCUPATION)
        assert info.get("is_farmer") == False
    
    def test_extract_info_pesticide(self):
        """Test pesticide exposure detection"""
        from app.services.voice_agent_service import VoiceAgentService, ConversationStep
        
        agent = VoiceAgentService()
        
        info = agent._extract_info("Yes I spray pesticides", ConversationStep.PESTICIDE_CHECK)
        assert info.get("pesticide_exposure") == True
        
        info = agent._extract_info("नहीं", ConversationStep.PESTICIDE_CHECK)
        assert info.get("pesticide_exposure") == False
    
    def test_extract_info_symptoms(self):
        """Test symptom detection from input"""
        from app.services.voice_agent_service import VoiceAgentService, ConversationStep
        
        agent = VoiceAgentService()
        
        info = agent._extract_info("I have chest pain and cough", ConversationStep.SYMPTOMS)
        assert info.get("chest_pain") == True
        assert info.get("has_cough") == True
    
    def test_determine_next_step_normal_flow(self):
        """Test normal conversation flow transitions"""
        from app.services.voice_agent_service import VoiceAgentService, ConversationStep
        
        agent = VoiceAgentService()
        
        # Greeting -> Symptoms (streamlined flow)
        next_step = agent._determine_next_step(
            ConversationStep.GREETING, "", {}
        )
        assert next_step == ConversationStep.SYMPTOMS
        
        # Symptoms -> Recording Intro
        next_step = agent._determine_next_step(
            ConversationStep.SYMPTOMS, "", {"has_cough": True}
        )
        assert next_step == ConversationStep.RECORDING_INTRO
    
    def test_determine_next_step_cough_fasttrack(self):
        """Test that mentioning cough fast-tracks to recording"""
        from app.services.voice_agent_service import VoiceAgentService, ConversationStep
        
        agent = VoiceAgentService()
        
        # User mentions cough from greeting - should go to recording
        next_step = agent._determine_next_step(
            ConversationStep.GREETING, "I have a cough", {}
        )
        assert next_step == ConversationStep.RECORDING
        
        # User mentions khansi (Hindi) from symptoms - should go to recording
        next_step = agent._determine_next_step(
            ConversationStep.SYMPTOMS, "mujhe khansi hai", {}
        )
        assert next_step == ConversationStep.RECORDING
    
    def test_determine_next_step_skip_farmer_questions(self):
        """Test skipping farmer questions for non-farmers"""
        from app.services.voice_agent_service import VoiceAgentService, ConversationStep
        
        agent = VoiceAgentService()
        
        # Non-farmer should skip pesticide check
        next_step = agent._determine_next_step(
            ConversationStep.SYMPTOMS, "", {"is_farmer": False}
        )
        assert next_step == ConversationStep.RECORDING_INTRO
    
    def test_get_initial_greeting_english(self):
        """Test English greeting"""
        from app.services.voice_agent_service import VoiceAgentService
        
        agent = VoiceAgentService()
        greeting = agent.get_initial_greeting("en")
        
        assert "Swasth Saathi" in greeting
        assert "health" in greeting.lower()
    
    def test_get_initial_greeting_hindi(self):
        """Test Hindi greeting"""
        from app.services.voice_agent_service import VoiceAgentService
        
        agent = VoiceAgentService()
        greeting = agent.get_initial_greeting("hi")
        
        assert "स्वास्थ" in greeting
        assert "नमस्ते" in greeting
    
    def test_get_fallback_message(self):
        """Test fallback messages when LLM fails"""
        from app.services.voice_agent_service import VoiceAgentService, ConversationState, ConversationStep
        
        agent = VoiceAgentService()
        
        state = ConversationState(
            call_sid="CA123",
            language="en",
            current_step=ConversationStep.OCCUPATION
        )
        
        message = agent._get_fallback_message(state)
        assert "farmer" in message.lower()
    
    def test_get_results_message_high_risk(self):
        """Test high risk results message"""
        from app.services.voice_agent_service import VoiceAgentService, ConversationState, ConversationStep
        
        agent = VoiceAgentService()
        
        state = ConversationState(
            call_sid="CA123",
            language="en",
            current_step=ConversationStep.RESULTS,
            collected_info={"is_farmer": True, "pesticide_exposure": True}
        )
        
        message = agent.get_results_message(state, "high", "See a doctor")
        
        assert "concerned" in message.lower() or "doctor" in message.lower()
    
    @pytest.mark.asyncio
    async def test_process_user_input(self, mock_openai_client, mock_rag_service):
        """Test processing user input"""
        from app.services.voice_agent_service import VoiceAgentService, ConversationStep
        
        # Mock LLM response
        mock_openai_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Thank you for sharing!"))]
        )
        
        agent = VoiceAgentService()
        agent._client = mock_openai_client
        
        response = await agent.process_user_input(
            call_sid="CA123",
            user_input="I am a farmer",
            language="en"
        )
        
        assert response.message == "Thank you for sharing!"
        assert response.gathered_info.get("is_farmer") == True


class TestVoiceAgentSingleton:
    """Test singleton behavior"""
    
    def test_get_voice_agent_service_returns_same_instance(self):
        """Test that get_voice_agent_service returns singleton"""
        from app.services.voice_agent_service import get_voice_agent_service
        
        service1 = get_voice_agent_service()
        service2 = get_voice_agent_service()
        
        assert service1 is service2
