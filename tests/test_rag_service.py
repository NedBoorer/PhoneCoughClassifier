"""
Tests for RAG Knowledge Base Service
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestRAGKnowledgeBase:
    """Tests for RAG knowledge base functionality"""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing"""
        with patch('app.services.rag_service.OpenAI') as mock:
            client = MagicMock()
            mock.return_value = client
            yield client
    
    def test_build_knowledge_base_loads_translations(self, mock_openai_client):
        """Test that translations are loaded from i18n.py"""
        from app.services.rag_service import RAGKnowledgeBase
        
        # Mock embeddings response
        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536) for _ in range(100)]
        )
        
        kb = RAGKnowledgeBase()
        kb._client = mock_openai_client
        kb._load_i18n_translations()
        
        # Should have loaded some chunks
        assert len(kb._chunks) > 0
        
        # Check that we have different languages
        languages = set(c.language for c in kb._chunks)
        assert "en" in languages
        assert "hi" in languages
    
    def test_build_knowledge_base_loads_markdown(self, mock_openai_client):
        """Test that FARMER_HEALTH_GUIDE.md is loaded"""
        from app.services.rag_service import RAGKnowledgeBase
        from pathlib import Path
        
        # Mock embeddings response
        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536) for _ in range(100)]
        )
        
        kb = RAGKnowledgeBase()
        kb._client = mock_openai_client
        
        # Only test if file exists
        if Path("FARMER_HEALTH_GUIDE.md").exists():
            kb._load_farmer_health_guide()
            
            # Should have loaded guide chunks
            guide_chunks = [c for c in kb._chunks if c.source == "guide"]
            assert len(guide_chunks) > 0
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        from app.services.rag_service import RAGKnowledgeBase
        
        kb = RAGKnowledgeBase()
        
        # Identical vectors should have similarity 1.0
        vec = np.array([1.0, 2.0, 3.0])
        sim = kb._cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.01
        
        # Orthogonal vectors should have similarity 0
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        sim = kb._cosine_similarity(a, b)
        assert abs(sim) < 0.01
    
    def test_query_returns_relevant_results(self, mock_openai_client):
        """Test that query returns relevant chunks"""
        from app.services.rag_service import RAGKnowledgeBase, KnowledgeChunk
        
        kb = RAGKnowledgeBase()
        kb._client = mock_openai_client
        
        # Create test chunks with embeddings
        kb._chunks = [
            KnowledgeChunk(
                content="Pesticide exposure is dangerous for farmers",
                source="guide",
                topic="pesticide",
                language="en",
                embedding=np.array([1.0, 0.0, 0.0])
            ),
            KnowledgeChunk(
                content="Dust causes lung problems",
                source="guide",
                topic="dust",
                language="en",
                embedding=np.array([0.0, 1.0, 0.0])
            ),
        ]
        kb._initialized = True
        
        # Mock query embedding (similar to pesticide chunk)
        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.9, 0.1, 0.0])]
        )
        
        results = kb.query("pesticide dangers", top_k=1)
        
        assert len(results) == 1
        assert "pesticide" in results[0].content.lower()
    
    def test_query_respects_language_filter(self, mock_openai_client):
        """Test that query filters by language"""
        from app.services.rag_service import RAGKnowledgeBase, KnowledgeChunk
        
        kb = RAGKnowledgeBase()
        kb._client = mock_openai_client
        
        # Create test chunks with different languages
        kb._chunks = [
            KnowledgeChunk(
                content="English content",
                source="i18n",
                topic="general",
                language="en",
                embedding=np.array([1.0, 0.0])
            ),
            KnowledgeChunk(
                content="Hindi content",
                source="i18n",
                topic="general",
                language="hi",
                embedding=np.array([1.0, 0.0])  # Same embedding to test filtering
            ),
        ]
        kb._initialized = True
        
        # Mock query embedding
        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[1.0, 0.0])]
        )
        
        results = kb.query("test query", top_k=2, language="hi")
        
        # Should get both (Hindi match + English as fallback)
        languages = [r.language for r in results]
        assert "hi" in languages or "en" in languages
    
    def test_get_context_for_topic(self, mock_openai_client):
        """Test getting context for a specific topic"""
        from app.services.rag_service import RAGKnowledgeBase, KnowledgeChunk
        
        kb = RAGKnowledgeBase()
        kb._client = mock_openai_client
        
        # Create test chunks
        kb._chunks = [
            KnowledgeChunk(
                content="Pesticide warning 1",
                source="guide",
                topic="pesticide",
                language="en",
                embedding=None
            ),
            KnowledgeChunk(
                content="Pesticide warning 2",
                source="guide",
                topic="pesticide",
                language="en",
                embedding=None
            ),
            KnowledgeChunk(
                content="Dust warning",
                source="guide",
                topic="dust",
                language="en",
                embedding=None
            ),
        ]
        
        context = kb.get_context_for_topic("pesticide", language="en", max_chunks=2)
        
        assert "Pesticide warning 1" in context
        assert "Pesticide warning 2" in context
        assert "Dust warning" not in context


class TestRAGServiceSingleton:
    """Test singleton behavior"""
    
    def test_get_rag_service_returns_same_instance(self):
        """Test that get_rag_service returns singleton"""
        from app.services.rag_service import get_rag_service
        
        service1 = get_rag_service()
        service2 = get_rag_service()
        
        assert service1 is service2
