"""
RAG Knowledge Base Service
Embeds and queries health knowledge from i18n translations and FARMER_HEALTH_GUIDE.md
"""
import logging
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeChunk:
    """A chunk of knowledge with metadata"""
    content: str
    source: str  # "i18n" or "guide"
    topic: str   # e.g., "pesticide", "depression", "farming"
    language: str
    embedding: Optional[np.ndarray] = None


class RAGKnowledgeBase:
    """
    RAG-based knowledge base for health Q&A.
    Uses OpenAI embeddings for semantic search.
    """
    
    def __init__(self):
        self._client: Optional[OpenAI] = None
        self._chunks: list[KnowledgeChunk] = []
        self._embeddings: Optional[np.ndarray] = None
        self._initialized = False
        self._query_cache: dict[str, np.ndarray] = {}  # Cache for query embeddings
        self._cache_max_size = 100  # Limit cache size
    
    @property
    def client(self) -> OpenAI:
        """Lazy load OpenAI client"""
        if self._client is None:
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not configured")
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client
    
    def build_knowledge_base(self) -> None:
        """
        Build the knowledge base by loading and embedding content.
        Should be called once at startup.
        """
        if self._initialized:
            logger.info("RAG knowledge base already initialized")
            return
        
        logger.info("Building RAG knowledge base...")
        
        # Load knowledge chunks
        self._load_i18n_translations()
        self._load_farmer_health_guide()
        
        # Create embeddings
        self._create_embeddings()
        
        self._initialized = True
        logger.info(f"RAG knowledge base built with {len(self._chunks)} chunks")
    
    def _load_i18n_translations(self) -> None:
        """Load translations from i18n.py as knowledge chunks"""
        from app.utils.i18n import TRANSLATIONS
        
        # Topic mapping for better retrieval
        topic_keywords = {
            "pesticide": ["pesticide", "chemical", "poison"],
            "dust": ["dust", "grain", "hay", "lung"],
            "farming": ["farmer", "farm", "season", "sowing", "harvest"],
            "respiratory": ["cough", "breathing", "chest", "wheeze"],
            "depression": ["mood", "tired", "stress", "mental"],
            "screening": ["record", "cough", "analysis", "result"],
            "helpline": ["helpline", "doctor", "hospital", "emergency"],
        }
        
        for key, translations in TRANSLATIONS.items():
            # Determine topic from key
            topic = "general"
            for topic_name, keywords in topic_keywords.items():
                if any(kw in key.lower() for kw in keywords):
                    topic = topic_name
                    break
            
            # Add each language variant as a chunk
            for lang, text in translations.items():
                self._chunks.append(KnowledgeChunk(
                    content=f"[{key}] {text}",
                    source="i18n",
                    topic=topic,
                    language=lang
                ))
    
    def _load_farmer_health_guide(self) -> None:
        """Load FARMER_HEALTH_GUIDE.md as knowledge chunks"""
        guide_path = Path("FARMER_HEALTH_GUIDE.md")
        
        if not guide_path.exists():
            logger.warning("FARMER_HEALTH_GUIDE.md not found, skipping")
            return
        
        content = guide_path.read_text(encoding="utf-8")
        
        # Split by sections (##)
        sections = content.split("\n## ")
        
        for section in sections:
            if not section.strip():
                continue
            
            # Get section title and content
            lines = section.strip().split("\n", 1)
            title = lines[0].strip("#").strip()
            body = lines[1] if len(lines) > 1 else ""
            
            # Determine topic from title
            topic = "general"
            title_lower = title.lower()
            if "pesticide" in title_lower or "occupational" in title_lower:
                topic = "pesticide"
            elif "dust" in title_lower or "lung" in title_lower:
                topic = "dust"
            elif "farmer" in title_lower or "seasonal" in title_lower:
                topic = "farming"
            elif "screening" in title_lower or "journey" in title_lower:
                topic = "screening"
            elif "asha" in title_lower or "worker" in title_lower:
                topic = "asha"
            
            # Chunk large sections (max ~500 chars per chunk)
            chunk_size = 500
            full_content = f"{title}\n{body}"
            
            for i in range(0, len(full_content), chunk_size):
                chunk_text = full_content[i:i + chunk_size]
                if chunk_text.strip():
                    self._chunks.append(KnowledgeChunk(
                        content=chunk_text,
                        source="guide",
                        topic=topic,
                        language="en"  # Guide is in English
                    ))
    
    def _create_embeddings(self) -> None:
        """Create embeddings for all chunks using OpenAI"""
        if not self._chunks:
            logger.warning("No chunks to embed")
            return
        
        # Batch embed (max 2048 inputs per request)
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(self._chunks), batch_size):
            batch = self._chunks[i:i + batch_size]
            texts = [chunk.content for chunk in batch]
            
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts
                )
                
                for j, embedding_data in enumerate(response.data):
                    self._chunks[i + j].embedding = np.array(embedding_data.embedding)
                    all_embeddings.append(embedding_data.embedding)
                    
            except Exception as e:
                logger.error(f"Failed to create embeddings: {e}")
                # Fall back to zero embeddings for this batch
                for chunk in batch:
                    chunk.embedding = np.zeros(1536)  # text-embedding-3-small dimension
                    all_embeddings.append(chunk.embedding.tolist())
        
        self._embeddings = np.array(all_embeddings)
        logger.info(f"Created embeddings for {len(self._chunks)} chunks")
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    
    def query(
        self,
        question: str,
        top_k: int = 3,
        language: Optional[str] = None,
        topic: Optional[str] = None
    ) -> list[KnowledgeChunk]:
        """
        Query the knowledge base for relevant chunks.
        Uses vectorized operations for speed.
        
        Args:
            question: User's question or input
            top_k: Number of top results to return
            language: Optional language filter
            topic: Optional topic filter
            
        Returns:
            List of most relevant knowledge chunks
        """
        if not self._initialized or self._embeddings is None:
            logger.warning("Knowledge base not initialized")
            return []
        
        # Check cache first
        cache_key = question[:100]  # Truncate for cache key
        if cache_key in self._query_cache:
            query_embedding = self._query_cache[cache_key]
        else:
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[question]
                )
                query_embedding = np.array(response.data[0].embedding)
                
                # Cache with size limit
                if len(self._query_cache) >= self._cache_max_size:
                    # Remove oldest entry
                    self._query_cache.pop(next(iter(self._query_cache)))
                self._query_cache[cache_key] = query_embedding
                
            except Exception as e:
                logger.error(f"Failed to embed query: {e}")
                return []
        
        # Build filter mask
        mask = np.ones(len(self._chunks), dtype=bool)
        for i, chunk in enumerate(self._chunks):
            if language and chunk.language != language and chunk.language != "en":
                mask[i] = False
            if topic and chunk.topic != topic and chunk.topic != "general":
                mask[i] = False
        
        # Vectorized similarity computation (much faster than loop)
        if mask.sum() == 0:
            return []
        
        filtered_embeddings = self._embeddings[mask]
        
        # Compute all similarities at once using matrix operations
        query_norm = np.linalg.norm(query_embedding)
        embedding_norms = np.linalg.norm(filtered_embeddings, axis=1)
        similarities = np.dot(filtered_embeddings, query_embedding) / (embedding_norms * query_norm + 1e-8)
        
        # Get top-k indices
        filtered_indices = np.where(mask)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [self._chunks[filtered_indices[i]] for i in top_indices]
    
    def get_context_for_topic(self, topic: str, language: str = "en", max_chunks: int = 5) -> str:
        """
        Get all relevant context for a specific topic.
        
        Args:
            topic: Topic to get context for
            language: Preferred language
            max_chunks: Maximum chunks to include
            
        Returns:
            Concatenated context string
        """
        relevant = [c for c in self._chunks if c.topic == topic]
        
        # Prefer matching language
        lang_chunks = [c for c in relevant if c.language == language]
        en_chunks = [c for c in relevant if c.language == "en"]
        
        # Combine: prefer language-specific, fall back to English
        result_chunks = lang_chunks[:max_chunks]
        if len(result_chunks) < max_chunks:
            remaining = max_chunks - len(result_chunks)
            result_chunks.extend([c for c in en_chunks if c not in result_chunks][:remaining])
        
        return "\n\n".join(c.content for c in result_chunks)


# Singleton instance
_rag_service: Optional[RAGKnowledgeBase] = None


def get_rag_service() -> RAGKnowledgeBase:
    """Get or create the RAG knowledge base singleton"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGKnowledgeBase()
    return _rag_service


async def initialize_rag_service() -> None:
    """Initialize the RAG service (call at app startup)"""
    service = get_rag_service()
    service.build_knowledge_base()
