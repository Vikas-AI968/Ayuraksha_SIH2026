"""
Semantic Retriever Wrapper for IP-SAKTI Sahayak.
"""
from typing import List, Optional
from vector_store.qdrant_store import QdrantVectorStore
from models.chunk import RetrievalResult


class SemanticRetriever:
    """Wrapper around QdrantVectorStore for semantic search."""

    def __init__(self, vector_store: QdrantVectorStore):
        self.vector_store = vector_store

    def search(self, query: str, top_k: int = 10, **filters) -> List[RetrievalResult]:
        return self.vector_store.search_vector(
            query=query,
            top_k=top_k,
            jurisdiction=filters.get("jurisdiction"),
            domain=filters.get("domain"),
            authority=filters.get("authority"),
            document_type=filters.get("document_type"),
            effective_date_from=filters.get("effective_date_from")
            ,status=filters.get("status"),
            source_priority_min=filters.get("source_priority_min"),
            authoritative=filters.get("authoritative")
        )
