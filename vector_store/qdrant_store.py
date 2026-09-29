"""
Qdrant Vector Database Integration for IP-SAKTI Sahayak.
Supports vector insertion, payload indexing, and structured metadata filtering.
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)

from models.chunk import Chunk, RetrievalResult
from models.document import DocumentMetadata
from embeddings.provider import BaseEmbeddingProvider, get_default_embedding_provider
from ingestion.paths import resolve_repo_path
from determinism import stable_point_id

logger = logging.getLogger("qdrant_store")

# Single source of truth for where the Qdrant collection lives. Both the
# ingestion pipeline (scripts/ingest_authoritative_corpus.py) and the FastAPI
# app (api/main.py) MUST read these same defaults via `get_qdrant_settings()`
# below -- previously they hardcoded two different collection names
# ("ip_sakti_chunks" vs "ip_sakti_authoritative") and both defaulted to
# ":memory:", so ingestion and the API silently populated two separate,
# non-persistent indexes that never saw each other's data.
DEFAULT_QDRANT_COLLECTION = "ip_sakti_authoritative_multilingual"
DEFAULT_QDRANT_LOCATION = "./data/qdrant"


def get_qdrant_settings() -> "tuple[str, str]":
    """Returns (location, collection_name) from the environment, falling back
    to the canonical persistent-local defaults. Call this from every entry
    point (API startup, ingestion scripts, reindexing utilities) instead of
    reading ``QDRANT_LOCATION``/``QDRANT_COLLECTION`` directly, so they can
    never drift apart again."""
    location = os.environ.get("QDRANT_LOCATION", DEFAULT_QDRANT_LOCATION)
    collection = os.environ.get("QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION)
    return location, collection


class QdrantVectorStore:
    """Manages collection, payload indexing, and vector similarity search in Qdrant."""

    def __init__(
        self,
        collection_name: str = DEFAULT_QDRANT_COLLECTION,
        location: str = DEFAULT_QDRANT_LOCATION,
        embedding_provider: Optional[BaseEmbeddingProvider] = None
    ):
        self.collection_name = collection_name
        self.location = location
        self.embedding_provider = embedding_provider or get_default_embedding_provider()
        self.client = self._build_client(location)
        self._ensure_collection()

    @staticmethod
    def _build_client(location: str) -> QdrantClient:
        """Builds the underlying qdrant_client for the given location.

        ``QdrantClient(location=...)`` only understands two things: the
        literal string ``":memory:"`` and remote server URLs
        (``http://``/``https://``). A bare filesystem path passed as
        ``location=`` is NOT persisted to disk -- it is not a supported value
        and would either error out or be misinterpreted as a remote host.
        Local, on-disk, persistent storage requires the dedicated ``path=``
        constructor argument instead. This method picks the right one so
        callers can keep passing a single ``location`` string (":memory:",
        a URL, or a local directory) without needing to know the distinction.
        """
        if location == ":memory:":
            return QdrantClient(location=":memory:")
        if location.startswith("http://") or location.startswith("https://"):
            return QdrantClient(url=location)
        # Treat anything else as a local persistent storage directory,
        # anchored to the repository root (not the process CWD) so it works
        # the same whether launched from the repo root or imported by a
        # test runner from elsewhere.
        resolved = Path(location) if os.path.isabs(location) else resolve_repo_path(location)
        resolved.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(resolved))

    def _ensure_collection(self) -> None:
        """Create Qdrant collection if it does not exist."""
        existing_collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing_collections:
            dim = self.embedding_provider.dimension
            logger.info(f"Creating Qdrant collection '{self.collection_name}' (vector dim={dim})...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def upsert_chunks(self, chunks: List[Chunk]) -> int:
        """Embeds chunks if missing vector, and upserts points into Qdrant."""
        if not chunks:
            return 0

        # Generate missing embeddings in batch
        unembedded_texts = [c.text for c in chunks if c.vector is None]
        if unembedded_texts:
            embeddings = self.embedding_provider.embed_batch(unembedded_texts)
            emb_idx = 0
            for c in chunks:
                if c.vector is None:
                    c.vector = embeddings[emb_idx]
                    emb_idx += 1

        points = []
        for idx, chunk in enumerate(chunks):
            # Deterministic UUID derived from chunk_id (stable across
            # processes/restarts), so repeated ingestion upserts in place.
            point_id = stable_point_id(chunk.chunk_id)
            payload = chunk.to_qdrant_payload()
            points.append(PointStruct(
                id=point_id,
                vector=chunk.vector,
                payload=payload
            ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Upserted {len(points)} points into Qdrant collection '{self.collection_name}'.")
        return len(points)

    def search_vector(
        self,
        query: str,
        top_k: int = 10,
        jurisdiction: Optional[str] = None,
        domain: Optional[str] = None,
        authority: Optional[str] = None,
        document_type: Optional[str] = None,
        effective_date_from: Optional[str] = None,
        status: Optional[str] = None,
        source_priority_min: Optional[float] = None,
        authoritative: Optional[bool] = None
    ) -> List[RetrievalResult]:
        """Perform semantic vector similarity search with structured metadata filtering."""
        query_vector = self.embedding_provider.embed_text(query)

        # Build Qdrant metadata filter
        conditions = []
        if jurisdiction:
            conditions.append(FieldCondition(key="jurisdiction", match=MatchValue(value=jurisdiction)))
        if domain:
            conditions.append(FieldCondition(key="domain", match=MatchValue(value=domain)))
        if authority:
            conditions.append(FieldCondition(key="authority", match=MatchValue(value=authority)))
        if document_type:
            conditions.append(FieldCondition(key="document_type", match=MatchValue(value=document_type)))
        if effective_date_from:
            conditions.append(FieldCondition(key="effective_date", range=Range(gte=effective_date_from)))
        if status:
            conditions.append(FieldCondition(key="status", match=MatchValue(value=status)))
        if source_priority_min is not None:
            conditions.append(FieldCondition(key="source_priority", range=Range(gte=source_priority_min)))
        if authoritative is not None:
            conditions.append(FieldCondition(key="authoritative", match=MatchValue(value=authoritative)))

        query_filter = Filter(must=conditions) if conditions else None

        # Execute search in Qdrant using query_points
        query_response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k
        )

        search_hits = query_response.points if hasattr(query_response, "points") else query_response

        results = []
        for hit in search_hits:
            payload = hit.payload
            meta = DocumentMetadata(
                title=payload.get("title", ""),
                source_name=payload.get("source_name") or None,
                issuing_authority=payload.get("issuing_authority") or None,
                authority=payload.get("authority", ""),
                jurisdiction=payload.get("jurisdiction", ""),
                domain=payload.get("domain", ""),
                document_type=payload.get("document_type", ""),
                effective_date=payload.get("effective_date", ""),
                source_url=payload.get("source_url", ""),
                document_url=payload.get("document_url") or None,
                source_type=payload.get("source_type", "synthetic"),
                source_id=payload.get("source_id") or None,
                authority_level=payload.get("authority_level", "development_only"),
                access_status=payload.get("access_status", "public"),
                authoritative=bool(payload.get("authoritative", False)),
                status=payload.get("status", "current"),
                source_priority=float(payload.get("source_priority", 0.35)),
                language=payload.get("language", "en"),
                original_language=payload.get("original_language") or None,
                publication_date=payload.get("publication_date"),
                version=payload.get("version", "1.0"),
                section=payload.get("section"),
                chapter=payload.get("chapter"),
                subsection=payload.get("subsection") or None,
                rule=payload.get("rule") or None,
                sub_rule=payload.get("sub_rule") or None,
                regulation=payload.get("regulation") or None,
                sub_regulation=payload.get("sub_regulation") or None,
                article=payload.get("article") or None,
                schedule=payload.get("schedule") or None,
                paragraph=str(payload.get("paragraph")) if payload.get("paragraph") is not None else None,
                document_hash=payload.get("document_hash"),
                content_hash=payload.get("content_hash") or payload.get("document_hash"),
                retrieved_at=payload.get("retrieved_at") or None,
                parent_document_id=payload.get("document_id")
            )

            res = RetrievalResult(
                chunk_id=payload.get("chunk_id", ""),
                document_id=payload.get("document_id", ""),
                score=float(hit.score),
                semantic_score=float(hit.score),
                retrieval_method="semantic",
                text=payload.get("text", ""),
                metadata=meta
            )
            res.build_citation()
            results.append(res)

        return results

    def clear(self) -> None:
        """Clear collection contents."""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()

    def count_points(self) -> int:
        """Cheap existence check (no data transfer) used at API startup to
        decide whether an index already exists -- never re-embeds or re-upserts."""
        try:
            return self.client.count(collection_name=self.collection_name, exact=True).count
        except Exception:
            return 0

    def count_for_document(self, document_id: str) -> int:
        """Cheap, filtered existence check: how many points this collection
        actually holds for a given document_id.

        This exists specifically so ingestion's "unchanged, skip re-embedding"
        dedup decision (ingestion/authoritative.py) never has to trust the
        manifest alone. A document can be recorded as ``fetch_status ==
        "ingested"`` (manifest + cached normalized text) while the Qdrant
        collection itself is empty -- e.g. after a fresh checkout, a restored
        manifest without the matching Qdrant data directory, or a collection
        that was deleted/recreated out from under an otherwise-untouched
        manifest. Without this check, "unchanged" ingestion re-runs would
        silently skip re-embedding forever, leaving semantic search
        permanently empty for that document even though BM25 (rebuilt fresh
        from source on every ingestion run, independent of Qdrant state)
        looks fully populated.
        """
        try:
            return self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]),
                exact=True,
            ).count
        except Exception:
            return 0
