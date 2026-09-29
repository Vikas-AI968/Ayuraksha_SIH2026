"""
Supporting API endpoints:
GET /api/v1/sources    -- authoritative source registry
GET /api/v1/documents  -- documents currently indexed in this running instance
GET /api/v1/health     -- liveness/readiness + component status
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

from retrieval.bm25 import BM25Retriever
from sources.registry import get_default_registry
from models.source import CorpusManifest
from pathlib import Path
import os

router = APIRouter(prefix="/api/v1", tags=["Sources & Health"])

# Module-level singletons (assigned by main)
bm25_retriever: Optional[BM25Retriever] = None
vector_store = None
orchestrator = None

_registry = get_default_registry()


@router.get("/sources")
def list_sources(jurisdiction: Optional[str] = None, domain: Optional[str] = None):
    """Lists the authoritative source registry (real bodies + the synthetic test corpus)."""
    entries = _registry.all()
    if jurisdiction:
        entries = [e for e in entries if e.jurisdiction.lower() == jurisdiction.lower()]
    if domain:
        entries = [e for e in entries if domain in e.domains]
    return {
        "summary": _registry.summary(),
        "sources": [e.model_dump() for e in entries],
    }


@router.get("/documents")
def list_documents(jurisdiction: Optional[str] = None, domain: Optional[str] = None, limit: int = 100):
    """Lists documents currently indexed (chunk-derived) in this running instance."""
    if bm25_retriever is None:
        return {"total_indexed_chunks": 0, "documents": []}

    seen: Dict[str, Dict[str, Any]] = {}
    for chunk in bm25_retriever.chunks:
        md = chunk.metadata
        if jurisdiction and md.jurisdiction.lower() != jurisdiction.lower():
            continue
        if domain and md.domain != domain:
            continue
        if chunk.document_id not in seen:
            seen[chunk.document_id] = {
                "document_id": chunk.document_id,
                "title": md.title,
                "authority": md.authority,
                "jurisdiction": md.jurisdiction,
                "domain": md.domain,
                "document_type": md.document_type,
                "source_type": md.source_type,
                "source_url": md.source_url,
                "effective_date": md.effective_date,
                "document_status": _registry.document_status(chunk.document_id),
                "chunk_count": 0,
            }
        seen[chunk.document_id]["chunk_count"] += 1

    documents: List[Dict[str, Any]] = list(seen.values())[:limit]
    return {"total_indexed_chunks": len(bm25_retriever.chunks), "total_documents": len(seen), "documents": documents}


@router.get("/health")
def health():
    components = {
        "bm25_retriever": bm25_retriever is not None,
        "vector_store": vector_store is not None,
        "orchestrator": orchestrator is not None,
    }
    indexed_chunks = len(bm25_retriever.chunks) if bm25_retriever else 0
    status = "healthy" if all(components.values()) and indexed_chunks > 0 else "degraded"
    return {
        "status": status,
        "components": components,
        "indexed_chunks": indexed_chunks,
        "qdrant_collection": vector_store.collection_name if vector_store else None,
        "embedding_model": getattr(vector_store.embedding_provider, "model_name", "hash") if vector_store else None,
        "llm_provider": os.environ.get("LLM_PROVIDER", "auto"),
        "multilingual_languages": ["en", "hi", "te"],
        "source_registry_summary": _registry.summary(),
    }


@router.get("/corpus/health")
def corpus_health():
    manifest_path = Path(os.environ.get("AUTHORITATIVE_MANIFEST", "data/authoritative/manifests/corpus_manifest.json"))
    manifest = CorpusManifest.model_validate_json(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else CorpusManifest()
    entries = manifest.entries
    authoritative = [entry for entry in entries if entry.authoritative and entry.fetch_status == "ingested"]
    synthetic = [entry for entry in entries if not entry.authoritative]
    graph = orchestrator.knowledge_graph if orchestrator else None
    entity_count = relationship_count = 0
    if graph:
        entity_count = graph._connection.execute("SELECT COUNT(*) FROM knowledge_entities").fetchone()[0]
        relationship_count = graph._connection.execute("SELECT COUNT(*) FROM knowledge_relationships").fetchone()[0]
    qdrant_points = 0
    if vector_store is not None:
        try:
            qdrant_points = vector_store.client.count(vector_store.collection_name).count
        except Exception:
            qdrant_points = 0
    return {
        "mode": "development_synthetic" if os.environ.get("ALLOW_SYNTHETIC_CORPUS", "false").lower() in {"1", "true", "yes"} else "authoritative",
        "authoritative_documents": len(authoritative),
        "current_documents": sum(entry.status == "current" for entry in authoritative),
        "historical_documents": sum(entry.status in {"historical", "superseded"} for entry in authoritative),
        "draft_documents": sum(entry.status == "draft" for entry in authoritative),
        "synthetic_documents": len(synthetic),
        "authoritative_chunks": sum(entry.chunk_count for entry in authoritative),
        "failed_documents": sum(entry.fetch_status == "failed" for entry in entries),
        "bm25_chunks": len(bm25_retriever.chunks) if bm25_retriever else 0,
        "qdrant_points": qdrant_points,
        "graph_entities": entity_count,
        "graph_relationships": relationship_count,
        "healthy": not any(entry.fetch_status == "failed" for entry in entries),
    }
