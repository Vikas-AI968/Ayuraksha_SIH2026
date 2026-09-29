from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ingestion.validator import DocumentValidator
from ingestion.pipeline import IngestionPipeline
from vector_store.qdrant_store import QdrantVectorStore
from retrieval.bm25 import BM25Retriever

router = APIRouter(prefix="/api/v1", tags=["Ingestion"])


class IngestRequest(BaseModel):
    document_id: Optional[str] = None
    title: str = Field(..., json_schema_extra={"example": "Sample Ayurveda Patent Guidance"})
    authority: str = Field(..., json_schema_extra={"example": "IP-SAKTI Synthetic Patent Office"})
    jurisdiction: str = Field(..., json_schema_extra={"example": "India"})
    domain: str = Field(..., json_schema_extra={"example": "Patents"})
    document_type: str = Field(..., json_schema_extra={"example": "Guidance"})
    effective_date: str = Field(..., json_schema_extra={"example": "2026-01-01"})
    source_url: str = Field(..., json_schema_extra={"example": "synthetic://patent-guidance-001"})
    content: str = Field(..., json_schema_extra={"example": "Full text of document..."})
    source_type: Optional[str] = Field(default="synthetic")
    version: Optional[str] = Field(default="1.0")


class IngestResponse(BaseModel):
    document_id: str
    status: str
    chunk_count: int
    embedding_status: str
    index_status: str
    duration_seconds: float
    logs: List[str]
    errors: Optional[List[str]] = None


# Module-level singletons (assigned by main)
ingestion_pipeline: Optional[IngestionPipeline] = None
vector_store: Optional[QdrantVectorStore] = None
bm25_retriever: Optional[BM25Retriever] = None


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(payload: IngestRequest):
    """
    Ingests a document through:
    Fetch -> Parse -> Normalize -> Metadata -> Chunk -> Embed -> Qdrant Index + BM25 Index
    """
    validator = DocumentValidator()
    doc_dict = payload.model_dump()
    is_valid, validation_errors = validator.validate(doc_dict)
    if not is_valid:
        raise HTTPException(status_code=400, detail={"message": "Document validation failed", "errors": validation_errors})

    if ingestion_pipeline is None or vector_store is None or bm25_retriever is None:
        raise HTTPException(status_code=500, detail="Ingestion pipeline or indexes not initialized.")

    # Execute ingestion pipeline
    res = ingestion_pipeline.process(doc_dict)
    if res["status"] == "failed":
        raise HTTPException(status_code=500, detail={"message": "Ingestion stage failed", "error": res.get("error"), "logs": res["logs"]})

    chunks = res["chunks"]
    embedding_status = "skipped"
    index_status = "skipped"

    if chunks:
        # Upsert vectors to Qdrant
        vector_store.upsert_chunks(chunks)
        embedding_status = "success"

        # Index in BM25 and persist immediately so a restart does not silently
        # revert to whatever was last saved by the offline ingestion script
        # (Qdrant is already durable on upsert; BM25Okapi is in-memory only
        # and must be explicitly pickled to disk after every mutation).
        bm25_retriever.index_chunks(chunks)
        try:
            bm25_retriever.save()
            index_status = "success"
        except Exception as exc:  # persistence failure must not silently claim success
            index_status = "in_memory_only"
            res.setdefault("logs", []).append(f"BM25 index updated in memory but persistence failed: {exc}")

    return IngestResponse(
        document_id=res["document_id"],
        status=res["status"],
        chunk_count=res["chunk_count"],
        embedding_status=embedding_status,
        index_status=index_status,
        duration_seconds=res.get("duration_seconds", 0.0),
        logs=res["logs"]
    )
