"""
Retrieval API Endpoint - Stage 16 of Prompt Specification.
POST /api/v1/retrieve

Also hosts the Intelligence Layer's main endpoint, POST /api/v1/query
(Query Intake -> Analysis -> Classification -> Routing -> Retrieval ->
Evidence Pack -> LLM Reasoning -> Citation Validation -> Confidence ->
Abstention -> Structured Response).
"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from retrieval.semantic import SemanticRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import LightweightScoringReranker
from retrieval.citation import CitationTraceabilityService
from models.chunk import RetrievalResult, RetrievalResponse
from models.query import QueryIntakeRequest
from models.answer import QueryResponse
from services.orchestrator import QueryOrchestrator

router = APIRouter(prefix="/api/v1", tags=["Retrieval"])


class RetrievalRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "Can I patent a traditional Ayurvedic formulation under Section 3(p)?"})
    top_k: int = Field(default=5, ge=1, le=50)
    candidate_k: int = Field(default=20, ge=5, le=100)
    jurisdiction: Optional[str] = Field(default=None, json_schema_extra={"example": "India"})
    domain: Optional[str] = Field(default=None, json_schema_extra={"example": "Patents"})
    authority: Optional[str] = Field(default=None)
    document_type: Optional[str] = Field(default=None)
    status: Optional[str] = Field(default="current", pattern="^(current|historical|draft)$")
    source_priority_min: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    authoritative: Optional[bool] = Field(default=None)
    retrieval_method: str = Field(default="hybrid", json_schema_extra={"example": "hybrid"})  # 'hybrid', 'semantic', 'bm25'
    use_reranker: bool = Field(default=True)


# Module-level singletons (assigned by main)
semantic_retriever: Optional[SemanticRetriever] = None
bm25_retriever: Optional[BM25Retriever] = None
hybrid_retriever: Optional[HybridRetriever] = None
reranker: Optional[LightweightScoringReranker] = None
citation_service = CitationTraceabilityService()
orchestrator: Optional[QueryOrchestrator] = None


@router.post("/retrieve", response_model=RetrievalResponse)
def retrieve_evidence(payload: RetrievalRequest):
    """
    Executes citation-grounded retrieval:
    User Query -> Metadata Filtering -> Semantic Search + BM25 Search -> Hybrid Merge -> Top-K Reranking
    """
    if hybrid_retriever is None or semantic_retriever is None or bm25_retriever is None:
        raise HTTPException(status_code=500, detail="Retrieval engine services not initialized.")

    query = payload.query
    top_k = payload.top_k
    method = payload.retrieval_method.lower()

    filters = {
        "jurisdiction": payload.jurisdiction,
        "domain": payload.domain,
        "authority": payload.authority,
        "document_type": payload.document_type,
        "status": payload.status,
        "source_priority_min": payload.source_priority_min,
        "authoritative": payload.authoritative,
    }

    if method == "semantic":
        candidates = semantic_retriever.search(query, top_k=payload.candidate_k, **filters)
    elif method == "bm25":
        candidates = bm25_retriever.search(query, top_k=payload.candidate_k, **filters)
    else:  # hybrid
        candidates = hybrid_retriever.search(
            query=query,
            top_k=payload.candidate_k,
            candidate_k=payload.candidate_k,
            jurisdiction=payload.jurisdiction,
            domain=payload.domain,
            authority=payload.authority,
            document_type=payload.document_type,
            status=payload.status,
            source_priority_min=payload.source_priority_min,
            authoritative=payload.authoritative,
        )

    # Apply Top-K Reranking if enabled
    if payload.use_reranker and reranker:
        final_results = reranker.rerank(query, candidates, top_k=top_k)
    else:
        final_results = candidates[:top_k]

    # Ensure citations are built
    for res in final_results:
        if res.citation is None:
            res.build_citation()

    applied_filters = {k: v for k, v in filters.items() if v is not None}

    return RetrievalResponse(
        query=query,
        total_candidates=len(candidates),
        top_k=len(final_results),
        results=final_results,
        filters_applied=applied_filters
    )


@router.get("/grounding", response_model=Dict[str, Any])
def get_grounding_context(
    query: str = Query(..., description="User query"),
    top_k: int = Query(5, ge=1, le=20),
    jurisdiction: Optional[str] = Query(None),
    domain: Optional[str] = Query(None)
):
    """
    Returns structured claim-to-evidence grounding chain for LLM context injection.
    """
    request = RetrievalRequest(query=query, top_k=top_k, jurisdiction=jurisdiction, domain=domain)
    res = retrieve_evidence(request)
    return citation_service.generate_grounding_context(query, res.results)


@router.post("/query", response_model=QueryResponse, tags=["Intelligence Layer"])
def run_query(payload: QueryIntakeRequest):
    """
    Full Intelligence Layer pipeline:
    Query Intake -> Query Analysis -> Product Classification -> Jurisdiction/Domain
    Routing -> Retrieval + Reranking -> Evidence Pack -> LLM Reasoning -> Citation
    Validation -> Confidence -> Safe Abstention -> Structured Response.
    """
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Query orchestrator is not initialized.")
    try:
        return orchestrator.run(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query pipeline failed: {e}")


@router.get("/query/{query_id}", response_model=Dict[str, Any], tags=["Intelligence Layer"])
def get_query_by_id(query_id: str):
    """Fetch a previously computed /api/v1/query response by its query_id."""
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Query orchestrator is not initialized.")
    record = orchestrator.log_store.get(query_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No query found with query_id='{query_id}'.")
    return record
