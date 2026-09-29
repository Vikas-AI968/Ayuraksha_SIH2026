"""
Evidence Pack models sitting between Retrieval and LLM Reasoning.
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    evidence_id: str = Field(..., description="Stable, response-local id, e.g. 'E001'")
    document_id: str
    chunk_id: str
    source_id: Optional[str] = None
    official_url: Optional[str] = None
    authority: str
    jurisdiction: str
    # Optional fields with defaults so lightweight test fixtures can omit them
    domain: str = ""
    document_type: str = ""
    title: str = ""
    section: Optional[str] = None
    subsection: Optional[str] = None
    rule: Optional[str] = None
    sub_rule: Optional[str] = None
    regulation: Optional[str] = None
    sub_regulation: Optional[str] = None
    article: Optional[str] = None
    schedule: Optional[str] = None
    page: Optional[int] = None
    text: str
    relevance_score: float = 0.0
    rerank_score: Optional[float] = None
    authority_score: float = 0.0
    source_priority: float = 0.0
    authority_level: str = "development_only"
    status: str = "current"
    provenance: str = "retrieval"
    retrieval_method: str = ""
    citation: str = ""
    source_url: str = ""
    source_type: str = Field(default="statute", description="'synthetic' for demo/test data, else 'official'/'treaty'/'regulation'/'guideline'")
    access_status: str = "public"
    effective_date: str = ""
    version: str = ""


class EvidencePack(BaseModel):
    query: str
    items: List[EvidenceItem] = Field(default_factory=list)
    synthetic_only: bool = True
    jurisdictions_present: List[str] = Field(default_factory=list)
    domains_present: List[str] = Field(default_factory=list)
    average_relevance: float = 0.0
    top_relevance: float = 0.0
