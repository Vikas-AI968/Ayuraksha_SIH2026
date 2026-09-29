"""
Chunk and Retrieval Result models for IP-SAKTI Sahayak.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from models.document import DocumentMetadata


class Chunk(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk ID")
    document_id: str = Field(..., description="Parent document ID")
    text: str = Field(..., description="Chunk content text")
    metadata: DocumentMetadata = Field(..., description="Metadata inherited from document plus chunk specifics")
    section: Optional[str] = None
    subsection: Optional[str] = None
    rule: Optional[str] = None
    sub_rule: Optional[str] = None
    regulation: Optional[str] = None
    sub_regulation: Optional[str] = None
    article: Optional[str] = None
    schedule: Optional[str] = None
    chapter: Optional[str] = None
    paragraph: Optional[int] = None
    vector: Optional[List[float]] = None

    def to_qdrant_payload(self) -> Dict[str, Any]:
        payload = self.metadata.to_qdrant_payload()
        payload.update({
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "section": self.section or self.metadata.section or "",
            "chapter": self.chapter or self.metadata.chapter or "",
            "subsection": self.subsection or self.metadata.subsection or "",
            "rule": self.rule or self.metadata.rule or "",
            "sub_rule": self.sub_rule or self.metadata.sub_rule or "",
            "regulation": self.regulation or self.metadata.regulation or "",
            "sub_regulation": self.sub_regulation or self.metadata.sub_regulation or "",
            "article": self.article or self.metadata.article or "",
            "schedule": self.schedule or self.metadata.schedule or "",
            "paragraph": self.paragraph or 0,
        })
        return payload


class CitationTrace(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    authority: str
    jurisdiction: str
    domain: str
    document_type: str
    effective_date: str
    source_url: str
    source_type: str
    version: str
    status: str = "current"
    page: Optional[int] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    official_url: Optional[str] = None
    citation_text: str


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None
    retrieval_method: str = Field(..., description="semantic | bm25 | hybrid | reranked")
    text: str
    metadata: DocumentMetadata
    citation: Optional[CitationTrace] = None

    def build_citation(self) -> CitationTrace:
        location = self.metadata.section or self.metadata.rule or self.metadata.regulation or self.metadata.article
        if self.metadata.subsection:
            location = f"{location}({self.metadata.subsection})" if location else f"({self.metadata.subsection})"
        page = f", p. {self.metadata.page_number}" if self.metadata.page_number else ""
        citation_str = (
            f"[{self.metadata.authority}] {self.metadata.title} "
            f"({self.metadata.document_type}, {self.metadata.jurisdiction}, {location or 'location unavailable'}{page}, "
            f"status={self.metadata.status}, version={self.metadata.version}) - {self.metadata.source_url}"
        )
        self.citation = CitationTrace(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            title=self.metadata.title,
            authority=self.metadata.authority,
            jurisdiction=self.metadata.jurisdiction,
            domain=self.metadata.domain,
            document_type=self.metadata.document_type,
            effective_date=self.metadata.effective_date,
            source_url=self.metadata.source_url,
            source_type=self.metadata.source_type,
            version=self.metadata.version,
            status=self.metadata.status,
            page=self.metadata.page_number,
            section=location,
            subsection=self.metadata.subsection,
            official_url=self.metadata.source_url,
            citation_text=citation_str
        )
        return self.citation


class RetrievalResponse(BaseModel):
    query: str
    total_candidates: int
    top_k: int
    results: List[RetrievalResult]
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
