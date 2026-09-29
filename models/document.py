"""
Data models for IP-SAKTI Sahayak Ingestion & Retrieval Pipeline.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class DocumentMetadata(BaseModel):
    title: str = Field(..., description="Document title")
    source_name: Optional[str] = None
    issuing_authority: Optional[str] = None
    authority: str = Field(..., description="Issuing authority or body")
    jurisdiction: str = Field(..., description="Jurisdiction, e.g. India or International")
    domain: str = Field(..., description="Domain, e.g. Patents, Traditional Knowledge, ABS, Drug Classification")
    document_type: str = Field(..., description="Document type, e.g. Act, Guidance, Rule, Treaty, Scenario")
    effective_date: str = Field(..., description="Effective date in YYYY-MM-DD format")
    source_url: str = Field(..., description="URL or synthetic URI for source traceability")
    document_url: Optional[str] = None
    source_type: str = Field(default="synthetic", description="Must be 'synthetic' for sample data, 'official' for real data")
    source_id: Optional[str] = None
    authority_level: str = Field(default="development_only", description="primary|secondary|development_only")
    access_status: str = Field(default="public", description="public|restricted|metadata_only|unavailable")
    authoritative: bool = False
    status: str = Field(default="current", description="current|historical|draft")
    source_priority: float = Field(default=0.35, ge=0.0, le=1.0)
    
    # Optional metadata fields
    language: str = Field(default="en")
    original_language: Optional[str] = None
    publication_date: Optional[str] = None
    version: str = Field(default="1.0")
    section: Optional[str] = None
    subsection: Optional[str] = None
    rule: Optional[str] = None
    sub_rule: Optional[str] = None
    regulation: Optional[str] = None
    sub_regulation: Optional[str] = None
    article: Optional[str] = None
    schedule: Optional[str] = None
    paragraph: Optional[str] = None
    chapter: Optional[str] = None
    page_number: Optional[int] = None
    document_hash: Optional[str] = None
    ingested_at: Optional[str] = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    retrieved_at: Optional[str] = None
    content_hash: Optional[str] = None
    last_updated: Optional[str] = None
    parent_document_id: Optional[str] = None
    chunk_id: Optional[str] = None

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert metadata to flat JSON dictionary suitable for Qdrant payload filtering."""
        return {
            "title": self.title,
            "source_name": self.source_name or "",
            "issuing_authority": self.issuing_authority or self.authority,
            "authority": self.authority,
            "jurisdiction": self.jurisdiction,
            "domain": self.domain,
            "document_type": self.document_type,
            "effective_date": self.effective_date,
            "source_url": self.source_url,
            "document_url": self.document_url or self.source_url,
            "source_type": self.source_type,
            "source_id": self.source_id or "",
            "authority_level": self.authority_level,
            "access_status": self.access_status,
            "authoritative": self.authoritative,
            "status": self.status,
            "source_priority": self.source_priority,
            "language": self.language,
            "original_language": self.original_language or self.language,
            "publication_date": self.publication_date or "",
            "version": self.version,
            "section": self.section or "",
            "subsection": self.subsection or "",
            "rule": self.rule or "",
            "sub_rule": self.sub_rule or "",
            "regulation": self.regulation or "",
            "sub_regulation": self.sub_regulation or "",
            "article": self.article or "",
            "schedule": self.schedule or "",
            "paragraph": self.paragraph or "",
            "chapter": self.chapter or "",
            "document_hash": self.document_hash or "",
            "content_hash": self.content_hash or self.document_hash or "",
            "retrieved_at": self.retrieved_at or self.ingested_at or "",
            "ingested_at": self.ingested_at or "",
        }


class Document(BaseModel):
    document_id: str = Field(..., description="Unique document ID")
    title: str = Field(...)
    authority: str = Field(...)
    jurisdiction: str = Field(...)
    domain: str = Field(...)
    document_type: str = Field(...)
    effective_date: str = Field(...)
    source_url: str = Field(...)
    content: str = Field(..., description="Normalized document content")
    metadata: Optional[DocumentMetadata] = None

    def model_post_init(self, __context: Any) -> None:
        if self.metadata is None:
            self.metadata = DocumentMetadata(
                title=self.title,
                authority=self.authority,
                jurisdiction=self.jurisdiction,
                domain=self.domain,
                document_type=self.document_type,
                effective_date=self.effective_date,
                source_url=self.source_url,
                source_type="synthetic" if self.source_url.startswith("synthetic://") else "official"
            )
