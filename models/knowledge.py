"""Structured regulatory knowledge models used by the graph and decision layer."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class KnowledgeStatus(str, Enum):
    CURRENT = "current"
    HISTORICAL = "historical"
    DRAFT = "draft"


class KnowledgeDocumentType(str, Enum):
    STATUTE = "statute"
    RULE = "rule"
    GUIDELINE = "guideline"
    REGULATION = "regulation"
    ORDER = "order"
    TREATY = "treaty"
    DATABASE_REGISTRY = "database_registry"
    SECONDARY_REFERENCE = "secondary_reference"


class KnowledgeEntity(BaseModel):
    entity_id: str
    name: str
    entity_type: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeRelationship(BaseModel):
    relationship_id: str
    source_entity_id: str
    relationship_type: str
    target_entity_id: str
    source_document_id: Optional[str] = None
    evidence_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeDocument(BaseModel):
    document_id: str
    source_name: str
    issuing_authority: str
    official_url: Optional[str] = None
    document_title: str
    document_type: KnowledgeDocumentType
    jurisdiction: str
    category: str
    version: str = "1.0"
    publication_date: Optional[str] = None
    effective_date: Optional[str] = None
    status: KnowledgeStatus = KnowledgeStatus.CURRENT
    section_identifier: Optional[str] = None
    page: Optional[int] = None
    language: str = "en"
    access_notes: Optional[str] = None
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_hash: Optional[str] = None
    authoritative: bool = False


class KnowledgeGraphContext(BaseModel):
    entities: list[KnowledgeEntity] = Field(default_factory=list)
    relationships: list[KnowledgeRelationship] = Field(default_factory=list)
    documents: list[KnowledgeDocument] = Field(default_factory=list)
    matched_entity_ids: list[str] = Field(default_factory=list)
    traversal_depth: int = 0
