"""Source entities and explicitly configured document records."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class ConfiguredDocument(BaseModel):
    document_id: str
    source_id: str
    source_name: str
    issuing_authority: str
    official_url: str
    document_title: str
    document_type: str
    jurisdiction: str
    domain: str
    version: str
    status: str = "current"
    publication_date: Optional[str] = None
    effective_date: Optional[str] = None
    language: str = "en"
    source_priority: float = 1.0
    access_status: str = "public"
    authoritative: bool = True
    local_path: Optional[str] = None
    notes: Optional[str] = None


class IngestionManifestEntry(BaseModel):
    document_id: str
    source_id: str
    authority: str = ""
    title: str
    official_url: str
    document_type: str
    jurisdiction: str
    domain: str
    version: str
    status: str
    effective_date: Optional[str] = None
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    content_hash: Optional[str] = None
    source_hash: Optional[str] = None
    authoritative: bool = False
    fetch_status: str = "not_attempted"
    error: Optional[str] = None
    raw_path: Optional[str] = None
    normalized_path: Optional[str] = None
    chunk_count: int = 0


class CorpusManifest(BaseModel):
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    entries: list[IngestionManifestEntry] = Field(default_factory=list)
