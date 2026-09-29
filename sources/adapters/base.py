"""
Ingestion Adapter Interface.

Concrete adapters (e.g. an "IP India adapter", a "TKDL adapter") should
implement `fetch_documents()` to pull real documents from an authoritative
source and hand them to `ingestion.pipeline.IngestionPipeline.process()`.

No concrete adapter is implemented here yet -- per RULE 9/16 in the
implementation brief, we do not fabricate scrapers against sites we have
not verified programmatic access to. This module defines the contract so
a real adapter can be dropped in later without touching the rest of the
pipeline.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Iterable


class SourceAdapterNotConfigured(RuntimeError):
    """Raised when an adapter is invoked before it has a real implementation."""


class BaseSourceAdapter(ABC):
    """Contract for a real authoritative-source ingestion adapter."""

    source_id: str

    @abstractmethod
    def fetch_documents(self) -> Iterable[Dict[str, Any]]:
        """Yield raw documents in the shape expected by IngestionPipeline.process().

        Must raise SourceAdapterNotConfigured if the adapter has no working
        fetch implementation yet, rather than returning fabricated content.
        """
        raise NotImplementedError


class UnconfiguredSourceAdapter(BaseSourceAdapter):
    """Placeholder adapter for every registry entry that has no real fetch logic yet."""

    def __init__(self, source_id: str):
        self.source_id = source_id

    def fetch_documents(self) -> Iterable[Dict[str, Any]]:
        raise SourceAdapterNotConfigured(
            f"No ingestion adapter is configured for source '{self.source_id}'. "
            "Only the synthetic Phase-1 corpus is currently ingested for this source's domains."
        )
