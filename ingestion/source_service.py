"""Controlled source-adapter execution and ingestion status reporting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from ingestion.pipeline import IngestionPipeline
from sources.adapters.base import BaseSourceAdapter, SourceAdapterNotConfigured


@dataclass
class SourceIngestionReport:
    source_id: str
    status: str = "not_ingested"
    document_count: int = 0
    chunk_count: int = 0
    errors: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)


class SourceIngestionService:
    """Runs only explicitly registered adapters and preserves failure visibility."""

    def __init__(self, pipeline: IngestionPipeline):
        self.pipeline = pipeline

    def ingest(self, adapter: BaseSourceAdapter) -> SourceIngestionReport:
        report = SourceIngestionReport(source_id=adapter.source_id)
        try:
            documents = adapter.fetch_documents()
            for raw_document in documents:
                result = self.pipeline.process(raw_document)
                if result.get("status") not in {"success", "skipped"}:
                    report.errors.append(str(result.get("error", "ingestion failed")))
                    continue
                report.document_count += 1
                report.chunk_count += int(result.get("chunk_count", 0))
                report.document_ids.append(str(result.get("document_id")))
            report.status = "ingested" if report.document_count and not report.errors else (
                "partial" if report.document_count else "failed"
            )
        except SourceAdapterNotConfigured as exc:
            report.status = "not_ingested"
            report.errors.append(str(exc))
        except Exception as exc:
            report.status = "failed"
            report.errors.append(str(exc))
        return report
