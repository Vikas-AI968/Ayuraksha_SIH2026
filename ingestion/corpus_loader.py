"""Load already archived authoritative snapshots into runtime indexes."""
from __future__ import annotations

from pathlib import Path

from ingestion.paths import resolve_repo_path
from models.source import CorpusManifest


def load_manifest_into_indexes(manifest_path: str, pipeline, vector_store, bm25_retriever) -> dict:
    path = resolve_repo_path(manifest_path)
    if not path.is_file():
        return {"documents": 0, "chunks": 0, "failed": 0}
    manifest = CorpusManifest.model_validate_json(path.read_text(encoding="utf-8"))
    documents = chunks = failed = 0
    for entry in manifest.entries:
        normalized_path = resolve_repo_path(entry.normalized_path) if entry.normalized_path else None
        if entry.fetch_status != "ingested" or not normalized_path or not normalized_path.is_file():
            continue
        try:
            text = normalized_path.read_text(encoding="utf-8")
            result = pipeline.process({
                "document_id": entry.document_id, "title": entry.title,
                "authority": entry.authority, "jurisdiction": entry.jurisdiction,
                "domain": entry.domain, "document_type": entry.document_type,
                "effective_date": entry.effective_date or entry.version, "source_url": entry.official_url,
                "document_url": entry.official_url, "source_id": entry.source_id,
                "source_type": "official", "authority_level": "primary",
                "source_priority": 1.0, "access_status": "public",
                "authoritative": True, "status": entry.status, "version": entry.version,
                "content": text,
            })
            if result.get("status") == "success":
                vector_store.upsert_chunks(result["chunks"])
                bm25_retriever.index_chunks(result["chunks"])
                documents += 1
                chunks += len(result["chunks"])
        except Exception:
            failed += 1
    return {"documents": documents, "chunks": chunks, "failed": failed}
