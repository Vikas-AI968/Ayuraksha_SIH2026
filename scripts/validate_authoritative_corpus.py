"""Validate the persisted authoritative corpus manifest and metadata."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import bootstrap_env  # noqa: F401  # loads .env before any other project import touches os.environ

import json
import os
from collections import Counter
from pathlib import Path

from ingestion.paths import resolve_repo_path
from models.source import CorpusManifest
from services.knowledge_graph import KnowledgeGraph


def validate(manifest_path: str = "data/authoritative/manifests/corpus_manifest.json") -> tuple[dict, list[str]]:
    errors: list[str] = []
    path = resolve_repo_path(manifest_path)
    manifest = CorpusManifest.model_validate_json(path.read_text(encoding="utf-8")) if path.is_file() else CorpusManifest()
    entries = manifest.entries
    authoritative = [entry for entry in entries if entry.authoritative]
    for entry in authoritative:
        for field in ("source_id", "authority", "official_url", "title", "jurisdiction", "version", "status", "content_hash"):
            if not getattr(entry, field, None):
                errors.append(f"{entry.document_id}: missing {field}")
        if not entry.raw_path or not resolve_repo_path(entry.raw_path).is_file():
            errors.append(f"{entry.document_id}: missing raw archive")
        if not entry.normalized_path or not resolve_repo_path(entry.normalized_path).is_file():
            errors.append(f"{entry.document_id}: missing normalized document")
    hashes = [entry.content_hash for entry in entries if entry.content_hash]
    duplicate_hashes = sum(count - 1 for count in Counter(hashes).values() if count > 1)
    duplicate_documents = len(entries) - len({entry.document_id for entry in entries})
    invalid_status = [entry.document_id for entry in entries if entry.status not in {"current", "historical", "superseded", "draft", "unknown"}]
    missing_structure = sum(
        1 for entry in authoritative
        if entry.domain in {"Patents", "Drug Classification", "Biodiversity/ABS"} and entry.chunk_count == 0
    )
    graph = KnowledgeGraph()
    graph_entities = graph._connection.execute("SELECT COUNT(*) FROM knowledge_entities").fetchone()[0]
    graph_relationships = graph._connection.execute("SELECT COUNT(*) FROM knowledge_relationships").fetchone()[0]
    graph_provenance_rows = graph._connection.execute(
        "SELECT metadata_json FROM knowledge_relationships"
    ).fetchall()
    graph_provenance = sum(
        1 for (payload,) in graph_provenance_rows
        if json.loads(payload).get("provenance")
    )
    by_domain = Counter(entry.domain for entry in authoritative)
    by_authority = Counter(entry.authority for entry in authoritative)
    errors.extend(f"{document_id}: invalid status" for document_id in invalid_status)
    if duplicate_documents:
        errors.append(f"duplicate document IDs: {duplicate_documents}")
    report = {
        "total_documents": len(entries),
        "authoritative_documents": len(authoritative),
        "synthetic_documents": sum(not entry.authoritative for entry in entries),
        "current_documents": sum(entry.status == "current" for entry in entries),
        "historical_documents": sum(entry.status in {"historical", "superseded"} for entry in entries),
        "draft_documents": sum(entry.status == "draft" for entry in entries),
        "failed_documents": sum(entry.fetch_status == "failed" for entry in entries),
        "chunks": sum(entry.chunk_count for entry in entries),
        "duplicate_hashes": duplicate_hashes,
        "duplicate_documents": duplicate_documents,
        "missing_structure_documents": missing_structure,
        "invalid_status_documents": invalid_status,
        "documents_by_domain": dict(by_domain),
        "documents_by_authority": dict(by_authority),
        "graph_entities": graph_entities,
        "graph_relationships": graph_relationships,
        "graph_relationships_with_provenance": graph_provenance,
        "errors": errors,
        "healthy": not errors,
    }
    return report, errors


def main() -> int:
    report, errors = validate(os.environ.get("AUTHORITATIVE_MANIFEST", "data/authoritative/manifests/corpus_manifest.json"))
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
