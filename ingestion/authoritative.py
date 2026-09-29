"""Controlled authoritative-document fetch, archive, ingestion, and manifest updates."""
from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
import pypdf

from ingestion.paths import resolve_repo_path, to_portable_relpath
from ingestion.pipeline import IngestionPipeline
from models.source import ConfiguredDocument, CorpusManifest, IngestionManifestEntry
from sources.registry import SourceRegistry

logger = logging.getLogger("authoritative_ingestion")

# Fetch resilience knobs (all overridable via environment so hackathon judges
# / CI can tune without code changes). These exist specifically to stop the
# ingestion process from appearing to hang forever on a slow/stalled remote
# PDF download, a slow parse, or a slow embedding-model download.
_CONNECT_TIMEOUT = float(os.environ.get("AUTHORITATIVE_FETCH_CONNECT_TIMEOUT", "10.0"))
_READ_TIMEOUT = float(os.environ.get("AUTHORITATIVE_FETCH_READ_TIMEOUT", "45.0"))
_WRITE_TIMEOUT = float(os.environ.get("AUTHORITATIVE_FETCH_WRITE_TIMEOUT", "10.0"))
_POOL_TIMEOUT = float(os.environ.get("AUTHORITATIVE_FETCH_POOL_TIMEOUT", "10.0"))
_MAX_RETRIES = int(os.environ.get("AUTHORITATIVE_FETCH_MAX_RETRIES", "2"))
_RETRY_BACKOFF_SECONDS = float(os.environ.get("AUTHORITATIVE_FETCH_RETRY_BACKOFF", "2.0"))
_REUSE_CACHED_RAW = os.environ.get("AUTHORITATIVE_REUSE_CACHED_RAW", "true").lower() in {"1", "true", "yes"}

# Errors worth retrying: transient network conditions. Anything else (host
# validation failures, restricted-source PermissionError, 4xx client errors)
# is not retried -- retrying those would just waste time on a guaranteed
# repeat failure.
_RETRYABLE_EXCEPTIONS = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)


class AuthoritativeCorpusService:
    def __init__(self, pipeline: IngestionPipeline, vector_store=None, bm25_retriever=None, graph=None,
                 root: str = "data/authoritative"):
        self.pipeline = pipeline
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.graph = graph
        # Anchor to the repository root rather than the current working
        # directory, so this works the same whether invoked via
        # `python -m scripts.ingest_authoritative_corpus` from the repo root
        # or imported from a different CWD (e.g. under a test runner).
        self.root = resolve_repo_path(root) if not Path(root).is_absolute() else Path(root)
        self.raw_dir = self.root / "raw"
        self.normalized_dir = self.root / "normalized"
        self.manifest_path = self.root / "manifests" / "corpus_manifest.json"
        for directory in (self.raw_dir, self.normalized_dir, self.manifest_path.parent, self.root / "failed"):
            directory.mkdir(parents=True, exist_ok=True)

    def _find_cached_raw(self, document_id: str):
        """Returns a previously-archived raw file for this document_id, if
        one exists on disk from an earlier ingestion run."""
        if not self.raw_dir.is_dir():
            return None
        matches = sorted(self.raw_dir.glob(f"{document_id}-*"))
        return matches[0] if matches else None

    def _load_manifest(self) -> CorpusManifest:
        if not self.manifest_path.is_file():
            return CorpusManifest()
        return CorpusManifest.model_validate_json(self.manifest_path.read_text(encoding="utf-8"))

    def _save_manifest(self, manifest: CorpusManifest) -> None:
        self.manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def _verify_url(document: ConfiguredDocument) -> None:
        parsed = urlparse(document.official_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("official_url must use http or https")
        host = parsed.hostname or ""
        source_host = urlparse(document.official_url).hostname or ""
        if not host or source_host != host:
            raise ValueError("official_url host validation failed")
        if document.source_id == "tkdl" or document.access_status == "restricted":
            raise PermissionError(f"Source '{document.source_id}' is restricted and cannot be bulk fetched")

    @staticmethod
    def _extract_content(raw_bytes: bytes, content_type: str, title: str) -> str:
        is_pdf = "pdf" in content_type.lower() or raw_bytes[:4] == b"%PDF"
        if is_pdf:
            import io
            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            pages = []
            for page_number, page in enumerate(reader.pages, start=1):
                pages.append(f"--- Page {page_number} ---\n{page.extract_text() or ''}")
            return "\n\n".join(pages)
        soup = BeautifulSoup(raw_bytes, "html.parser")
        for node in soup(["script", "style", "noscript"]):
            node.decompose()
        heading_lines = []
        for element in soup.find_all(["h1", "h2", "h3", "h4"]):
            text = " ".join(element.get_text(" ", strip=True).split())
            if text:
                heading_lines.append(text)
        body = soup.get_text("\n", strip=True)
        return "\n".join([title, *heading_lines, body])

    def ingest_document(self, document: ConfiguredDocument, timeout: float = 45.0) -> IngestionManifestEntry:
        manifest = self._load_manifest()
        entry = IngestionManifestEntry(
            document_id=document.document_id, source_id=document.source_id, authority=document.issuing_authority,
            title=document.document_title,
            official_url=document.official_url, document_type=document.document_type,
            jurisdiction=document.jurisdiction, domain=document.domain, version=document.version,
            status=document.status, effective_date=document.effective_date,
            authoritative=document.authoritative,
        )
        try:
            self._verify_url(document)
            headers = {"User-Agent": "IP-SAKTI-Sahayak/1.0 authoritative-ingestion"}
            if document.local_path:
                logger.info("[%s] reading local file %s", document.document_id, document.local_path)
                raw_path = resolve_repo_path(document.local_path)
                raw_bytes = raw_path.read_bytes()
                content_type = mimetypes.guess_type(str(raw_path))[0] or ""
                logger.info("[%s] local file read: %d bytes", document.document_id, len(raw_bytes))
            elif _REUSE_CACHED_RAW and (cached_raw_path := self._find_cached_raw(document.document_id)) is not None:
                # Authoritative sources rarely change. Re-running ingestion
                # (idempotent re-index, restart recovery) should not have to
                # re-fetch every official PDF from the network every time --
                # reuse the already-archived raw bytes from a prior run. Set
                # AUTHORITATIVE_REUSE_CACHED_RAW=false to force a fresh fetch.
                logger.info("[INGEST] document=%s FETCH (cached, offline reuse) -> %s",
                            document.document_id, cached_raw_path)
                raw_bytes = cached_raw_path.read_bytes()
                content_type = mimetypes.guess_type(str(cached_raw_path))[0] or ""
            else:
                logger.info("[INGEST] document=%s FETCH", document.document_id)
                # Explicit connect/read/write/pool timeouts (rather than a
                # single float) so a stalled TCP connection or a stalled body
                # read cannot block the ingestion process forever. A bounded
                # retry loop absorbs transient network blips without masking
                # a genuinely broken/unreachable source.
                fetch_timeout = httpx.Timeout(
                    connect=_CONNECT_TIMEOUT, read=timeout or _READ_TIMEOUT,
                    write=_WRITE_TIMEOUT, pool=_POOL_TIMEOUT,
                )
                attempt = 0
                while True:
                    attempt += 1
                    logger.info("[%s] fetching %s (attempt %d/%d)",
                                document.document_id, document.official_url, attempt, _MAX_RETRIES + 1)
                    try:
                        response = httpx.get(
                            document.official_url, headers=headers,
                            timeout=fetch_timeout, follow_redirects=True,
                        )
                        response.raise_for_status()
                        break
                    except _RETRYABLE_EXCEPTIONS as exc:
                        if attempt > _MAX_RETRIES:
                            raise
                        logger.warning(
                            "[%s] fetch attempt %d failed (%s: %s); retrying in %.1fs",
                            document.document_id, attempt, type(exc).__name__, exc, _RETRY_BACKOFF_SECONDS,
                        )
                        time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
                logger.info("[%s] fetch complete: HTTP %s, %s bytes",
                            document.document_id, response.status_code,
                            response.headers.get("content-length", "unknown"))
                final_host = urlparse(str(response.url)).hostname or ""
                expected_host = urlparse(document.official_url).hostname or ""
                if final_host != expected_host:
                    raise ValueError(f"redirected away from configured official host: {final_host}")
                raw_bytes = response.content
                content_type = response.headers.get("content-type", "")
            content_hash = hashlib.sha256(raw_bytes).hexdigest()
            logger.info("[INGEST] document=%s SHA256=%s", document.document_id, content_hash[:12])

            # --- Idempotency (Section 5 requirement) ---------------------------------
            # An unchanged document (same content hash as the last successful
            # ingestion) must not be re-parsed, re-chunked, re-embedded, or
            # re-upserted. We still recompute the lightweight chunk objects
            # (cheap: regex-based legal-structure chunking, no ML) so this
            # process's BM25/graph can be populated, but we skip the expensive
            # embedding + Qdrant upsert entirely and never write duplicate points.
            previously_ingested = next(
                (e for e in manifest.entries
                 if e.document_id == document.document_id and e.content_hash == content_hash
                 and e.fetch_status == "ingested"),
                None,
            )
            raw_extension = ".pdf" if "pdf" in content_type.lower() or raw_bytes[:4] == b"%PDF" else ".html"
            raw_path = self.raw_dir / f"{document.document_id}-{content_hash[:12]}{raw_extension}"
            if not raw_path.is_file():
                raw_path.write_bytes(raw_bytes)
            normalized_path = self.normalized_dir / f"{document.document_id}-{content_hash[:12]}.txt"
            # The manifest alone is not proof the vectors still exist: a fresh
            # checkout, a deleted/recreated Qdrant data directory, or a
            # restored manifest without its matching Qdrant collection would
            # all report "previously_ingested" while the collection is
            # actually empty for this document. Verify against the vector
            # store itself before trusting the cache, so ingestion self-heals
            # instead of leaving semantic search permanently empty.
            vector_already_present = True
            if self.vector_store is not None:
                try:
                    vector_already_present = self.vector_store.count_for_document(document.document_id) > 0
                except Exception as exc:
                    logger.warning(
                        "[INGEST] document=%s could not verify existing vector count (%s); "
                        "treating as NOT present and re-embedding to be safe.",
                        document.document_id, exc,
                    )
                    vector_already_present = False
            if previously_ingested is not None and normalized_path.is_file() and vector_already_present:
                logger.info(
                    "[INGEST] document=%s UNCHANGED (content_hash matches prior ingestion) -- "
                    "skipping embedding + Qdrant upsert; reusing cached chunk/version metadata",
                    document.document_id,
                )
                text = normalized_path.read_text(encoding="utf-8")
                skip_embedding = True
            else:
                if previously_ingested is not None and normalized_path.is_file() and not vector_already_present:
                    logger.warning(
                        "[INGEST] document=%s manifest says already ingested but the vector store has "
                        "0 points for this document_id -- manifest/index drift detected. Re-embedding "
                        "and re-upserting to repair it instead of trusting the stale cache.",
                        document.document_id,
                    )
                text = self._extract_content(raw_bytes, content_type, document.document_title)
                logger.info("[INGEST] document=%s PDF PARSE extracted %d characters", document.document_id, len(text))
                normalized_path.write_text(text, encoding="utf-8")
                logger.info("[INGEST] document=%s NORMALIZE complete", document.document_id)
                skip_embedding = False
            payload: dict[str, Any] = {
                "document_id": document.document_id, "title": document.document_title,
                "authority": document.issuing_authority, "jurisdiction": document.jurisdiction,
                "domain": document.domain, "document_type": document.document_type,
                "effective_date": document.effective_date or document.publication_date or "unknown",
                "publication_date": document.publication_date, "source_url": document.official_url,
                "document_url": document.official_url, "source_id": document.source_id,
                "source_name": document.source_name, "source_type": "official",
                "authority_level": "primary" if document.source_priority >= 0.9 else "secondary",
                "source_priority": document.source_priority, "access_status": document.access_status,
                "authoritative": document.authoritative, "status": document.status,
                "version": document.version, "language": document.language, "content": text,
                "source_hash": content_hash, "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
            result = self.pipeline.process(payload)
            if result.get("status") not in {"success", "skipped"}:
                raise RuntimeError(str(result.get("error", "pipeline failed")))
            chunks = result.get("chunks", [])
            logger.info("[INGEST] document=%s CHUNK chunks=%d", document.document_id, len(chunks))
            if chunks and self.vector_store is not None:
                if skip_embedding:
                    logger.info("[INGEST] document=%s EMBED + QDRANT UPSERT skipped (unchanged content)", document.document_id)
                else:
                    logger.info("[INGEST] document=%s EMBED + QDRANT UPSERT started", document.document_id)
                    self.vector_store.upsert_chunks(chunks)
                    logger.info("[INGEST] document=%s QDRANT UPSERT complete", document.document_id)
            if chunks and self.bm25_retriever is not None:
                self.bm25_retriever.index_chunks(chunks)
                logger.info("[INGEST] document=%s BM25 INDEX complete", document.document_id)
            if self.graph is not None:
                self._populate_graph(document, chunks, content_hash)
                logger.info("[INGEST] document=%s GRAPH UPDATE complete", document.document_id)
            entry.fetch_status = "ingested"
            entry.retrieved_at = payload["retrieved_at"]
            entry.content_hash = content_hash
            entry.source_hash = content_hash
            # Store portable (POSIX, repo-relative) paths so the manifest can
            # be read back correctly regardless of which OS wrote it or which
            # OS/CWD later loads it (see ingestion/paths.py).
            entry.raw_path = to_portable_relpath(raw_path)
            entry.normalized_path = to_portable_relpath(normalized_path)
            entry.chunk_count = len(chunks)
            logger.info("[%s] ingestion complete: %d chunks", document.document_id, len(chunks))
        except Exception as exc:
            entry.fetch_status = "failed"
            entry.error = str(exc)
            logger.error("[%s] ingestion FAILED at document=%s url=%s: %s",
                         document.document_id, document.document_id, document.official_url, exc)
            failed_path = self.root / "failed" / f"{document.document_id}.json"
            failed_path.write_text(entry.model_dump_json(indent=2), encoding="utf-8")
        manifest.entries = [old for old in manifest.entries if old.document_id != entry.document_id or old.content_hash != entry.content_hash]
        manifest.entries.append(entry)
        self._save_manifest(manifest)
        logger.info("[INGEST] document=%s MANIFEST UPDATE complete", entry.document_id)
        logger.info("[INGEST] document=%s %s", entry.document_id,
                    "COMPLETE" if entry.fetch_status == "ingested" else "FAILED")
        return entry

    def _populate_graph(self, document: ConfiguredDocument, chunks: list, content_hash: str) -> None:
        from models.knowledge import KnowledgeDocument, KnowledgeEntity, KnowledgeRelationship
        doc_entity_id = f"document:{document.document_id}"
        document_type = {
            "act": "statute", "statute": "statute", "rule": "rule", "regulation": "regulation",
            "guideline": "guideline", "treaty": "treaty", "order": "order",
        }.get(document.document_type.lower(), "secondary_reference")
        self.graph.add_document(KnowledgeDocument(
            document_id=document.document_id, source_name=document.source_name,
            issuing_authority=document.issuing_authority, official_url=document.official_url,
            document_title=document.document_title, document_type=document_type,
            jurisdiction=document.jurisdiction, category=document.domain, version=document.version,
            publication_date=document.publication_date, effective_date=document.effective_date,
            status=document.status, language=document.language, source_hash=content_hash,
            authoritative=document.authoritative,
        ))
        self.graph.add_entity(KnowledgeEntity(
            entity_id=doc_entity_id, name=document.document_title, entity_type="document",
            metadata={"source_id": document.source_id, "authoritative": document.authoritative,
                      "source_hash": content_hash},
        ))
        if document.document_type.lower() == "rule" and "amendment" in document.document_title.lower():
            target_id = "document:patents-rules-2003"
            self.graph.add_entity(KnowledgeEntity(
                entity_id=target_id, name="Patents Rules, 2003", entity_type="rule_set",
                metadata={"jurisdiction": "India", "domain": "Patents"},
            ))
            relation = "proposes_amendment_to" if document.status == "draft" else "amends"
            self.graph.add_relationship(KnowledgeRelationship(
                relationship_id=f"{document.document_id}:{relation}:patents-rules-2003",
                source_entity_id=doc_entity_id, relationship_type=relation,
                target_entity_id=target_id, source_document_id=document.document_id,
                metadata={"confidence": 1.0, "provenance": "configured_official_document_metadata",
                          "status": document.status},
            ))
        for index, chunk in enumerate(chunks):
            section = chunk.section or chunk.metadata.section
            if not section:
                continue
            section_entity_id = f"{document.document_id}:section:{section}"
            self.graph.add_entity(KnowledgeEntity(entity_id=section_entity_id, name=f"Section {section}",
                                                  entity_type="section", metadata={"document_id": document.document_id}))
            self.graph.add_relationship(KnowledgeRelationship(
                relationship_id=f"{document.document_id}:contains:{section}:{index}",
                source_entity_id=doc_entity_id, relationship_type="contains", target_entity_id=section_entity_id,
                source_document_id=document.document_id, evidence_id=chunk.chunk_id,
                metadata={"source_section": section, "source_chunk_id": chunk.chunk_id,
                          "confidence": 1.0, "provenance": "authoritative_document_structure"},
            ))

    def ingest_configured(self, registry: SourceRegistry) -> list[IngestionManifestEntry]:
        return [self.ingest_document(document) for document in registry.documents()]
