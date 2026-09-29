"""
Ingestion Pipeline Orchestrator for IP-SAKTI Sahayak.
"""
from typing import Dict, Any, List, Tuple, Union, Optional
import logging
from datetime import datetime, timezone

from ingestion.fetcher import DocumentFetcher
from ingestion.parser import DocumentParser
from ingestion.normalizer import DocumentNormalizer
from ingestion.metadata import MetadataExtractor
from ingestion.chunker import LegalChunker
from ingestion.deduplicator import Deduplicator
from models.document import Document, DocumentMetadata
from models.chunk import Chunk
from ingestion.version_store import DocumentVersionStore

logger = logging.getLogger("ingestion_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class IngestionPipeline:
    """Full independent-stage ingestion pipeline for citation-grounded RAG."""

    def __init__(
        self,
        target_chunk_size: int = 500,
        chunk_overlap: int = 100,
        dedup_strategy: str = "skip",  # 'skip', 'update'
        registry_file: Optional[str] = "data/processed/hash_registry.json"
        ,version_store: Optional[DocumentVersionStore] = None
    ):
        self.fetcher = DocumentFetcher()
        self.parser = DocumentParser()
        self.normalizer = DocumentNormalizer()
        self.metadata_extractor = MetadataExtractor()
        self.chunker = LegalChunker(target_chunk_size=target_chunk_size, chunk_overlap=chunk_overlap)
        self.deduplicator = Deduplicator(registry_file=registry_file)
        self.dedup_strategy = dedup_strategy
        self.version_store = version_store or DocumentVersionStore()

    def process(self, source: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes stages:
        Fetch -> Parse -> Normalize -> Metadata -> Deduplicate -> Chunk
        Returns structured dictionary containing document, chunks, and ingestion summary.
        """
        logs = []
        start_time = datetime.now(timezone.utc)

        try:
            # Stage 1: Fetch / Load
            raw_doc = self.fetcher.fetch(source)
            doc_id = raw_doc["document_id"]
            logs.append(f"FETCH -> SUCCESS (doc_id={doc_id})")

            # Stage 2: Parse
            parsed_doc = self.parser.parse(raw_doc)
            logs.append("PARSE -> SUCCESS")

            # Stage 3: Normalize
            normalized_doc = self.normalizer.normalize(parsed_doc)
            logs.append("NORMALIZE -> SUCCESS")

            # Stage 4: Metadata Extraction & Validation
            doc_metadata = self.metadata_extractor.extract_and_validate(normalized_doc)
            logs.append("METADATA -> SUCCESS")

            related_versions = self.version_store.find_related(doc_metadata.title, doc_metadata.authority)
            known_new_date = doc_metadata.effective_date and doc_metadata.effective_date.lower() not in {"unknown", ""}
            if known_new_date and related_versions and any(
                version.effective_date and version.effective_date.lower() not in {"unknown", ""}
                and version.effective_date > doc_metadata.effective_date
                for version in related_versions
            ):
                doc_metadata.status = "historical"
            self.version_store.record(doc_id, doc_metadata)

            # Check Deduplication
            dup_status = self.deduplicator.check_duplicate(doc_id, doc_metadata.document_hash)
            if dup_status == "exact_duplicate" and self.dedup_strategy == "skip":
                logs.append("DEDUPLICATION -> SKIPPED (exact duplicate found)")
                return {
                    "document_id": doc_id,
                    "status": "skipped",
                    "reason": "exact_duplicate",
                    "chunk_count": 0,
                    "document": None,
                    "chunks": [],
                    "logs": logs,
                    "duration_seconds": (datetime.utcnow() - start_time).total_seconds()
                }

            # Create Document object
            document = Document(
                document_id=doc_id,
                title=doc_metadata.title,
                authority=doc_metadata.authority,
                jurisdiction=doc_metadata.jurisdiction,
                domain=doc_metadata.domain,
                document_type=doc_metadata.document_type,
                effective_date=doc_metadata.effective_date,
                source_url=doc_metadata.source_url,
                content=normalized_doc["content"],
                metadata=doc_metadata
            )

            # Stage 5: Chunking
            chunks = self.chunker.chunk_document(document)
            logs.append(f"CHUNK -> SUCCESS ({len(chunks)} chunks produced)")

            # Register hash
            self.deduplicator.register(
                doc_id=doc_id,
                doc_hash=doc_metadata.document_hash,
                version=doc_metadata.version,
                effective_date=doc_metadata.effective_date
            )

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {
                "document_id": doc_id,
                "status": "success",
                "chunk_count": len(chunks),
                "document": document,
                "chunks": chunks,
                "logs": logs,
                "duration_seconds": duration
            }

        except Exception as e:
            logger.error(f"Ingestion failed for source {source}: {str(e)}", exc_info=True)
            logs.append(f"ERROR -> {str(e)}")
            return {
                "document_id": source if isinstance(source, str) else source.get("document_id", "unknown"),
                "status": "failed",
                "error": str(e),
                "chunk_count": 0,
                "document": None,
                "chunks": [],
                "logs": logs,
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds()
            }
