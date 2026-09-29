"""
Metadata Extractor Module - Stage 4 of Ingestion.
Extracts, validates, and computes document hashes for full traceability.
"""
import hashlib
from typing import Dict, Any
from models.document import DocumentMetadata


class MetadataExtractor:
    """Extracts and validates document metadata, enforcing synthetic markers and content hashes."""

    def extract_and_validate(self, normalized_doc: Dict[str, Any]) -> DocumentMetadata:
        title = normalized_doc.get("title")
        authority = normalized_doc.get("authority")
        jurisdiction = normalized_doc.get("jurisdiction")
        domain = normalized_doc.get("domain")
        document_type = normalized_doc.get("document_type")
        effective_date = normalized_doc.get("effective_date")
        source_url = normalized_doc.get("source_url")
        content = normalized_doc.get("content", "")

        # Validation rules
        missing_fields = []
        if not title: missing_fields.append("title")
        if not authority: missing_fields.append("authority")
        if not jurisdiction: missing_fields.append("jurisdiction")
        if not domain: missing_fields.append("domain")
        if not document_type: missing_fields.append("document_type")
        if not effective_date: missing_fields.append("effective_date")
        if not source_url: missing_fields.append("source_url")
        if not content or not content.strip(): missing_fields.append("content")

        if missing_fields:
            raise ValueError(f"Document validation failed. Missing required fields: {', '.join(missing_fields)}")

        # Enforce source_type designation
        raw_source_type = normalized_doc.get("source_type")
        if not raw_source_type:
            if source_url.startswith("synthetic://") or "Synthetic" in authority:
                source_type = "synthetic"
            else:
                source_type = "official"
        else:
            source_type = raw_source_type

        status = str(normalized_doc.get("status", "current"))
        if status not in {"current", "historical", "draft"}:
            raise ValueError("Document validation failed. status must be current, historical, or draft")
        source_priority = float(normalized_doc.get(
            "source_priority", 1.0 if source_type == "official" else 0.0
        ))
        authority_level = str(normalized_doc.get(
            "authority_level", "development_only" if source_type == "synthetic" else "primary"
        ))
        access_status = str(normalized_doc.get("access_status", "public"))
        authoritative = bool(normalized_doc.get("authoritative", source_type != "synthetic"))

        # Compute SHA256 document hash
        doc_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        metadata = DocumentMetadata(
            title=title,
            source_name=normalized_doc.get("source_name"),
            issuing_authority=normalized_doc.get("issuing_authority", authority),
            authority=authority,
            jurisdiction=jurisdiction,
            domain=domain,
            document_type=document_type,
            effective_date=effective_date,
            source_url=source_url,
            document_url=normalized_doc.get("document_url", source_url),
            source_type=source_type,
            source_id=normalized_doc.get("source_id"),
            authority_level=authority_level,
            access_status=access_status,
            authoritative=authoritative,
            status=status,
            source_priority=max(0.0, min(1.0, source_priority)),
            language=normalized_doc.get("language", "en"),
            original_language=normalized_doc.get("original_language", normalized_doc.get("language", "en")),
            publication_date=normalized_doc.get("publication_date"),
            version=str(normalized_doc.get("version", "1.0")),
            section=normalized_doc.get("section"),
            chapter=normalized_doc.get("chapter"),
            document_hash=doc_hash,
            content_hash=normalized_doc.get("content_hash", doc_hash),
            retrieved_at=normalized_doc.get("retrieved_at"),
            parent_document_id=normalized_doc.get("document_id")
        )
        return metadata
