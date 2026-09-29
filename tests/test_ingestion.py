"""
Unit and Integration Tests for Ingestion Pipeline.
"""
import pytest
from ingestion.pipeline import IngestionPipeline
from ingestion.validator import DocumentValidator
from models.document import DocumentMetadata


def test_document_validation():
    validator = DocumentValidator()
    valid_doc = {
        "title": "Test Title",
        "authority": "Test Authority",
        "jurisdiction": "India",
        "domain": "Patents",
        "document_type": "Guidance",
        "effective_date": "2026-01-01",
        "source_url": "synthetic://test-doc-001",
        "content": "This is valid test content for ingestion validation."
    }
    is_valid, errors = validator.validate(valid_doc)
    assert is_valid is True
    assert len(errors) == 0


def test_ingestion_pipeline_execution():
    pipeline = IngestionPipeline(registry_file=":memory:")
    sample_doc = {
        "document_id": "TEST-ING-001",
        "title": "Sample Patent Guidance on Section 3(p)",
        "authority": "IP-SAKTI Synthetic Patent Office",
        "jurisdiction": "India",
        "domain": "Patents",
        "document_type": "Guidance",
        "effective_date": "2026-01-01",
        "source_url": "synthetic://test-patent-ingest-001",
        "source_type": "synthetic",
        "content": """CHAPTER I: INTELLECTUAL PROPERTY
Section 1. Purpose
These guidelines outline traditional knowledge protection.

Section 2. Section 3(p) Rules
Under Section 3(p), traditional knowledge aggregations are not patentable."""
    }

    res = pipeline.process(sample_doc)
    assert res["status"] == "success"
    assert res["chunk_count"] > 0
    assert res["document"].metadata.source_type == "synthetic"
    assert res["document"].metadata.document_hash is not None
