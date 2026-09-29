"""Regression tests over the locally archived official corpus."""
from pathlib import Path

from embeddings.provider import HashEmbeddingProvider
from ingestion.corpus_loader import load_manifest_into_indexes
from ingestion.pipeline import IngestionPipeline
from ingestion.legal_structure import iter_structured_segments
from models.source import CorpusManifest
from retrieval.bm25 import BM25Retriever
from vector_store.qdrant_store import QdrantVectorStore


MANIFEST = Path("data/authoritative/manifests/corpus_manifest.json")


def test_authoritative_manifest_contains_verified_documents():
    manifest = CorpusManifest.model_validate_json(MANIFEST.read_text(encoding="utf-8"))
    assert len(manifest.entries) >= 4
    assert all(entry.authoritative for entry in manifest.entries)
    assert all(entry.fetch_status == "ingested" for entry in manifest.entries)
    assert any(entry.status == "draft" for entry in manifest.entries)
    assert sum(entry.chunk_count for entry in manifest.entries) >= 200


def test_authoritative_patents_act_is_retrievable():
    vector_store = QdrantVectorStore(
        collection_name="test_authoritative_corpus", location=":memory:",
        embedding_provider=HashEmbeddingProvider(),
    )
    bm25 = BM25Retriever()
    result = load_manifest_into_indexes(
        str(MANIFEST), IngestionPipeline(registry_file=":memory:"), vector_store, bm25
    )
    results = bm25.search("Section 3(p) traditional knowledge", top_k=10, authoritative=True)

    assert result["documents"] >= 4
    assert result["chunks"] >= 200
    assert results
    assert any(
        item.document_id == "IPINDIA-PATENTS-ACT-1970"
        and item.metadata.authoritative is True
        and "traditional knowledge" in item.text.lower()
        for item in results
    )


def test_patents_act_section_3p_has_precise_structure_and_page():
    text = Path("data/authoritative/normalized/IPINDIA-PATENTS-ACT-1970-d57084888c69.txt").read_text(encoding="utf-8")
    matches = [(content, context) for content, context in iter_structured_segments(text)
               if "traditional knowledge" in content.lower()]

    assert any(
        context.chapter == "II" and context.section == "3" and context.subsection == "p"
        and context.page_number == 10
        for _, context in matches
    )


def test_current_retrieval_does_not_return_draft_rules():
    bm25 = BM25Retriever()
    result = load_manifest_into_indexes(
        str(MANIFEST), IngestionPipeline(registry_file=":memory:"),
        QdrantVectorStore(collection_name="test_current_filter", location=":memory:", embedding_provider=HashEmbeddingProvider()),
        bm25,
    )
    current = bm25.search("Patent Rules amendment", top_k=30, authoritative=True, status="current")

    assert result["documents"] >= 6
    assert current
    assert all(item.metadata.status == "current" for item in current)
