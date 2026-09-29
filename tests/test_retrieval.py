"""
Unit and Integration Tests for Retrieval Engine & Reranker.
"""
import pytest
from vector_store.qdrant_store import QdrantVectorStore
from embeddings.provider import get_default_embedding_provider
from retrieval.bm25 import BM25Retriever
from retrieval.semantic import SemanticRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import LightweightScoringReranker
from ingestion.pipeline import IngestionPipeline


@pytest.fixture
def indexed_engine():
    provider = get_default_embedding_provider()
    v_store = QdrantVectorStore(collection_name="test_chunks", location=":memory:", embedding_provider=provider)
    bm25 = BM25Retriever()
    pipeline = IngestionPipeline(registry_file=":memory:")

    sample_doc_1 = {
        "document_id": "TEST-PAT-001",
        "title": "Guidelines on Section 3(p) Traditional Knowledge",
        "authority": "Synthetic Patent Office",
        "jurisdiction": "India",
        "domain": "Patents",
        "document_type": "Guidance",
        "effective_date": "2026-01-01",
        "source_url": "synthetic://test-pat-001",
        "content": "Section 3(p) under Indian Patents Act states that an invention which is traditional knowledge is not patentable."
    }

    sample_doc_2 = {
        "document_id": "TEST-INT-001",
        "title": "PCT Patent Application Procedures",
        "authority": "WIPO Secretariat",
        "jurisdiction": "International",
        "domain": "Patents",
        "document_type": "Treaty",
        "effective_date": "2024-05-01",
        "source_url": "synthetic://test-int-001",
        "content": "Rule 1. An applicant seeking PCT international patent protection files a single PCT application."
    }

    res1 = pipeline.process(sample_doc_1)
    res2 = pipeline.process(sample_doc_2)

    v_store.upsert_chunks(res1["chunks"] + res2["chunks"])
    bm25.index_chunks(res1["chunks"] + res2["chunks"])

    sem = SemanticRetriever(v_store)
    hyb = HybridRetriever(sem, bm25)
    reranker = LightweightScoringReranker()

    return sem, bm25, hyb, reranker


def test_bm25_exact_keyword_retrieval(indexed_engine):
    sem, bm25, hyb, reranker = indexed_engine
    results = bm25.search("Section 3(p)", top_k=5)
    assert len(results) > 0
    assert "Section 3(p)" in results[0].text or "Section 3(p)" in results[0].metadata.title


def test_hybrid_merging_and_reranking(indexed_engine):
    sem, bm25, hyb, reranker = indexed_engine
    candidates = hyb.search("PCT international patent filing route", top_k=5)
    assert len(candidates) > 0
    reranked = reranker.rerank("PCT international patent filing route", candidates, top_k=2)
    assert len(reranked) > 0
    assert reranked[0].metadata.source_url == "synthetic://test-int-001"
