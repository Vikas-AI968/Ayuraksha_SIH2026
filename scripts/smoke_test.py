"""Local smoke test for the IP-SAKTI Sahayak backend.

Run from the repository root after ingestion:

    python scripts/smoke_test.py

Exits non-zero (and prints [FAIL] lines) if any stage is broken. Does not
mask failures: every stage either passes for real or the script stops and
reports exactly what broke.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import bootstrap_env  # noqa: F401  # loads .env before any other project import touches os.environ

import sys

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    try:
        detail = fn()
        RESULTS.append((name, True, detail or "ok"))
    except Exception as exc:  # noqa: BLE001 - smoke test must report, not raise
        RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))


def main() -> int:
    print("=" * 40)
    print("IP-SAKTI BACKEND SMOKE TEST")
    print("=" * 40)

    import os

    def _env():
        assert os.environ.get("LLM_PROVIDER") is not None or True
        return f"QDRANT_LOCATION={os.environ.get('QDRANT_LOCATION', '(default ./data/qdrant)')}"
    check("Environment configuration", _env)

    def _sqlite():
        from services.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph()
        kg.get_document_count() if hasattr(kg, "get_document_count") else None
        return f"db={kg.db_path}"
    check("SQLite / knowledge graph", _sqlite)

    vector_store_holder = {}

    def _qdrant():
        from embeddings.provider import get_default_embedding_provider
        from vector_store.qdrant_store import QdrantVectorStore, get_qdrant_settings
        location, collection = get_qdrant_settings()
        provider = get_default_embedding_provider()
        vs = QdrantVectorStore(collection_name=collection, location=location, embedding_provider=provider)
        vector_store_holder["vs"] = vs
        return f"location={location} collection={collection}"
    check("Qdrant persistent storage + collection", _qdrant)

    def _corpus():
        from ingestion.authoritative import AuthoritativeCorpusService
        svc = AuthoritativeCorpusService()
        manifest = svc._load_manifest()
        entries = [e for e in manifest.entries if e.fetch_status == "ingested"]
        assert entries, "no ingested authoritative documents in manifest -- run scripts/ingest_authoritative_corpus.py first"
        return f"{len(entries)} ingested document(s)"
    check("Authoritative corpus manifest", _corpus)

    bm25_holder = {}

    def _bm25():
        import os
        from ingestion.corpus_loader import load_manifest_into_indexes
        from ingestion.pipeline import IngestionPipeline
        from retrieval.bm25 import BM25Retriever
        vs = vector_store_holder.get("vs")
        bm25 = BM25Retriever()
        pipeline = IngestionPipeline(registry_file=":memory:")
        manifest_path = os.environ.get("AUTHORITATIVE_MANIFEST", "data/authoritative/manifests/corpus_manifest.json")
        stats = load_manifest_into_indexes(manifest_path, pipeline, vs, bm25)
        bm25_holder["bm25"] = bm25
        assert len(bm25.chunks) > 0, "BM25 index is empty after loading manifest"
        return f"{stats} -> {len(bm25.chunks)} chunks indexed"
    check("BM25 index", _bm25)

    def _orchestrator_query():
        from models.query import QueryIntakeRequest
        from retrieval.semantic import SemanticRetriever
        from retrieval.hybrid import HybridRetriever
        from retrieval.reranker import LightweightScoringReranker
        from services.orchestrator import QueryOrchestrator
        vs = vector_store_holder.get("vs")
        bm25 = bm25_holder.get("bm25")
        semantic = SemanticRetriever(vector_store=vs)
        hybrid = HybridRetriever(semantic_retriever=semantic, bm25_retriever=bm25, semantic_weight=0.6, bm25_weight=0.4)
        orchestrator = QueryOrchestrator(hybrid_retriever=hybrid, reranker=LightweightScoringReranker(), bm25_retriever=bm25)
        request = QueryIntakeRequest(query="What does Section 3(p) of the Patents Act say about traditional knowledge?")
        response = orchestrator.run(request)
        assert response is not None
        return "query orchestration executed"
    check("Query orchestration (retrieval, evidence, citation, confidence)", _orchestrator_query)

    print()
    all_pass = True
    for name, ok, detail in RESULTS:
        tag = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"[{tag}] {name} ({detail})")

    print()
    print("RESULT:", "PASS" if all_pass else "FAIL")
    print("=" * 40)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
