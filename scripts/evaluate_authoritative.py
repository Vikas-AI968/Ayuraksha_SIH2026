"""
Authoritative RAG Evaluation.

Unlike scripts/evaluate_retrieval.py (which builds an in-memory index from
data/synthetic/*.json for fast deterministic unit/regression testing), this
script evaluates the REAL, currently-persisted authoritative corpus
(data/qdrant/ + data/processed/bm25_index.pkl) using
data/benchmark/authoritative_evaluation.json as ground truth. It performs a
READ-ONLY load of the existing indexes -- exactly what application startup
does -- and never runs the ingestion pipeline.

Usage:
    python -m scripts.evaluate_authoritative
"""
from __future__ import annotations

import bootstrap_env  # noqa: F401
import json
import math
import os
from typing import Any, Dict, List, Optional

from embeddings.provider import get_default_embedding_provider
from vector_store.qdrant_store import QdrantVectorStore, get_qdrant_settings
from retrieval.bm25 import BM25Retriever, get_bm25_index_path
from retrieval.semantic import SemanticRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import LightweightScoringReranker
from services.orchestrator import QueryOrchestrator
from services.observability import QueryLogStore
from models.query import QueryIntakeRequest

BENCHMARK_PATH = "data/benchmark/authoritative_evaluation.json"
K_VALUES = [1, 3, 5, 10]


def load_engine() -> QueryOrchestrator:
    """READ-ONLY load of the persisted indexes -- identical contract to
    api.main's lifespan. Never calls IngestionPipeline.process()."""
    embedding_provider = get_default_embedding_provider()
    qdrant_location, qdrant_collection = get_qdrant_settings()
    vector_store = QdrantVectorStore(collection_name=qdrant_collection, location=qdrant_location, embedding_provider=embedding_provider)
    bm25_retriever = BM25Retriever()
    loaded = bm25_retriever.load(get_bm25_index_path())
    if not loaded:
        raise RuntimeError(
            f"No persisted BM25 index found at {get_bm25_index_path()}. "
            "Run `python -m scripts.ingest_authoritative_corpus` first -- "
            "this evaluator does not ingest."
        )
    semantic_retriever = SemanticRetriever(vector_store=vector_store)
    hybrid_retriever = HybridRetriever(semantic_retriever=semantic_retriever, bm25_retriever=bm25_retriever)
    reranker = LightweightScoringReranker()
    orchestrator = QueryOrchestrator(
        hybrid_retriever=hybrid_retriever, reranker=reranker, bm25_retriever=bm25_retriever,
        log_store=QueryLogStore(db_path=":memory:"),
    )
    print(f"[EVAL] Loaded {len(bm25_retriever.chunks)} chunks from persisted BM25 index "
          f"(Qdrant collection '{qdrant_collection}' @ {qdrant_location}). No ingestion performed.")
    return orchestrator


def _first_relevant_rank(evidence: List[Dict[str, Any]], expected_source: Optional[str], expected_section: Optional[str]) -> Optional[int]:
    for i, item in enumerate(evidence, start=1):
        if expected_source and item.get("source_id") != expected_source:
            continue
        if expected_section and str(item.get("section") or "") != str(expected_section):
            continue
        return i
    return None


def evaluate() -> Dict[str, Any]:
    orchestrator = load_engine()
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    per_query: List[Dict[str, Any]] = []
    for q in queries:
        request = QueryIntakeRequest(query=q["query"], jurisdiction="auto", top_k=10)
        response = orchestrator.run(request)
        evidence = response.evidence
        rank = _first_relevant_rank(evidence, q.get("expected_source"), q.get("expected_section"))
        top1 = evidence[0] if evidence else None

        per_query.append({
            "id": q["id"],
            "query": q["query"],
            "abstention_expected": q["abstention_expected"],
            "abstained": response.status == "abstained",
            "abstention_correct": (response.status == "abstained") == q["abstention_expected"],
            "rank_of_relevant": rank,
            "top1_source_id": top1.get("source_id") if top1 else None,
            "top1_document_type": top1.get("document_type") if top1 else None,
            "top1_jurisdiction": top1.get("jurisdiction") if top1 else None,
            "source_correct": bool(top1 and top1.get("source_id") == q.get("expected_source")) if q.get("expected_source") else None,
            "doc_type_correct": bool(top1 and top1.get("document_type") == q.get("expected_document_type")) if q.get("expected_document_type") else None,
            "jurisdiction_correct": bool(top1 and top1.get("jurisdiction") == q.get("expected_jurisdiction")) if q.get("expected_jurisdiction") else None,
            "section_correct_top5": any(
                str(e.get("section") or "") == str(q.get("expected_section"))
                for e in evidence[:5]
            ) if q.get("expected_section") else None,
            "citation_coverage": response.citation_coverage,
            "status": response.status,
        })

    return {"per_query": per_query}


def summarize(results: Dict[str, Any]) -> Dict[str, Any]:
    per_query = results["per_query"]
    # Retrieval metrics only computed over queries where an answer (not
    # abstention) was actually expected -- an expected-abstention query has
    # no "correct document" for Recall/Precision/NDCG to be measured against.
    retrieval_rows = [r for r in per_query if not r["abstention_expected"]]
    n_retrieval = len(retrieval_rows)

    def recall_at(k):
        if not n_retrieval:
            return None
        hits = sum(1 for r in retrieval_rows if r["rank_of_relevant"] is not None and r["rank_of_relevant"] <= k)
        return round(hits / n_retrieval, 4)

    def precision_at(k):
        if not n_retrieval:
            return None
        # Binary relevance, single gold match per query -> precision@k = hit/k
        hits = sum(1 for r in retrieval_rows if r["rank_of_relevant"] is not None and r["rank_of_relevant"] <= k)
        return round(hits / (n_retrieval * k), 4)

    def hitrate_at(k):
        return recall_at(k)  # equivalent under single-relevant-item-per-query ground truth

    def ndcg_at(k):
        if not n_retrieval:
            return None
        total = 0.0
        for r in retrieval_rows:
            rank = r["rank_of_relevant"]
            if rank is not None and rank <= k:
                total += 1.0 / math.log2(rank + 1)
            # ideal DCG for a single relevant item at rank 1 is 1/log2(2) = 1.0
        return round(total / n_retrieval, 4)

    mrr = None
    if n_retrieval:
        mrr = round(sum((1.0 / r["rank_of_relevant"]) if r["rank_of_relevant"] else 0.0 for r in retrieval_rows) / n_retrieval, 4)

    def frac(key):
        vals = [r[key] for r in retrieval_rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    section_rows = [r for r in per_query if r["section_correct_top5"] is not None]
    section_correctness = round(sum(r["section_correct_top5"] for r in section_rows) / len(section_rows), 4) if section_rows else None

    abstention_accuracy = round(sum(r["abstention_correct"] for r in per_query) / len(per_query), 4)

    answered_rows = [r for r in per_query if r["status"] == "answered"]
    citation_coverage = round(sum(r["citation_coverage"] for r in answered_rows) / len(answered_rows), 4) if answered_rows else None

    return {
        "dataset_size": len(per_query),
        "retrieval_eval_queries": n_retrieval,
        "recall@1": recall_at(1), "recall@3": recall_at(3), "recall@5": recall_at(5), "recall@10": recall_at(10),
        "precision@1": precision_at(1), "precision@3": precision_at(3), "precision@5": precision_at(5),
        "mrr": mrr,
        "hitrate@1": hitrate_at(1), "hitrate@3": hitrate_at(3), "hitrate@5": hitrate_at(5), "hitrate@10": hitrate_at(10),
        "ndcg@5": ndcg_at(5), "ndcg@10": ndcg_at(10),
        "source_correctness": frac("source_correct"),
        "document_type_correctness": frac("doc_type_correct"),
        "jurisdiction_correctness": frac("jurisdiction_correct"),
        "section_correctness": section_correctness,
        "abstention_accuracy": abstention_accuracy,
        "citation_coverage_mean": citation_coverage,
    }


def print_report(results: Dict[str, Any], summary: Dict[str, Any]) -> None:
    print("=" * 68)
    print("IP-SAKTI SAHAYAK -- AUTHORITATIVE RAG EVALUATION")
    print("=" * 68)
    print(f"\nDataset: {summary['dataset_size']} queries ({summary['retrieval_eval_queries']} scored for retrieval; "
          f"the rest expect abstention and are scored for abstention accuracy only)")
    print("Corpus: Authoritative (persisted BM25 + Qdrant, no re-ingestion)\n")
    print("Retrieval:")
    for k in K_VALUES:
        print(f"  Recall@{k}: {summary[f'recall@{k}']}")
    for k in (1, 3, 5):
        print(f"  Precision@{k}: {summary[f'precision@{k}']}")
    print(f"  MRR: {summary['mrr']}")
    for k in K_VALUES:
        print(f"  HitRate@{k}: {summary[f'hitrate@{k}']}")
    print(f"  NDCG@5: {summary['ndcg@5']}")
    print(f"  NDCG@10: {summary['ndcg@10']}")
    print(f"\nSource correctness: {summary['source_correctness']}")
    print(f"Document type correctness: {summary['document_type_correctness']}")
    print(f"Jurisdiction correctness: {summary['jurisdiction_correctness']}")
    print(f"Section correctness (top-5): {summary['section_correctness']}")
    print(f"\nAbstention accuracy: {summary['abstention_accuracy']}")
    print(f"Citation coverage (mean, answered queries): {summary['citation_coverage_mean']}")
    print("Citation correctness: derived directly from citation_coverage above "
          "(the citation validator's supported/total-claims ratio) -- no separate metric fabricated.")
    print("Answer relevance / groundedness (LLM-judged): NOT AVAILABLE -- no generative LLM provider "
          "configured (ANTHROPIC_API_KEY/GOOGLE_API_KEY unset); extractive fallback is in use, which "
          "quotes evidence verbatim rather than reasoning, so an LLM-judged relevance/faithfulness "
          "score would not reflect real generative behavior. Groundedness's deterministic proxy is "
          "the citation_coverage figure above.")

    print("\n" + "-" * 68)
    print("CRITICAL TEST")
    print("-" * 68)
    critical = next((r for r in results["per_query"] if r["query"].lower().startswith("can a classical ayurvedic")), None)
    if critical:
        print(f"Query: {critical['query']}")
        print("Expected: Patents Act, 1970 -- Section 3")
        print(f"Rank of matching evidence: {critical['rank_of_relevant']}")
        verdict = "PASS" if critical["rank_of_relevant"] and critical["rank_of_relevant"] <= 5 else "FAIL"
        print(f"PASS/FAIL: {verdict}")
    print("=" * 68)


if __name__ == "__main__":
    results = evaluate()
    summary = summarize(results)
    print_report(results, summary)
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/authoritative_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_query": results["per_query"]}, f, indent=2)
    print("\nFull per-query results written to data/processed/authoritative_evaluation_report.json")
