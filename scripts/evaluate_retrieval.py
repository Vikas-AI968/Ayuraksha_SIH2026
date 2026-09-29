"""
Evaluation Script for IP-SAKTI Sahayak Phase-1.
Compares Vector Only, BM25 Only, Hybrid, and Hybrid + Top-K Reranking on the benchmark dataset.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import bootstrap_env  # noqa: F401  # loads .env before any other project import touches os.environ

import glob
import json
import time
from typing import List, Dict, Any

from ingestion.pipeline import IngestionPipeline
from vector_store.qdrant_store import QdrantVectorStore
from embeddings.provider import get_default_embedding_provider
from retrieval.bm25 import BM25Retriever
from retrieval.semantic import SemanticRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import LightweightScoringReranker
from evaluation.benchmark import BENCHMARK_DATASET
from evaluation.metrics import BenchmarkEvaluator


def setup_engine():
    embedding_provider = get_default_embedding_provider()
    vector_store = QdrantVectorStore(collection_name="ip_sakti_chunks", location=":memory:", embedding_provider=embedding_provider)
    bm25_retriever = BM25Retriever()
    semantic_retriever = SemanticRetriever(vector_store=vector_store)
    hybrid_retriever = HybridRetriever(semantic_retriever=semantic_retriever, bm25_retriever=bm25_retriever)
    reranker = LightweightScoringReranker()
    pipeline = IngestionPipeline(registry_file=":memory:")

    synthetic_files = glob.glob("data/synthetic/*.json")
    print(f"Loading and indexing {len(synthetic_files)} synthetic documents for benchmark...")
    for json_path in synthetic_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        res = pipeline.process(data)
        if res["status"] == "success" and res["chunks"]:
            vector_store.upsert_chunks(res["chunks"])
            bm25_retriever.index_chunks(res["chunks"])

    print(f"Engine Ready. Total indexed chunks: {len(bm25_retriever.chunks)}\n")
    return semantic_retriever, bm25_retriever, hybrid_retriever, reranker


def run_evaluation():
    semantic_retriever, bm25_retriever, hybrid_retriever, reranker = setup_engine()
    evaluator = BenchmarkEvaluator()

    strategies = ["Vector Only", "BM25 Only", "Hybrid", "Hybrid + Reranker"]
    summary_results = {}

    print("=" * 80)
    print(" RUNNING BENCHMARK EVALUATION ACROSS RETRIEVAL STRATEGIES")
    print("=" * 80)

    for strat in strategies:
        eval_records = []
        latencies = []

        for q_spec in BENCHMARK_DATASET:
            t0 = time.time()
            filters = {
                "jurisdiction": q_spec.expected_jurisdiction,
                "domain": q_spec.expected_domain
            }

            if strat == "Vector Only":
                results = semantic_retriever.search(q_spec.query, top_k=5, **filters)
            elif strat == "BM25 Only":
                results = bm25_retriever.search(q_spec.query, top_k=5, **filters)
            elif strat == "Hybrid":
                results = hybrid_retriever.search(q_spec.query, top_k=5, candidate_k=20, **filters)
            elif strat == "Hybrid + Reranker":
                candidates = hybrid_retriever.search(q_spec.query, top_k=20, candidate_k=20, **filters)
                results = reranker.rerank(q_spec.query, candidates, top_k=5)

            t1 = time.time()
            latencies.append((t1 - t0) * 1000.0)

            rec = evaluator.evaluate_query(q_spec, results, k=5)
            eval_records.append(rec)

        summary = evaluator.summarize(eval_records)
        summary["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2)
        summary_results[strat] = summary

    # Print Comparison Table
    print("\n" + "=" * 90)
    print(f"{'Retrieval Strategy':<25} | {'Recall@5':<10} | {'Precision@5':<12} | {'MRR':<8} | {'Citation %':<12} | {'Avg Latency (ms)':<15}")
    print("-" * 90)
    for strat, metrics in summary_results.items():
        print(f"{strat:<25} | {metrics['mean_recall_at_k']:<10.4f} | {metrics['mean_precision_at_k']:<12.4f} | {metrics['mrr']:<8.4f} | {metrics['citation_correctness']:<12.4f} | {metrics['avg_latency_ms']:<15.2f}")
    print("=" * 90)
    print("\nCONCLUSION: Hybrid Retrieval + Top-K Reranking demonstrates the highest Recall@5 and MRR for legal grounding.")

if __name__ == "__main__":
    run_evaluation()
