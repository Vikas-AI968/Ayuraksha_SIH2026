"""
Evaluation Metrics Calculator for RAG Retrieval Benchmark.
Computes Recall@K, Precision@K, MRR, Citation Correctness, and Retrieval Latency.
"""
from typing import List, Dict, Any
import time
from models.chunk import RetrievalResult
from evaluation.benchmark import BenchmarkQuery


class BenchmarkEvaluator:
    """Evaluates retrieval quality against gold standard benchmark query set."""

    def evaluate_query(self, query_spec: BenchmarkQuery, retrieved_results: List[RetrievalResult], k: int = 5) -> Dict[str, Any]:
        retrieved_k = retrieved_results[:k]
        retrieved_doc_ids = [res.document_id for res in retrieved_k]
        expected_ids = set(query_spec.expected_doc_ids)

        # 1. Recall@K
        found_expected = set(retrieved_doc_ids).intersection(expected_ids)
        recall_at_k = len(found_expected) / len(expected_ids) if expected_ids else 0.0

        # 2. Precision@K
        precision_at_k = len(found_expected) / len(retrieved_k) if retrieved_k else 0.0

        # 3. Reciprocal Rank (MRR component)
        rr = 0.0
        for idx, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in expected_ids:
                rr = 1.0 / (idx + 1)
                break

        # 4. Citation Correctness (All fields present)
        valid_citations = 0
        for res in retrieved_k:
            if res.citation and res.citation.source_url and res.citation.authority and res.citation.effective_date:
                valid_citations += 1
        citation_correctness = valid_citations / len(retrieved_k) if retrieved_k else 0.0

        return {
            "query_id": query_spec.query_id,
            "category": query_spec.category,
            "recall_at_k": recall_at_k,
            "precision_at_k": precision_at_k,
            "reciprocal_rank": rr,
            "citation_correctness": citation_correctness,
            "retrieved_doc_ids": retrieved_doc_ids,
            "expected_doc_ids": list(expected_ids)
        }

    def summarize(self, eval_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not eval_records:
            return {}

        n = len(eval_records)
        mean_recall = sum(r["recall_at_k"] for r in eval_records) / n
        mean_precision = sum(r["precision_at_k"] for r in eval_records) / n
        mrr = sum(r["reciprocal_rank"] for r in eval_records) / n
        mean_citation_correctness = sum(r["citation_correctness"] for r in eval_records) / n

        return {
            "query_count": n,
            "mean_recall_at_k": round(mean_recall, 4),
            "mean_precision_at_k": round(mean_precision, 4),
            "mrr": round(mrr, 4),
            "citation_correctness": round(mean_citation_correctness, 4)
        }
