"""
Hybrid Search Merger Module - Combines Vector & BM25 search.
"""
from typing import List, Dict, Any, Optional
import numpy as np

from retrieval.semantic import SemanticRetriever
from retrieval.bm25 import BM25Retriever
from models.chunk import RetrievalResult


class HybridRetriever:
    """Merges semantic vector search and BM25 lexical search using weighted score normalization."""

    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        bm25_retriever: BM25Retriever,
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4
    ):
        self.semantic_retriever = semantic_retriever
        self.bm25_retriever = bm25_retriever
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 20,
        jurisdiction: Optional[str] = None,
        domain: Optional[str] = None,
        authority: Optional[str] = None,
        document_type: Optional[str] = None,
        effective_date_from: Optional[str] = None,
        status: Optional[str] = None,
        source_priority_min: Optional[float] = None,
        authoritative: Optional[bool] = None
    ) -> List[RetrievalResult]:
        """
        Retrieves Top-N candidates from Semantic and BM25, normalizes scores,
        computes weighted hybrid score, and returns sorted candidate set.
        """
        filters = {
            "jurisdiction": jurisdiction,
            "domain": domain,
            "authority": authority,
            "document_type": document_type,
            "effective_date_from": effective_date_from
            ,"status": status,
            "source_priority_min": source_priority_min,
            "authoritative": authoritative
        }

        # 1. Fetch candidates from both retrievers
        semantic_results = self.semantic_retriever.search(query, top_k=candidate_k, **filters)
        bm25_results = self.bm25_retriever.search(query, top_k=candidate_k, **filters)

        # Map by chunk_id
        chunk_map: Dict[str, RetrievalResult] = {}
        semantic_scores: Dict[str, float] = {}
        bm25_scores: Dict[str, float] = {}

        for res in semantic_results:
            chunk_map[res.chunk_id] = res
            semantic_scores[res.chunk_id] = res.score

        for res in bm25_results:
            if res.chunk_id not in chunk_map:
                chunk_map[res.chunk_id] = res
            bm25_scores[res.chunk_id] = res.score

        if not chunk_map:
            return []

        # 2. Normalize scores (Min-Max)
        def _normalize(score_dict: Dict[str, float]) -> Dict[str, float]:
            if not score_dict:
                return {}
            vals = list(score_dict.values())
            min_v, max_v = min(vals), max(vals)
            if max_v == min_v:
                return {k: 1.0 for k in score_dict}
            return {k: (v - min_v) / (max_v - min_v) for k, v in score_dict.items()}

        norm_semantic = _normalize(semantic_scores)
        norm_bm25 = _normalize(bm25_scores)

        # 3. Combine with weights
        hybrid_results: List[RetrievalResult] = []
        for chunk_id, base_res in chunk_map.items():
            s_score = norm_semantic.get(chunk_id, 0.0)
            b_score = norm_bm25.get(chunk_id, 0.0)

            hybrid_score = (self.semantic_weight * s_score) + (self.bm25_weight * b_score)

            merged_res = RetrievalResult(
                chunk_id=base_res.chunk_id,
                document_id=base_res.document_id,
                score=float(hybrid_score),
                semantic_score=semantic_scores.get(chunk_id),
                bm25_score=bm25_scores.get(chunk_id),
                retrieval_method="hybrid",
                text=base_res.text,
                metadata=base_res.metadata
            )
            merged_res.build_citation()
            hybrid_results.append(merged_res)

        # 4. Sort descending
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        return hybrid_results[:top_k]
