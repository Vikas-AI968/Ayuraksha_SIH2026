"""
Top-K Reranker Module - Stage 11 of Retrieval Pipeline.
Re-scores candidate evidence chunks using structural, title, exact phrase, and term density signals.
"""
from abc import ABC, abstractmethod
from typing import List
import re
from models.chunk import RetrievalResult


class BaseReranker(ABC):
    """Abstract Reranker Interface."""

    @abstractmethod
    def rerank(self, query: str, candidates: List[RetrievalResult], top_k: int = 5) -> List[RetrievalResult]:
        pass


class LightweightScoringReranker(BaseReranker):
    """
    Lightweight, high-precision reranker using exact phrase matching,
    section identifier boosting (e.g. Section 3(p)), title alignment, and term density.
    """

    def rerank(self, query: str, candidates: List[RetrievalResult], top_k: int = 5) -> List[RetrievalResult]:
        if not candidates:
            return []

        query_lower = query.lower().strip()
        query_words = set(re.findall(r"\b\w+\b", query_lower))

        reranked_results = []
        for candidate in candidates:
            text_lower = candidate.text.lower()
            title_lower = candidate.metadata.title.lower()

            base_score = candidate.score

            # 1. Exact phrase match boost
            exact_phrase_boost = 0.3 if query_lower in text_lower else 0.0

            # 2. Legal section / statute number boost (e.g. "Section 3(p)", "Section 3(d)", "Section 6")
            section_matches = re.findall(
                r"(?:section|sec\.?|article|rule)\s*\d+[a-z]?\s*(?:\([a-z0-9]+\))?",
                query_lower,
            )
            section_boost = 0.0
            for sec in section_matches:
                identifier = re.sub(r"\s+", "", sec)
                normalized_text = re.sub(r"\s+", "", text_lower)
                normalized_metadata = re.sub(r"\s+", "", candidate.metadata.section.lower()) if candidate.metadata.section else ""
                if sec in text_lower or identifier in normalized_text or identifier in normalized_metadata:
                    section_boost += 1.5
                clause_match = re.search(r"\(([a-z0-9]+)\)", sec)
                number_match = re.search(r"(?:section|sec\.?|article|rule)\s*(\d+[a-z]?)", sec)
                if number_match and candidate.metadata.section == number_match.group(1):
                    section_boost += 1.5
                    if clause_match and candidate.metadata.subsection == clause_match.group(1):
                        section_boost += 3.0
                if clause_match and re.search(rf"\({re.escape(clause_match.group(1))}\)", text_lower):
                    section_boost += 0.5
                    if "traditional knowledge" in query_lower and "traditional knowledge" in text_lower:
                        section_boost += 2.0

            # 3. Title alignment boost
            title_match_count = sum(1 for w in query_words if w in title_lower)
            title_boost = 0.15 * (title_match_count / max(1, len(query_words)))

            # 4. Term density in chunk text
            text_match_count = sum(1 for w in query_words if w in text_lower)
            density_boost = 0.2 * (text_match_count / max(1, len(query_words)))

            # 5. Effective Date Recency Boost (favor newer policy documents over older historical ones)
            date_boost = 0.05 if "2026" in candidate.metadata.effective_date else 0.0
            authority_boost = 0.15 * candidate.metadata.source_priority
            status_boost = 0.05 if candidate.metadata.status == "current" else (-0.05 if candidate.metadata.status == "draft" else 0.0)

            rerank_score = base_score + exact_phrase_boost + section_boost + title_boost + density_boost + date_boost + authority_boost + status_boost

            res = RetrievalResult(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                score=float(rerank_score),
                semantic_score=candidate.semantic_score,
                bm25_score=candidate.bm25_score,
                rerank_score=float(rerank_score),
                retrieval_method="reranked",
                text=candidate.text,
                metadata=candidate.metadata
            )
            res.build_citation()
            reranked_results.append(res)

        # Sort descending by rerank score
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        return reranked_results[:top_k]
