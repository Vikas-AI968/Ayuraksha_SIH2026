"""
Citation Traceability Service.
Builds verifiable claim-to-evidence citation chains for LLM grounding.
"""
from typing import Dict, Any, List
from models.chunk import RetrievalResult, CitationTrace


class CitationTraceabilityService:
    """Formats retrieved evidence chunks into structured citations for LLM answer generation."""

    def generate_grounding_context(self, query: str, results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Formats evidence items with explicit trace paths:
        Claim / Query -> Evidence Chunk ID -> Source Title -> Authority -> Jurisdiction -> Source URL
        """
        evidence_items = []
        for idx, res in enumerate(results):
            citation = res.citation or res.build_citation()
            evidence_items.append({
                "evidence_index": idx + 1,
                "chunk_id": res.chunk_id,
                "document_id": res.document_id,
                "score": round(res.score, 4),
                "retrieval_method": res.retrieval_method,
                "source_title": citation.title,
                "authority": citation.authority,
                "jurisdiction": citation.jurisdiction,
                "domain": citation.domain,
                "document_type": citation.document_type,
                "effective_date": citation.effective_date,
                "source_url": citation.source_url,
                "source_type": citation.source_type,
                "citation_text": citation.citation_text,
                "extracted_text": res.text
            })

        return {
            "query": query,
            "evidence_count": len(evidence_items),
            "evidence_chain": evidence_items
        }
