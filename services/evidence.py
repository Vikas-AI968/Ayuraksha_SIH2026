"""
Evidence Pack builder (stage 5).

Normalizes raw RetrievalResult objects from the existing Phase-1 retrieval
pipeline into the EvidencePack the LLM Reasoning Engine is allowed to see.
The LLM never receives raw unrestricted retrieval output -- only this
normalized, citation-carrying structure.
"""
from __future__ import annotations

from typing import List

from models.chunk import RetrievalResult
from models.evidence import EvidenceItem, EvidencePack


class EvidencePackBuilder:
    def build(self, query: str, results: List[RetrievalResult]) -> EvidencePack:
        items: List[EvidenceItem] = []
        for idx, res in enumerate(results, start=1):
            citation = res.citation or res.build_citation()
            evidence_id = f"E{idx:03d}"
            # Reranked/hybrid scores can exceed 1.0 due to additive boosts;
            # clip to a [0, 1] relevance scale for downstream consumers.
            relevance = max(0.0, min(1.0, res.score))
            items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    document_id=res.document_id,
                    chunk_id=res.chunk_id,
                    source_id=res.metadata.source_id,
                    official_url=res.metadata.source_url,
                    authority=res.metadata.authority,
                    jurisdiction=res.metadata.jurisdiction,
                    domain=res.metadata.domain,
                    document_type=res.metadata.document_type,
                    title=res.metadata.title,
                    section=res.metadata.section,
                    subsection=res.metadata.subsection,
                    rule=res.metadata.rule,
                    sub_rule=res.metadata.sub_rule,
                    regulation=res.metadata.regulation,
                    sub_regulation=res.metadata.sub_regulation,
                    article=res.metadata.article,
                    schedule=res.metadata.schedule,
                    page=res.metadata.page_number,
                    text=res.text,
                    relevance_score=round(relevance, 4),
                    rerank_score=res.rerank_score,
                    authority_score=round(res.metadata.source_priority, 4),
                    source_priority=round(res.metadata.source_priority, 4),
                    authority_level=res.metadata.authority_level,
                    status=res.metadata.status,
                    provenance="synthetic_development_corpus" if res.metadata.source_type == "synthetic" else "indexed_source_chunk",
                    retrieval_method=res.retrieval_method,
                    citation=citation.citation_text,
                    source_url=res.metadata.source_url,
                    source_type=res.metadata.source_type,
                    access_status=res.metadata.access_status,
                    effective_date=res.metadata.effective_date,
                    version=res.metadata.version,
                )
            )

        jurisdictions_present = sorted({i.jurisdiction for i in items})
        domains_present = sorted({res.metadata.domain for res in results})
        synthetic_only = bool(items) and all(i.source_type == "synthetic" for i in items)
        avg_relevance = round(sum(i.relevance_score for i in items) / len(items), 4) if items else 0.0
        top_relevance = round(max((i.relevance_score for i in items), default=0.0), 4)

        return EvidencePack(
            query=query,
            items=items,
            synthetic_only=synthetic_only,
            jurisdictions_present=jurisdictions_present,
            domains_present=domains_present,
            average_relevance=avg_relevance,
            top_relevance=top_relevance,
        )
