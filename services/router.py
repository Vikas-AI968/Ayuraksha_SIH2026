"""
Jurisdiction + Domain Router (stage 3.4).

Combines Query Analysis and Product Classification into a routing decision:
which jurisdiction to prioritize, which retrieval filters to apply, and
which authoritative source categories are relevant (advisory, from the
Source Registry). Never silently mixes conflicting jurisdictions -- if the
analyzer marked jurisdiction "ambiguous", the router propagates that rather
than guessing.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from models.query import QueryAnalysis, IPDomain, RegulatoryDomain
from models.classification import ProductClassificationResult
from sources.registry import SourceRegistry, SourceRegistryEntry, get_default_registry

# Maps our internal domain taxonomy onto the corpus/retrieval domain strings
# used as Qdrant/BM25 payload values (see data/synthetic/*.json "domain").
_IP_DOMAIN_TO_CORPUS = {
    IPDomain.PATENT: "Patents",
    IPDomain.TRADEMARK: "Trademarks",
    IPDomain.COPYRIGHT: "Copyright",
    IPDomain.DESIGN: "Designs",
    IPDomain.TRADITIONAL_KNOWLEDGE: "Traditional Knowledge",
    # No corpus domain for trade_secret in the Phase-1 synthetic corpus.
}
_REGULATORY_DOMAIN_TO_CORPUS = {
    RegulatoryDomain.AYURVEDA: "Drug Classification",
    RegulatoryDomain.DRUG: "Drug Classification",
    RegulatoryDomain.FOOD: "Drug Classification",
    RegulatoryDomain.COSMETIC: "Drug Classification",
    RegulatoryDomain.BIODIVERSITY: "Biodiversity/ABS",
    RegulatoryDomain.ABS: "Biodiversity/ABS",
    # No corpus domain for import_export in the Phase-1 synthetic corpus.
}


class RoutingDecision(BaseModel):
    jurisdiction: str
    jurisdiction_is_confident: bool
    retrieval_filters: Dict[str, Optional[str]] = Field(default_factory=dict)
    corpus_domains: List[str] = Field(default_factory=list)
    sources_selected: List[str] = Field(default_factory=list)
    sources_detail: List[Dict[str, Any]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class JurisdictionDomainRouter:
    def __init__(self, registry: Optional[SourceRegistry] = None):
        self.registry = registry or get_default_registry()

    def route(
        self,
        analysis: QueryAnalysis,
        classification: Optional[ProductClassificationResult] = None,
    ) -> RoutingDecision:
        notes: List[str] = []

        jurisdiction = analysis.jurisdiction
        jurisdiction_is_confident = jurisdiction not in ("unknown", "ambiguous")
        if jurisdiction == "ambiguous":
            notes.append(
                f"Query mentions multiple jurisdictions ({', '.join(analysis.jurisdiction_candidates)}); "
                "NOT silently merging them -- routing will not apply a jurisdiction filter."
            )
        elif jurisdiction == "unknown":
            notes.append("No jurisdiction signal detected; defaulting retrieval to unfiltered jurisdiction.")

        corpus_domains: List[str] = []
        for d in analysis.ip_domains:
            mapped = _IP_DOMAIN_TO_CORPUS.get(d)
            if mapped and mapped not in corpus_domains:
                corpus_domains.append(mapped)
        for d in analysis.regulatory_domains:
            mapped = _REGULATORY_DOMAIN_TO_CORPUS.get(d)
            if mapped and mapped not in corpus_domains:
                corpus_domains.append(mapped)

        if classification and classification.applicable_domains:
            for d in classification.applicable_domains:
                if d not in corpus_domains:
                    corpus_domains.append(d)

        # Only apply a hard `domain` retrieval filter when exactly one domain
        # is implicated -- multiple plausible domains should widen recall,
        # not risk filtering out the correct evidence.
        domain_filter = corpus_domains[0] if len(corpus_domains) == 1 else None
        if len(corpus_domains) > 1:
            notes.append(
                f"Multiple domains implicated ({', '.join(corpus_domains)}); "
                "not applying a hard domain filter so reranking can select across all of them."
            )

        jurisdiction_filter = jurisdiction if jurisdiction_is_confident else None

        retrieval_filters = {
            "jurisdiction": jurisdiction_filter,
            "domain": domain_filter,
        }

        sources = self.registry.select(
            jurisdiction=jurisdiction if jurisdiction_is_confident else None,
            domains=corpus_domains,
        )

        return RoutingDecision(
            jurisdiction=jurisdiction,
            jurisdiction_is_confident=jurisdiction_is_confident,
            retrieval_filters=retrieval_filters,
            corpus_domains=corpus_domains,
            sources_selected=[s.source_id for s in sources],
            sources_detail=[s.model_dump() for s in sources],
            notes=notes,
        )
