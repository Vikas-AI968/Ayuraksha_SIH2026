"""
Confidence Engine (stage 8).

The LLM's own self-reported confidence is never used. Confidence is
computed backend-side from measurable signals with a fixed, deterministic,
documented formula.
"""
from __future__ import annotations

from typing import Optional

from models.answer import ConfidenceResult
from models.classification import ProductClassificationResult
from models.evidence import EvidencePack
from models.query import QueryAnalysis
from services.router import RoutingDecision
from services.decision_engine import DecisionPlan
from models.knowledge import KnowledgeGraphContext

# Fixed weights for the always-applicable evidence signals.
_W_RETRIEVAL = 0.30
_W_AUTHORITY = 0.20
_W_JURISDICTION = 0.15
_W_CITATION_COVERAGE = 0.25
_W_GRAPH = 0.10
_W_CLASSIFICATION = 0.10



class ConfidenceEngine:
    def compute(
        self,
        analysis: QueryAnalysis,
        routing: RoutingDecision,
        evidence_pack: EvidencePack,
        citation_coverage: float,
        classification: Optional[ProductClassificationResult] = None,
        decision_plan: Optional[DecisionPlan] = None,
        graph_context: Optional[KnowledgeGraphContext] = None,
    ) -> ConfidenceResult:
        retrieval_factor = evidence_pack.average_relevance  # already clipped to [0, 1]

        # Authority: full credit if any non-synthetic (real) source is present;
        # synthetic-only evidence is explicitly discounted so confidence
        # reflects that it is demo/test data, not a real legal authority.
        authority_factor = max((item.source_priority for item in evidence_pack.items), default=0.0)
        if evidence_pack.items and not evidence_pack.synthetic_only:
            authority_factor = round(sum(item.source_priority for item in evidence_pack.items) / len(evidence_pack.items), 4)
        if not evidence_pack.items:
            authority_factor = 0.0

        classification_factor = classification.confidence if classification else None

        jurisdiction_factor = 1.0 if routing.jurisdiction_is_confident else 0.4

        citation_factor = citation_coverage
        graph_factor = 1.0 if graph_context and graph_context.relationships else 0.5
        required_checks = len(decision_plan.decision_path) if decision_plan else 0
        evidence_completeness = min(1.0, len(evidence_pack.items) / max(1, required_checks))

        conflict_penalty = 0.0
        if len(evidence_pack.jurisdictions_present) > 1 and routing.jurisdiction_is_confident:
            # Evidence spans jurisdictions even though the query targeted one.
            conflict_penalty += 0.15

        # Normalize weights over signals that are actually applicable. Product
        # classification is optional; it must never default to a perfect 1.0.
        weighted = (
            _W_RETRIEVAL * retrieval_factor
            + _W_AUTHORITY * authority_factor
            + _W_JURISDICTION * jurisdiction_factor
            + _W_CITATION_COVERAGE * citation_factor
            + _W_GRAPH * graph_factor
        )
        weight_total = _W_RETRIEVAL + _W_AUTHORITY + _W_JURISDICTION + _W_CITATION_COVERAGE + _W_GRAPH
        if classification_factor is not None:
            weighted += _W_CLASSIFICATION * classification_factor
            weight_total += _W_CLASSIFICATION
        score = weighted / weight_total

        # Evidence completeness is a small penalty, not an additive bonus.
        if decision_plan and evidence_completeness < 1.0:
            score -= 0.08 * (1.0 - evidence_completeness)
        score -= conflict_penalty
        if evidence_pack.synthetic_only:
            score = min(score, 0.20)
        # An authoritative but irrelevant document must not generate a high-confidence answer:
        if retrieval_factor < 0.35:
            score = min(score, 0.49)
        elif retrieval_factor < 0.50:
            score = min(score, 0.74)

        # Reserve a little headroom: legal/regulatory answers should not be
        # presented as mathematically certain even with strong evidence.
        score = round(max(0.0, min(0.97, score)), 4)

        if score >= 0.75:
            level = "high"
        elif score >= 0.5:
            level = "medium"
        else:
            level = "low"

        return ConfidenceResult(
            score=score,
            level=level,
            factors={
                "retrieval": round(retrieval_factor, 4),
                "authority": round(authority_factor, 4),
                "classification": round(classification_factor, 4) if classification_factor is not None else None,
                "jurisdiction": round(jurisdiction_factor, 4),
                "citation_coverage": round(citation_factor, 4),
                "graph_provenance": round(graph_factor, 4),
                "evidence_completeness": round(evidence_completeness, 4),
                "conflict_penalty": round(conflict_penalty, 4),
            },
        )
