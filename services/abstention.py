"""
Safe Abstention (stage 9). Abstention is a FEATURE, not a failure.

Evaluated in two passes by the orchestrator:
  1. Pre-reasoning: can we even justify asking the LLM to answer? (saves a
     call and guarantees we never generate an answer from near-zero evidence)
  2. Post-reasoning: after citation validation, is coverage too low to trust
     the generated answer?
"""
from __future__ import annotations

from typing import Optional

from models.answer import AbstentionDecision
from models.classification import ProductClass, ProductClassificationResult
from models.evidence import EvidencePack
from models.query import QueryAnalysis, PRODUCT_SENSITIVE_INTENTS, JURISDICTION_SENSITIVE_INTENTS
from services.router import RoutingDecision

RETRIEVAL_QUALITY_THRESHOLD = 0.20
CITATION_COVERAGE_THRESHOLD = 0.50
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.35


class AbstentionEngine:
    def check_pre_reasoning(
        self,
        analysis: QueryAnalysis,
        routing: RoutingDecision,
        evidence_pack: EvidencePack,
        classification: Optional[ProductClassificationResult] = None,
    ) -> AbstentionDecision:
        missing: list = []

        if not evidence_pack.items:
            return AbstentionDecision(
                should_abstain=True,
                reason="No supporting evidence was retrieved for this query.",
                missing_information=["Relevant authoritative or synthetic evidence for this query."],
                recommended_next_step="Rephrase the query with more specific terms (e.g. the relevant Act, section, or product).",
            )

        sensitive_query = (
            analysis.intent in JURISDICTION_SENSITIVE_INTENTS
            or bool(analysis.ip_domains)
            or bool(analysis.regulatory_domains)
            or analysis.ayurveda_relevance
        )
        if evidence_pack.synthetic_only and sensitive_query:
            return AbstentionDecision(
                should_abstain=True,
                reason="Only synthetic development evidence is available for this jurisdiction-sensitive query.",
                missing_information=["A retrieved current authoritative source document and supporting passage."],
                recommended_next_step="Ingest or provide the applicable official Act, rule, regulation, or guidance before relying on an answer.",
            )

        if evidence_pack.top_relevance < RETRIEVAL_QUALITY_THRESHOLD:
            return AbstentionDecision(
                should_abstain=True,
                reason=f"Retrieval quality is below the safety threshold (top relevance "
                       f"{evidence_pack.top_relevance:.2f} < {RETRIEVAL_QUALITY_THRESHOLD}).",
                missing_information=["Evidence more directly relevant to the query."],
                recommended_next_step="Rephrase the query or narrow it to a specific Act, section, or product category.",
            )

        if analysis.jurisdiction == "ambiguous":
            missing.append("A single, unambiguous jurisdiction.")
            return AbstentionDecision(
                should_abstain=True,
                reason=f"The query references multiple jurisdictions ({', '.join(analysis.jurisdiction_candidates)}); "
                       "answering would require silently mixing jurisdictions, which this system will not do.",
                missing_information=missing,
                recommended_next_step="Ask about one jurisdiction at a time (e.g. 'in India' or 'for the EU market').",
            )

        if classification and classification.classification == ProductClass.UNKNOWN \
                and analysis.intent in PRODUCT_SENSITIVE_INTENTS \
                and (classification.confidence < CLASSIFICATION_CONFIDENCE_THRESHOLD):
            return AbstentionDecision(
                should_abstain=True,
                reason="Product classification is too uncertain for a regulatory-classification/compliance question.",
                missing_information=classification.missing_information or ["Ingredients, dosage form, and marketing claims for the product."],
                recommended_next_step="Provide the product's ingredients, dosage form, intended use, and any label claims.",
            )

        if len(evidence_pack.jurisdictions_present) > 1 and routing.jurisdiction_is_confident:
            return AbstentionDecision(
                should_abstain=True,
                reason=f"Retrieved evidence spans conflicting jurisdictions "
                       f"({', '.join(evidence_pack.jurisdictions_present)}) for a single-jurisdiction query.",
                missing_information=["Jurisdiction-consistent evidence."],
                recommended_next_step="Narrow the query to the specific jurisdiction of interest.",
            )

        return AbstentionDecision(should_abstain=False)

    def check_post_reasoning(self, citation_coverage: float) -> AbstentionDecision:
        if citation_coverage < CITATION_COVERAGE_THRESHOLD:
            return AbstentionDecision(
                should_abstain=True,
                reason=f"Citation coverage ({citation_coverage:.2f}) is below the safety threshold "
                       f"({CITATION_COVERAGE_THRESHOLD}); too many claims in the generated answer "
                       "could not be verified against retrieved evidence.",
                missing_information=["Evidence that directly supports the generated claims."],
                recommended_next_step="Rephrase the query, or treat the answer as preliminary and consult a professional.",
            )
        return AbstentionDecision(should_abstain=False)
