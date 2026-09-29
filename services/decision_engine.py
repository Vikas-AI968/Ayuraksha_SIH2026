"""Deterministic regulatory decision planning; it does not decide legal outcomes."""
from __future__ import annotations

from pydantic import BaseModel, Field

from models.classification import ProductClass, ProductClassificationResult
from models.query import QueryAnalysis
from services.router import RoutingDecision


class DecisionStep(BaseModel):
    step: str
    purpose: str
    required_evidence: list[str] = Field(default_factory=list)
    status: str = "check"


class DecisionPlan(BaseModel):
    applicable_concepts: list[str] = Field(default_factory=list)
    decision_path: list[DecisionStep] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RegulatoryDecisionEngine:
    """Maps observed signals to evidence checks, never to hardcoded legal conclusions."""

    def evaluate(self, analysis: QueryAnalysis, routing: RoutingDecision,
                 classification: ProductClassificationResult | None = None) -> DecisionPlan:
        concepts = [domain.value for domain in analysis.ip_domains + analysis.regulatory_domains]
        steps = [DecisionStep(step="jurisdiction", purpose="Confirm the governing jurisdiction.",
                              required_evidence=["jurisdiction", "source_status"])]
        missing = []
        warnings = []
        if not routing.jurisdiction_is_confident:
            missing.append("A single, unambiguous jurisdiction.")
        if classification:
            concepts.append(classification.classification.value)
            steps.insert(0, DecisionStep(step="product_classification", purpose="Classify the product from supplied facts.",
                                         required_evidence=["ingredients", "dosage_form", "intended_use", "claims"],
                                         status="uncertain" if classification.confidence < 0.6 else "check"))
            missing.extend(classification.missing_information)
            if classification.classification in {
                ProductClass.CLASSICAL_MEDICINE, ProductClass.PROPRIETARY_MEDICINE,
                ProductClass.NEW_DRUG,
            } or analysis.traditional_knowledge_relevance:
                concepts.append("traditional_knowledge")
                steps.append(DecisionStep(step="traditional_knowledge", purpose="Check whether the subject overlaps codified or traditional knowledge.",
                                          required_evidence=["authoritative_TK_source", "relevant_section_or_record"]))
            if classification.classification == ProductClass.PHYTOPHARMACEUTICAL \
                    or analysis.biological_resource_relevance:
                concepts.append("biological_resource")
                steps.append(DecisionStep(step="biological_resource", purpose="Check source/origin and access-and-benefit-sharing requirements.",
                                          required_evidence=["source_of_biological_material", "ABS_or_NBA_material"]))
        if "Patents" in routing.corpus_domains or any("patent" in c.lower() for c in concepts):
            steps.append(DecisionStep(step="patentability_provisions", purpose="Retrieve applicable current patent provisions and rules.",
                                      required_evidence=["current_statute_text", "current_rules", "section_identifier", "prior_art"]))
        if "Drug Classification" in routing.corpus_domains:
            steps.append(DecisionStep(step="product_regulatory_path", purpose="Retrieve the applicable AYUSH, drug, food, or cosmetic regulatory material.",
                                      required_evidence=["current_regulatory_text", "product_facts"]))
        if "Biodiversity/ABS" in routing.corpus_domains:
            steps.append(DecisionStep(step="ABS_review", purpose="Check biological-resource access and benefit-sharing material separately from patentability.",
                                      required_evidence=["current_ABS_text", "source_or_origin_facts"]))
        if not routing.sources_selected:
            warnings.append("No source categories were selected by the registry.")
        return DecisionPlan(applicable_concepts=sorted(set(concepts)), decision_path=steps,
                            missing_information=sorted(set(missing)), warnings=warnings)
