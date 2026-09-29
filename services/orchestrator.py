"""
End-to-end orchestrator for POST /api/v1/query.

USER QUERY -> Query Intake -> Query Analysis -> Product Classification ->
Jurisdiction/Domain Routing -> (existing) Retrieval + Reranking ->
Evidence Pack -> [pre-reasoning Abstention check] -> LLM Reasoning ->
Citation Validation -> [post-reasoning Abstention check] -> Confidence ->
Structured Response.

This module is the "wiring": each stage is implemented by a dedicated
service so this file stays orchestration-only, and every new component is
actually invoked here rather than left as an unused stub (RULE 15).
"""
from __future__ import annotations

from typing import Optional
import os

from models.answer import QueryResponse
from models.chunk import RetrievalResult
from models.classification import ProductClassificationRequest, ProductClassificationResult
from models.query import QueryIntakeRequest, QueryIntent, RegulatoryDomain

from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import BaseReranker

from services.abstention import AbstentionEngine
from services.citation_validator import CitationValidator, compute_metrics
from services.classifier import ProductClassifier
from services.confidence import ConfidenceEngine
from services.evidence import EvidencePackBuilder
from services.llm_provider import BaseLLMProvider
from services.observability import QueryLogStore, QueryTrace, get_default_query_log_store
from services.query_analyzer import QueryAnalyzer, QueryIntakeService
from services.query_understanding import QueryUnderstandingService
from services.reasoning import ReasoningEngine
from services.router import JurisdictionDomainRouter
from services.decision_engine import DecisionPlan, RegulatoryDecisionEngine
from services.knowledge_graph import KnowledgeGraph
from retrieval.query_expansion import build_search_signals, extract_provisions, parse_provision

DISCLAIMER = (
    "IP-SAKTI Sahayak is an information/research assistant, not a lawyer. It does not provide "
    "legal advice. Verify any answer against the primary authoritative source before relying on it, "
    "and consult a qualified IP attorney or regulatory professional for binding decisions."
)


class QueryOrchestrator:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: BaseReranker,
        bm25_retriever: Optional[BM25Retriever] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        log_store: Optional[QueryLogStore] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.bm25_retriever = bm25_retriever

        self.intake_service = QueryIntakeService()
        self.analyzer = QueryAnalyzer()
        self.query_understanding = QueryUnderstandingService(provider=llm_provider)
        self.classifier = ProductClassifier()
        self.router = JurisdictionDomainRouter()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.decision_engine = RegulatoryDecisionEngine()
        self.evidence_builder = EvidencePackBuilder()
        self.reasoning_engine = ReasoningEngine(provider=llm_provider)
        self.citation_validator = CitationValidator()
        self.confidence_engine = ConfidenceEngine()
        self.abstention_engine = AbstentionEngine()
        self.log_store = log_store or get_default_query_log_store()

    def run(self, request: QueryIntakeRequest) -> QueryResponse:
        # --- 3.1 Query Intake ---
        import time as _time
        _t0 = _time.perf_counter()
        normalized = self.intake_service.intake(request)
        _intake_latency_ms = round((_time.perf_counter() - _t0) * 1000, 2)

        trace = QueryTrace(query_id=normalized.query_id, session_id=normalized.session_id)
        trace.stages.append({
            "stage": "QUERY_INTAKE",
            "latency_ms": _intake_latency_ms,
            "language": normalized.language,
        })

        # --- 3.2 Query Analysis ---
        with trace.stage("QUERY_ANALYSIS") as _:
            analysis = self.analyzer.analyze(normalized)
        trace.stages[-1]["intent"] = analysis.intent.value
        trace.stages[-1]["jurisdiction"] = analysis.jurisdiction

        # --- Stage 1: optional LLM-assisted query understanding ---
        # Advisory only -- adds retrieval phrasings/provisions, never
        # overrides the deterministic intent/jurisdiction/domain routing
        # decided above. See services/query_understanding.py.
        with trace.stage("QUERY_UNDERSTANDING") as _:
            query_understanding = self.query_understanding.understand(normalized, analysis)
        trace.stages[-1]["used"] = query_understanding.used
        trace.stages[-1]["provider"] = query_understanding.provider
        if query_understanding.error:
            trace.stages[-1]["error"] = query_understanding.error

        # --- 3.3 Product Classification (only meaningful with product context) ---
        classification: Optional[ProductClassificationResult] = None
        with trace.stage("CLASSIFICATION") as _:
            # Only run if product_description is explicitly provided or intent/domain is product sensitive
            is_product_sensitive = (
                analysis.intent in (
                    QueryIntent.REGULATORY_CLASSIFICATION,
                    QueryIntent.PRODUCT_COMPLIANCE,
                    QueryIntent.LABELING,
                )
                or any(d in analysis.regulatory_domains for d in (RegulatoryDomain.DRUG, RegulatoryDomain.FOOD, RegulatoryDomain.COSMETIC))
            )
            product_text = normalized.product_description or (
                normalized.normalized_query if (analysis.ayurveda_relevance and is_product_sensitive) else None
            )
            if product_text:
                classification = self.classifier.classify(
                    ProductClassificationRequest(product_description=product_text)
                )
        if classification:
            trace.stages[-1]["classification"] = classification.classification.value

        # --- 3.4 Jurisdiction / Domain Routing ---
        with trace.stage("ROUTING") as _:
            routing = self.router.route(analysis, classification)
        trace.stages[-1]["jurisdiction"] = routing.jurisdiction
        trace.stages[-1]["domains"] = routing.corpus_domains

        with trace.stage("KNOWLEDGE_GRAPH") as _:
            graph_context = self.knowledge_graph.context_for(normalized.normalized_query)
        trace.stages[-1]["entity_count"] = len(graph_context.entities)
        trace.stages[-1]["relationship_count"] = len(graph_context.relationships)

        with trace.stage("DECISION_PLAN") as _:
            decision_plan = self.decision_engine.evaluate(analysis, routing, classification)
        trace.stages[-1]["step_count"] = len(decision_plan.decision_path)

        # --- 3.5 Retrieval + Reranking (reuses existing Phase-1 pipeline) ---
        with trace.stage("RETRIEVAL") as _:
            matched_ids = set(graph_context.matched_entity_ids)
            graph_terms = " ".join(entity.name for entity in graph_context.entities if entity.entity_id in matched_ids)
            query_lower = normalized.normalized_query.lower()
            requested_status = None
            if not any(term in query_lower for term in ("draft", "historical", "superseded", "old version", "previous version")):
                requested_status = "current"

            # Multi-signal retrieval: a single pass over the user's literal
            # wording does not reliably surface a differently-worded statutory
            # provision (e.g. "traditional Ayurvedic formulation" vs. the
            # statute's "aggregation or duplication of known properties of
            # traditionally known component"). Generate a small set of GENERIC
            # additional retrieval signals (explicit provision refs + concept
            # expansions keyed off structured analysis, not the literal query
            # string -- see retrieval/query_expansion.py) and merge candidates
            # by chunk_id, keeping the best score seen for each chunk.
            llm_extra_signals = list(query_understanding.retrieval_queries) + list(query_understanding.llm_provisions)
            search_signals = build_search_signals(
                normalized.normalized_query, analysis, graph_terms or None,
                extra_signals=llm_extra_signals or None,
            )
            per_signal_k = max(normalized.top_k * 4, 20)
            merged: dict = {}
            for signal in search_signals:
                signal_results = self.hybrid_retriever.search(
                    query=signal,
                    top_k=per_signal_k,
                    candidate_k=per_signal_k,
                    jurisdiction=routing.retrieval_filters.get("jurisdiction"),
                    domain=routing.retrieval_filters.get("domain"),
                    authoritative=True if os.environ.get("ALLOW_SYNTHETIC_CORPUS", "false").lower() not in {"1", "true", "yes"} else None,
                    status=requested_status,
                )
                for r in signal_results:
                    existing = merged.get(r.chunk_id)
                    if existing is None or r.score > existing.score:
                        merged[r.chunk_id] = r

            # Explicit-provision exact match: if the user referenced a specific
            # provision ("Section 3(p)"), make sure the chunk(s) whose metadata
            # actually carry that section/subsection are present as candidates
            # even if lexical/semantic scoring alone ranked them low. This is
            # generic metadata matching against WHATEVER provision the regex
            # found -- it does not special-case "3(p)" or any other number.
            requested_provisions = extract_provisions(normalized.normalized_query)
            if requested_provisions and self.bm25_retriever is not None:
                for chunk in self.bm25_retriever.chunks:
                    if chunk.chunk_id in merged:
                        continue
                    md = chunk.metadata
                    for provision in requested_provisions:
                        parsed = parse_provision(provision)
                        if not parsed:
                            continue
                        kind, number, sub = parsed
                        field_for_kind = {"section": md.section, "rule": md.rule, "article": md.article}.get(kind)
                        sub_field_for_kind = {"section": md.subsection, "rule": md.sub_rule, "article": None}.get(kind)
                        if field_for_kind is None:
                            continue
                        section_match = str(field_for_kind).strip().lower() == str(number).strip().lower()
                        sub_match = (not sub) or (
                            sub_field_for_kind is not None
                            and str(sub_field_for_kind).strip().lower() == str(sub).strip().lower()
                        )
                        if section_match and sub_match:
                            boosted = RetrievalResult(
                                chunk_id=chunk.chunk_id,
                                document_id=chunk.document_id,
                                score=1.0,
                                retrieval_method="exact_provision_match",
                                text=chunk.text,
                                metadata=chunk.metadata,
                            )
                            boosted.build_citation()
                            merged[chunk.chunk_id] = boosted
                            break

            candidates = sorted(merged.values(), key=lambda r: r.score, reverse=True)[:per_signal_k]
        trace.stages[-1]["candidate_count"] = len(candidates)
        trace.stages[-1]["search_signals"] = search_signals
        trace.stages[-1]["requested_provisions"] = requested_provisions
        trace.stages[-1]["status_filter"] = requested_status

        with trace.stage("RERANKING") as _:
            final_results: list[RetrievalResult] = self.reranker.rerank(
                normalized.normalized_query, candidates, top_k=normalized.top_k
            )
        trace.stages[-1]["result_count"] = len(final_results)
        trace.stages[-1]["top_scores"] = [round(r.score, 4) for r in final_results[:5]]

        # --- Evidence Pack ---
        with trace.stage("EVIDENCE_BUILD") as _:
            evidence_pack = self.evidence_builder.build(normalized.normalized_query, final_results)
        trace.stages[-1]["evidence_count"] = len(evidence_pack.items)
        trace.stages[-1]["synthetic_only"] = evidence_pack.synthetic_only

        # --- Pre-reasoning Abstention check ---
        with trace.stage("ABSTENTION", phase="pre") as _:
            pre_decision = self.abstention_engine.check_pre_reasoning(analysis, routing, evidence_pack, classification)

        if pre_decision.should_abstain:
            response = self._build_abstained_response(
                normalized, analysis, classification, routing, evidence_pack, pre_decision,
                decision_plan=decision_plan, graph_context=graph_context
            )
            with trace.stage("RESPONSE") as _:
                pass
            self.log_store.save(normalized.query_id, normalized.session_id, response.status,
                                 response.model_dump(), trace.stages)
            return response

        # --- LLM Reasoning ---
        with trace.stage("LLM_REASONING") as _:
            reasoning = self.reasoning_engine.reason(
                normalized.normalized_query, evidence_pack, graph_context, decision_plan,
                language=normalized.language,
            )
        trace.stages[-1]["provider"] = reasoning.provider

        # --- Citation Validation ---
        with trace.stage("CITATION_VALIDATION") as _:
            citation_validation = self.citation_validator.validate(reasoning, evidence_pack)
        trace.stages[-1]["citation_coverage"] = citation_validation.citation_coverage

        # --- Post-reasoning Abstention check ---
        with trace.stage("ABSTENTION", phase="post") as _:
            post_decision = self.abstention_engine.check_post_reasoning(citation_validation.citation_coverage)

        # --- Confidence (always computed; informs both answered & abstained paths) ---
        with trace.stage("CONFIDENCE") as _:
            confidence = self.confidence_engine.compute(
                analysis, routing, evidence_pack, citation_validation.citation_coverage, classification,
                decision_plan=decision_plan, graph_context=graph_context,
            )
        trace.stages[-1]["score"] = confidence.score
        trace.stages[-1]["level"] = confidence.level

        if post_decision.should_abstain:
            response = self._build_abstained_response(
                normalized, analysis, classification, routing, evidence_pack, post_decision,
                confidence=confidence, citation_validation=citation_validation,
                decision_plan=decision_plan, graph_context=graph_context,
            )
            with trace.stage("RESPONSE") as _:
                pass
            self.log_store.save(normalized.query_id, normalized.session_id, response.status,
                                 response.model_dump(), trace.stages)
            return response

        # --- Structured Response ---
        safe_key_points = self.citation_validator.apply_to_key_points(reasoning, citation_validation)
        warnings = list(reasoning.warnings)
        if citation_validation.unsupported_claims:
            warnings.append(
                f"{len(citation_validation.unsupported_claims)} claim(s) could not be fully verified "
                "against retrieved evidence and were flagged as uncertain rather than removed."
            )
        if evidence_pack.synthetic_only:
            warnings.append(
                "All supporting evidence is from the Phase-1 SYNTHETIC development corpus, not a real "
                "authoritative source. Treat this answer as illustrative only."
            )
        if not routing.jurisdiction_is_confident:
            warnings.append("Jurisdiction was not confidently determined; results are not jurisdiction-filtered.")

        with trace.stage("RESPONSE") as _:
            response = QueryResponse(
                query_id=normalized.query_id,
                status="answered",
                query=normalized.original_query,
                language=normalized.language,
                analysis=analysis.model_dump(),
                classification=classification.model_dump() if classification else {},
                routing=routing.model_dump(),
                decision_path=[step.model_dump() for step in decision_plan.decision_path],
                graph_context=graph_context.model_dump(),
                applicable_domains=decision_plan.applicable_concepts,
                answer=reasoning.answer,
                key_points=safe_key_points,
                confidence=confidence.model_dump(),
                evidence=[item.model_dump() for item in evidence_pack.items],
                citations=[c.model_dump() for c in citation_validation.claims],
                citation_coverage=citation_validation.citation_coverage,
                metrics=compute_metrics(evidence_pack, citation_validation, requested_provisions),
                uncertainties=reasoning.uncertainties,
                missing_information=decision_plan.missing_information,
                reasoning_summary=reasoning.reasoning_summary,
                abstained=False,
                warnings=warnings,
                disclaimer=DISCLAIMER,
            )

        self.log_store.save(normalized.query_id, normalized.session_id, response.status,
                             response.model_dump(), trace.stages)
        return response

    def _build_abstained_response(
        self, normalized, analysis, classification, routing, evidence_pack, decision,
        confidence=None, citation_validation=None, decision_plan: Optional[DecisionPlan] = None,
        graph_context=None,
    ) -> QueryResponse:
        warnings = [decision.reason] if decision.reason else []
        if evidence_pack.synthetic_only and evidence_pack.items:
            warnings.append("Only synthetic (non-authoritative) evidence was available.")

        confidence_payload = confidence.model_dump() if confidence else {
            "score": 0.0, "level": "low", "factors": {},
        }

        return QueryResponse(
            query_id=normalized.query_id,
            status="abstained",
            query=normalized.original_query,
            language=normalized.language,
            analysis=analysis.model_dump(),
            classification=classification.model_dump() if classification else {},
            routing=routing.model_dump(),
            decision_path=[step.model_dump() for step in decision_plan.decision_path] if decision_plan else [],
            graph_context=graph_context.model_dump() if graph_context else {},
            applicable_domains=decision_plan.applicable_concepts if decision_plan else [],
            answer="",
            key_points=[],
            confidence=confidence_payload,
            evidence=[item.model_dump() for item in evidence_pack.items],
            citations=[c.model_dump() for c in citation_validation.claims] if citation_validation else [],
            citation_coverage=citation_validation.citation_coverage if citation_validation else 0.0,
            uncertainties=decision.missing_information,
            missing_information=decision.missing_information,
            reasoning_summary="The system abstained before establishing sufficient source-backed support.",
            abstained=True,
            warnings=warnings,
            disclaimer=DISCLAIMER + " This query was abstained rather than answered: "
                       + (decision.reason or "insufficient grounds to answer safely.")
                       + (f" Suggested next step: {decision.recommended_next_step}" if decision.recommended_next_step else ""),
        )
