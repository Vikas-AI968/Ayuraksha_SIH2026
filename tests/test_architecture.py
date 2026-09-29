"""Focused tests for the Phase-2 regulatory-intelligence layers."""
from types import SimpleNamespace

from models.classification import ProductClass, ProductClassificationResult
from models.knowledge import KnowledgeEntity, KnowledgeRelationship
from models.query import QueryAnalysis, QueryIntent
from services.decision_engine import RegulatoryDecisionEngine
from services.knowledge_graph import KnowledgeGraph
from services.llm_provider import OllamaProvider, LLMProviderError, get_default_llm_provider, ExtractiveFallbackProvider
from services.router import RoutingDecision
from models.evidence import EvidenceItem, EvidencePack
from services.abstention import AbstentionEngine
from services.confidence import ConfidenceEngine
from ingestion.source_service import SourceIngestionService
from ingestion.pipeline import IngestionPipeline
from sources.adapters.configured import RestrictedSourceAdapter
from services.router import RoutingDecision


def test_knowledge_graph_returns_related_entities_and_relationships():
    graph = KnowledgeGraph(":memory:")
    graph.add_entity(KnowledgeEntity(entity_id="patents-act", name="Patents Act", entity_type="statute"))
    graph.add_entity(KnowledgeEntity(entity_id="section-3p", name="Section 3(p)", entity_type="provision"))
    graph.add_relationship(KnowledgeRelationship(
        relationship_id="r1", source_entity_id="patents-act", relationship_type="contains",
        target_entity_id="section-3p", source_document_id="doc-1",
    ))

    context = graph.context_for("Section 3(p) traditional knowledge")

    entity_ids = {entity.entity_id for entity in context.entities}
    assert {"patents-act", "section-3p"}.issubset(entity_ids)
    assert any(relationship.relationship_id == "r1" and relationship.relationship_type == "contains"
               for relationship in context.relationships)


def test_decision_engine_emits_checks_without_legal_conclusion():
    analysis = QueryAnalysis(
        intent=QueryIntent.PATENTABILITY,
        jurisdiction="India",
        confidence=0.9,
        traditional_knowledge_relevance=True,
    )
    routing = RoutingDecision(
        jurisdiction="India", jurisdiction_is_confident=True,
        retrieval_filters={"jurisdiction": "India"}, corpus_domains=["Patents"],
        sources_selected=["india-code"],
    )
    classification = ProductClassificationResult(
        classification=ProductClass.CLASSICAL_MEDICINE, confidence=0.8,
    )

    plan = RegulatoryDecisionEngine().evaluate(analysis, routing, classification)
    steps = {step.step for step in plan.decision_path}

    assert "traditional_knowledge" in steps
    assert "patentability_provisions" in steps
    assert not any("patentable" in step.purpose.lower() for step in plan.decision_path)


def test_ollama_provider_uses_configured_model_and_json(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs["json"]
        return SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": '{"answer":"grounded"}'}},
        )

    monkeypatch.setattr("httpx.post", fake_post)
    provider = OllamaProvider(model="qwen3:test", base_url="http://ollama.test")
    result = provider.generate_json("system", "user")

    assert result["answer"] == "grounded"
    assert captured["url"] == "http://ollama.test/api/chat"
    assert captured["json"]["model"] == "qwen3:test"
    assert captured["json"]["format"] == "json"


def test_ollama_cloud_provider_sends_bearer_token(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        captured["json"] = kwargs["json"]
        return SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": '{"answer":"grounded"}'}},
        )

    monkeypatch.setattr("httpx.post", fake_post)
    provider = OllamaProvider(
        model="gpt-oss:120b-cloud", base_url="https://ollama.com", api_key="test-key-123",
    )
    result = provider.generate_json("system", "user")

    assert result["answer"] == "grounded"
    assert captured["url"] == "https://ollama.com/api/chat"
    assert captured["headers"]["Authorization"] == "Bearer test-key-123"
    assert captured["json"]["model"] == "gpt-oss:120b-cloud"
    assert captured["json"]["format"] == "json"


def test_ollama_provider_without_api_key_omits_auth_header(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["headers"] = kwargs["headers"]
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"message": {"content": '{"answer":"ok"}'}},
        )

    monkeypatch.setattr("httpx.post", fake_post)
    provider = OllamaProvider(model="qwen3:test", base_url="http://localhost:11434", api_key=None)
    provider.generate_json("system", "user")

    assert "Authorization" not in captured["headers"]


def test_ollama_provider_401_raises_without_retry(monkeypatch):
    calls = {"count": 0}

    def fake_post(url, **kwargs):
        calls["count"] += 1
        return SimpleNamespace(status_code=401, text="unauthorized")

    monkeypatch.setattr("httpx.post", fake_post)
    provider = OllamaProvider(api_key="bad-key", max_retries=2)

    try:
        provider.generate_json("system", "user")
        assert False, "expected LLMProviderError"
    except LLMProviderError as e:
        assert "401" in str(e) or "auth" in str(e).lower()
    assert calls["count"] == 1  # auth failures are never retried


def test_ollama_provider_retries_on_5xx_then_succeeds(monkeypatch):
    calls = {"count": 0}

    def fake_post(url, **kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            return SimpleNamespace(status_code=503, text="unavailable")
        return SimpleNamespace(status_code=200, json=lambda: {"message": {"content": '{"answer":"recovered"}'}})

    monkeypatch.setattr("httpx.post", fake_post)
    monkeypatch.setattr(OllamaProvider, "_sleep_backoff", lambda self, attempt: None)
    provider = OllamaProvider(max_retries=2)
    result = provider.generate_json("system", "user")

    assert result["answer"] == "recovered"
    assert calls["count"] == 3


def test_ollama_provider_exhausted_retries_raises(monkeypatch):
    def fake_post(url, **kwargs):
        return SimpleNamespace(status_code=429, text="rate limited")

    monkeypatch.setattr("httpx.post", fake_post)
    monkeypatch.setattr(OllamaProvider, "_sleep_backoff", lambda self, attempt: None)
    provider = OllamaProvider(max_retries=1)

    try:
        provider.generate_json("system", "user")
        assert False, "expected LLMProviderError"
    except LLMProviderError as e:
        assert "429" in str(e) or "rate" in str(e).lower()


def test_ollama_provider_timeout_raises_llm_provider_error(monkeypatch):
    import httpx

    def fake_post(url, **kwargs):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr("httpx.post", fake_post)
    monkeypatch.setattr(OllamaProvider, "_sleep_backoff", lambda self, attempt: None)
    provider = OllamaProvider(max_retries=0)

    try:
        provider.generate_json("system", "user")
        assert False, "expected LLMProviderError"
    except LLMProviderError as e:
        assert "timed out" in str(e).lower()


def test_get_default_llm_provider_prefers_ollama_cloud_when_key_present(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OLLAMA_API_KEY", "cloud-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.com")
    monkeypatch.setenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    provider = get_default_llm_provider()

    assert isinstance(provider, OllamaProvider)
    assert provider.api_key == "cloud-key"
    assert provider.model == "gpt-oss:120b-cloud"


def test_get_default_llm_provider_falls_back_to_extractive_with_no_config(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_ENABLED", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    provider = get_default_llm_provider()

    assert isinstance(provider, ExtractiveFallbackProvider)


def test_graph_traversal_is_bounded():
    graph = KnowledgeGraph(":memory:")
    context = graph.context_for("Section 3(p)", max_depth=2)

    assert context.traversal_depth == 2
    assert len(context.entities) < 20
    assert "section-3p" in context.matched_entity_ids


def test_synthetic_sensitive_evidence_abstains_and_confidence_is_capped():
    evidence = EvidencePack(items=[EvidenceItem(
        evidence_id="E001", document_id="d1", chunk_id="c1", source_id="synthetic-phase1-corpus",
        official_url="synthetic://d1", authority="Synthetic", jurisdiction="India", domain="Patents",
        document_type="Guidance", title="Synthetic", text="Section 3(p) traditional knowledge.",
        relevance_score=1.0, source_url="synthetic://d1", source_type="synthetic",
        effective_date="2026-01-01", version="1.0", retrieval_method="hybrid",
        citation="Synthetic citation", provenance="synthetic_development_corpus",
    )], query="Can this be patented in India?", synthetic_only=True, jurisdictions_present=["India"], domains_present=["Patents"],
        average_relevance=1.0, top_relevance=1.0)
    analysis = QueryAnalysis(intent=QueryIntent.PATENTABILITY, jurisdiction="India")
    routing = RoutingDecision(jurisdiction="India", jurisdiction_is_confident=True,
                              retrieval_filters={}, corpus_domains=["Patents"], sources_selected=[])

    decision = AbstentionEngine().check_pre_reasoning(analysis, routing, evidence)
    confidence = ConfidenceEngine().compute(analysis, routing, evidence, 1.0)

    assert decision.should_abstain is True
    assert confidence.score <= 0.20
    assert confidence.level == "low"


def test_restricted_source_remains_not_ingested():
    report = SourceIngestionService(IngestionPipeline(registry_file=":memory:")).ingest(
        RestrictedSourceAdapter("tkdl")
    )

    assert report.status == "not_ingested"
    assert report.document_count == 0
    assert report.errors
