"""Tests for Stage-1 LLM-assisted query understanding (services/query_understanding.py).

The deterministic QueryAnalyzer remains authoritative for routing; these
tests check that the LLM layer (a) is skipped for simple/auto cases,
(b) degrades safely on provider failure, and (c) never lets a malformed or
adversarial LLM response override deterministic routing -- it can only
contribute additional retrieval phrasings.
"""
import os

from models.query import NormalizedQuery, QueryAnalysis, QueryIntent
from services.llm_provider import BaseLLMProvider, LLMProviderError
from services.query_understanding import QueryUnderstandingService


def _normalized(query: str) -> NormalizedQuery:
    return NormalizedQuery(
        query_id="q_test", session_id="s_test", original_query=query,
        normalized_query=query, language="en", requested_jurisdiction="auto", top_k=5,
    )


class _FakeProvider(BaseLLMProvider):
    name = "ollama"

    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def generate_json(self, system_prompt, user_prompt):
        if self._error:
            raise self._error
        return self._response


def test_auto_mode_skips_llm_for_short_confident_query(monkeypatch):
    monkeypatch.setenv("LLM_QUERY_UNDERSTANDING", "auto")
    provider = _FakeProvider(response={"retrieval_queries": ["should not be called"]})
    service = QueryUnderstandingService(provider=provider)

    analysis = QueryAnalysis(intent=QueryIntent.PATENTABILITY, jurisdiction="India", confidence=0.9)
    result = service.understand(_normalized("What is Section 3(p)?"), analysis)

    assert result.used is False
    assert result.retrieval_queries == []


def test_auto_mode_calls_llm_for_long_ambiguous_query(monkeypatch):
    monkeypatch.setenv("LLM_QUERY_UNDERSTANDING", "auto")
    provider = _FakeProvider(response={
        "intent": "patentability",
        "jurisdiction": "India",
        "domains": ["patents", "traditional_knowledge"],
        "concepts": ["Ayurvedic formulation", "traditional knowledge"],
        "provisions": ["Section 3(p)"],
        "retrieval_queries": ["traditional knowledge patentability India", "Section 3(p) exclusions"],
    })
    service = QueryUnderstandingService(provider=provider)

    analysis = QueryAnalysis(intent=QueryIntent.PATENTABILITY, jurisdiction="India", confidence=0.5,
                              traditional_knowledge_relevance=True)
    result = service.understand(
        _normalized("Can a traditional Ayurvedic formulation be patented in India?"), analysis,
    )

    assert result.used is True
    assert "Section 3(p)" in result.llm_provisions
    assert len(result.retrieval_queries) == 2


def test_llm_query_understanding_disabled_never_calls_provider(monkeypatch):
    monkeypatch.setenv("LLM_QUERY_UNDERSTANDING", "false")

    class _ExplodingProvider(BaseLLMProvider):
        name = "ollama"

        def generate_json(self, system_prompt, user_prompt):
            raise AssertionError("LLM should not be called when disabled")

    service = QueryUnderstandingService(provider=_ExplodingProvider())
    analysis = QueryAnalysis(intent=QueryIntent.UNKNOWN, jurisdiction="unknown")
    result = service.understand(_normalized("x"), analysis)

    assert result.used is False


def test_provider_failure_degrades_gracefully(monkeypatch):
    monkeypatch.setenv("LLM_QUERY_UNDERSTANDING", "true")
    provider = _FakeProvider(error=LLMProviderError("Ollama authentication failed (401)."))
    service = QueryUnderstandingService(provider=provider)

    analysis = QueryAnalysis(intent=QueryIntent.PATENTABILITY, jurisdiction="India")
    result = service.understand(_normalized("Can this be patented in India?"), analysis)

    assert result.used is False
    assert result.error is not None
    assert result.retrieval_queries == []


def test_malformed_llm_response_is_dropped_not_trusted(monkeypatch):
    monkeypatch.setenv("LLM_QUERY_UNDERSTANDING", "true")
    # Adversarial/malformed shape: wrong types, injected extra keys.
    provider = _FakeProvider(response={
        "domains": "not-a-list",
        "retrieval_queries": [{"nested": "object"}, "a real phrase", 123],
        "jurisdiction": None,
    })
    service = QueryUnderstandingService(provider=provider)

    analysis = QueryAnalysis(intent=QueryIntent.PATENTABILITY, jurisdiction="India")
    result = service.understand(_normalized("Can this be patented in India and the EU?"), analysis)

    assert result.llm_domains == []  # non-list dropped entirely, not coerced
    assert "a real phrase" in result.retrieval_queries
    assert "123" in result.retrieval_queries
    assert result.llm_jurisdiction is None
