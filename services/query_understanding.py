"""
LLM-Assisted Query Understanding (Stage 1 of the spec's two-stage LLM usage).

The deterministic QueryAnalyzer (services/query_analyzer.py) always runs and
remains the safety layer for intent/jurisdiction/domain routing -- it cannot
be overridden. This module optionally calls the configured LLM provider
(gpt-oss:120b-cloud via Ollama Cloud, by default) to interpret genuinely
complex or ambiguous natural-language queries and propose ADDITIONAL
retrieval signals: alternative phrasings, likely statutory provisions, and
concepts. Its output is treated purely as advisory retrieval hints:

- it is never trusted for jurisdiction/intent/domain routing decisions;
- malformed or missing fields are dropped, never guessed;
- any provider failure (timeout, auth, invalid JSON, rate limit) is caught
  and logged; the pipeline continues with deterministic-only signals -- a
  Stage-1 LLM failure must never break or block a query.

Controlled by LLM_QUERY_UNDERSTANDING = "true" | "false" | "auto" (default
"auto": skip the LLM call for short, already-unambiguous explicit queries to
avoid unnecessary external API calls; use it for longer natural-language
questions or when the deterministic analyzer could not confidently resolve
intent/jurisdiction).
"""
from __future__ import annotations

import logging
import os
import re
from typing import List, Optional

from pydantic import BaseModel, Field

from models.query import NormalizedQuery, QueryAnalysis
from services.llm_provider import BaseLLMProvider, LLMProviderError, get_default_llm_provider

logger = logging.getLogger("query_understanding")

SYSTEM_PROMPT = """You are the query-understanding component of IP-SAKTI Sahayak, an \
information/research assistant for Intellectual Property and regulatory questions \
related to Ayurveda, primarily under Indian law.

You do NOT answer the user's question. You only propose structured retrieval hints \
that help a downstream hybrid (BM25 + semantic) retrieval system find the right \
statutory text. You must never invent statutes, section numbers, rules, or legal \
conclusions -- if you are not sure a provision exists, omit it rather than guess.

Respond with ONLY a single JSON object, no prose outside it, matching exactly this shape:
{
  "intent": "<short label, e.g. patentability>",
  "jurisdiction": "<jurisdiction you infer, e.g. India, or 'unknown' if unclear>",
  "domains": ["<domain labels, e.g. patents, traditional_knowledge>"],
  "concepts": ["<key legal/technical concepts present in the query>"],
  "provisions": ["<statutory provisions the query plausibly concerns, e.g. Section 3(p)>"],
  "retrieval_queries": ["<3-6 short alternative search phrasings that would help retrieve the relevant statutory text, worded differently from the user's literal question>"]
}
Do not include any other keys.
"""


class QueryUnderstanding(BaseModel):
    """Advisory, non-authoritative output of Stage-1 LLM query understanding."""

    llm_intent: Optional[str] = None
    llm_jurisdiction: Optional[str] = None
    llm_domains: List[str] = Field(default_factory=list)
    llm_concepts: List[str] = Field(default_factory=list)
    llm_provisions: List[str] = Field(default_factory=list)
    retrieval_queries: List[str] = Field(default_factory=list)
    provider: str = "none"
    used: bool = False
    error: Optional[str] = None


_PROVISION_REF_RE = re.compile(r"\b(?:section|sec|rule|article|regulation|clause)\s*\d+[a-z]?(?:\s*\(\w+\))*", re.I)


def _is_complex_enough_for_llm(normalized_query: str, analysis: QueryAnalysis) -> bool:
    """Heuristic for LLM_QUERY_UNDERSTANDING=auto.

    Skips the LLM call for short, explicit queries the deterministic
    analyzer already resolved confidently (e.g. "What is Section 3(p)?"),
    and uses it for longer natural-language questions, or whenever the
    deterministic analyzer could not confidently resolve intent or
    jurisdiction on its own."""
    if analysis.intent.value == "unknown":
        return True
    if analysis.jurisdiction in ("unknown", "ambiguous"):
        return True
    if analysis.missing_information or analysis.confidence < 0.5:
        return True
    # A resolved query anchored on an explicit provision reference
    # (e.g. "Section 3(p)", "Rule 12", "Article 7") is deterministic enough.
    if _PROVISION_REF_RE.search(normalized_query):
        return False
    # Otherwise only very long free-form questions warrant the extra LLM step.
    return len(normalized_query.split()) >= 20


class QueryUnderstandingService:
    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self._explicit_provider = provider

    def understand(self, normalized: NormalizedQuery, analysis: QueryAnalysis) -> QueryUnderstanding:
        mode = os.environ.get("LLM_QUERY_UNDERSTANDING", "auto").strip().lower()
        if mode in ("false", "0", "no", "off"):
            return QueryUnderstanding()
        if mode == "auto" and not _is_complex_enough_for_llm(normalized.normalized_query, analysis):
            return QueryUnderstanding()

        provider = self._explicit_provider or get_default_llm_provider()
        if provider.name == "extractive":
            # No generative provider configured -- nothing useful to add,
            # and calling it would just re-echo the evidence-less prompt.
            return QueryUnderstanding(provider=provider.name)

        user_prompt = (
            f"USER QUERY:\n{normalized.normalized_query}\n\n"
            "Deterministic analysis so far (context only -- add value, don't just repeat it):\n"
            f"intent={analysis.intent.value}, jurisdiction={analysis.jurisdiction}, "
            f"ip_domains={[d.value for d in analysis.ip_domains]}, "
            f"regulatory_domains={[d.value for d in analysis.regulatory_domains]}, "
            f"ayurveda_relevance={analysis.ayurveda_relevance}, "
            f"traditional_knowledge_relevance={analysis.traditional_knowledge_relevance}"
        )

        try:
            raw = provider.generate_json(SYSTEM_PROMPT, user_prompt)
        except LLMProviderError as exc:
            logger.info(f"LLM query understanding unavailable ({exc}); continuing deterministic-only.")
            return QueryUnderstanding(provider=provider.name, error=str(exc))
        except Exception as exc:
            # A Stage-1 LLM failure of any kind must never break query
            # processing -- degrade to deterministic-only signals.
            logger.warning(f"LLM query understanding failed unexpectedly ({exc}); continuing deterministic-only.")
            return QueryUnderstanding(provider=provider.name, error=str(exc))

        try:
            domains = _clean_str_list(raw.get("domains"), limit=8)
            concepts = _clean_str_list(raw.get("concepts"), limit=8)
            provisions = _clean_str_list(raw.get("provisions"), limit=8)
            retrieval_queries = _clean_str_list(raw.get("retrieval_queries"), limit=6)
            llm_intent = raw.get("intent")
            llm_jurisdiction = raw.get("jurisdiction")
            return QueryUnderstanding(
                llm_intent=str(llm_intent) if llm_intent else None,
                llm_jurisdiction=str(llm_jurisdiction) if llm_jurisdiction else None,
                llm_domains=domains,
                llm_concepts=concepts,
                llm_provisions=provisions,
                retrieval_queries=retrieval_queries,
                provider=provider.name,
                used=True,
            )
        except Exception as exc:
            logger.warning(f"LLM query understanding returned a malformed structure ({exc}); ignoring it.")
            return QueryUnderstanding(provider=provider.name, error=str(exc))


def _clean_str_list(value, limit: int) -> List[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, (str, int, float)):
            text = str(item).strip()
            if text:
                out.append(text)
    return out[:limit]
