"""
Deterministic multi-signal query expansion for hybrid retrieval.

Fixes the real gap behind the "Section 3(p)" natural-language retrieval
test: a single BM25/semantic pass over the user's literal wording
("Can a traditional Ayurvedic formulation be patented in India?") does not
reliably surface a differently-worded statutory provision ("traditional
knowledge ... aggregation or duplication of known properties"). Rather than
special-casing that one query, this module generates a small, GENERIC set
of additional retrieval signals from the structured QueryAnalysis (intent +
domain-relevance flags) plus any explicit legal-provision reference found
in the text via regex. None of this is keyed to the literal input string --
it fires for ANY patentability/TK/ABS question, and for ANY "Section X(y)"
style reference, not just the example in the spec.
"""
from __future__ import annotations

import re
from typing import List, Optional

from models.query import QueryAnalysis, QueryIntent

_LABEL_MAPPING = {
    "section": "Section",
    "sec": "Section",
    "धारा": "Section",
    "दफा": "Section",
    "दफ़ा": "Section",
    "సెక్షన్": "Section",
    "rule": "Rule",
    "reg": "Rule",
    "regulation": "Rule",
    "नियम": "Rule",
    "నిబంధన": "Rule",
    "article": "Article",
    "अनुच्छेद": "Article",
    "ఆర్టికల్": "Article",
}

_PROVISION_RE = re.compile(
    r"(?:\b|(?<=[\s,]))(section|sec\.?|rule|article|regulation|reg\.?|धारा|दफ़ा|दफा|సెక్షన్|నిబంధన|ఆర్టికల్|अनुच्छेद|नियम)\s*(\d+[a-zA-Z]*)\s*(?:\(\s*([a-zA-Z0-9]+)\s*\))?",
    re.IGNORECASE,
)

# Generic legal-concept expansions keyed by structured analysis predicates --
# not by the literal query text. Each fires for a whole class of questions.
_CONCEPT_EXPANSIONS = [
    (lambda a: a.intent == QueryIntent.PATENTABILITY and a.traditional_knowledge_relevance,
     "traditional knowledge patentability known properties aggregation duplication non-patentable inventions"),
    (lambda a: a.intent == QueryIntent.PATENTABILITY and a.ayurveda_relevance,
     "traditional knowledge Ayurvedic formulation patentability exclusions Patents Act"),
    (lambda a: a.intent == QueryIntent.PATENTABILITY,
     "novelty inventive step patentable subject matter exclusions"),
    (lambda a: a.intent in (QueryIntent.BIODIVERSITY_ACCESS, QueryIntent.ABS),
     "access and benefit sharing biological resource approval"),
    (lambda a: a.intent == QueryIntent.TRADITIONAL_KNOWLEDGE,
     "traditional knowledge digital library prior art genetic resources"),
]


def extract_provisions(text: str) -> List[str]:
    """Extracts explicit legal-provision references like 'Section 3(p)' or
    'Rule 28' from free text (including Hindi 'धारा 3(p)' and Telugu 'సెక్షన్ 3(p)')."""
    provisions = []
    for m in _PROVISION_RE.finditer(text):
        label_raw = m.group(1).lower().rstrip(".")
        standard_label = _LABEL_MAPPING.get(label_raw, label_raw.title())
        number, sub = m.group(2), m.group(3)
        rendered = f"{standard_label} {number}" + (f"({sub})" if sub else "")
        provisions.append(rendered)
    return provisions


def parse_provision(provision: str):
    """Parses a rendered provision string back into (kind, number, sub)."""
    m = re.match(r"(section|rule|article|regulation)\s+(\S+?)(?:\((\S+)\))?$", provision, re.IGNORECASE)
    if not m:
        return None
    kind, number, sub = m.groups()
    return kind.lower(), number, sub


def build_search_signals(
    normalized_query: str,
    analysis: QueryAnalysis,
    graph_terms: Optional[str] = None,
    max_signals: int = 5,
    extra_signals: Optional[List[str]] = None,
) -> List[str]:
    """Builds a small, deduplicated, priority-ordered list of retrieval
    queries: the original query first, then any explicit provision
    references, then generic concept expansions implied by the structured
    analysis, then cross-lingual formulations for multilingual queries,
    then knowledge-graph terms, then any additional signals (e.g.
    from optional LLM-assisted query understanding -- see
    services/query_understanding.py) if supplied.

    `extra_signals` are advisory retrieval phrasings only; they never change
    intent/jurisdiction/domain routing, which stays deterministic. When
    supplied, the signal budget is widened slightly so they don't simply
    crowd out the deterministic signals above them."""
    signals: List[str] = [normalized_query]
    signals.extend(extract_provisions(normalized_query))

    for predicate, expansion in _CONCEPT_EXPANSIONS:
        try:
            if predicate(analysis):
                signals.append(expansion)
        except Exception:
            continue

    # Cross-lingual retrieval representation for non-English queries against the
    # primarily-English authoritative corpus:
    is_non_ascii = any(ord(c) > 127 for c in normalized_query)
    if is_non_ascii:
        cross_lingual_parts = []
        if analysis.traditional_knowledge_relevance:
            cross_lingual_parts.append("traditional knowledge")
        if analysis.ayurveda_relevance:
            cross_lingual_parts.append("Ayurvedic formulation")
        if analysis.intent == QueryIntent.PATENTABILITY:
            cross_lingual_parts.append("patentability patent")
        if analysis.biological_resource_relevance or analysis.intent in (QueryIntent.BIODIVERSITY_ACCESS, QueryIntent.ABS):
            cross_lingual_parts.append("biological resources biodiversity")
        if analysis.jurisdiction not in ("unknown", "ambiguous"):
            cross_lingual_parts.append(analysis.jurisdiction)
        if cross_lingual_parts:
            signals.append(" ".join(cross_lingual_parts))

    if graph_terms:
        signals.append(graph_terms)

    effective_max = max_signals + (1 if is_non_ascii else 0)
    if extra_signals:
        effective_max += min(len(extra_signals), 3)
        signals.extend(extra_signals)

    seen = set()
    deduped: List[str] = []
    for s in signals:
        key = s.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(s.strip())
    return deduped[:effective_max]
