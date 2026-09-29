"""
LLM Reasoning Engine (stage 6).

The model is only ever shown the Evidence Pack -- never raw retrieval
output, never asked to use outside/model knowledge as evidence. The system
prompt encodes the eleven rules from the spec verbatim in spirit. Output is
requested as strict JSON and is independently re-validated afterwards by
services/citation_validator.py; nothing here is trusted blindly.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from pydantic import ValidationError

from models.evidence import EvidencePack
from models.knowledge import KnowledgeGraphContext
from services.decision_engine import DecisionPlan
from models.answer import ReasoningOutput, KeyPoint
from services.llm_provider import BaseLLMProvider, ExtractiveFallbackProvider, get_default_llm_provider, LLMProviderError

logger = logging.getLogger("reasoning_engine")

_LANG_NAMES = {
    "en": "English",
    "hi": "Hindi (हिन्दी)",
    "te": "Telugu (తెలుగు)",
}


def _system_prompt_for(language: str) -> str:
    """System prompt with the answer language stated up front (models follow the system role best)."""
    lang_name = _LANG_NAMES.get(language, "English")
    return (
        f"RESPONSE LANGUAGE: {lang_name}. Write 'answer', every key_points 'point', 'warnings', "
        f"'uncertainties' and 'reasoning_summary' entirely in {lang_name}, even though the evidence is in English. "
        "Keep only statute/document names, section/rule numbers and evidence_ids in their original form.\n\n"
        + SYSTEM_PROMPT
    )

SYSTEM_PROMPT = """You are the reasoning component of IP-SAKTI Sahayak, an authoritative information/research assistant (NOT a lawyer) for Intellectual Property and regulatory guidance related to Ayurveda.

STRICT RULES:
1. Evidence is authoritative for this response -- you may ONLY use facts and legal principles from the supplied Evidence Pack.
2. ANSWER LANGUAGE: You MUST write the 'answer' field and all explanations in the user's requested language.
3. CITATION FAITHFULNESS: Do NOT translate or invent legal citations, document titles, statute names, section numbers, rule numbers, or authority names. Keep them faithful to their original authoritative form (e.g. "Patents Act, 1970", "Section 3(p)", "Office of the Controller General of Patents, Designs & Trade Marks (IP India)").
4. Do not invent facts, legislation, sections, rules, or citations. Every evidence_id you cite MUST exist in the supplied Evidence Pack.
5. If evidence conflicts, explicitly report the conflict in `warnings`.
6. If evidence is insufficient to answer any part of the question, explicitly state the limitation in `uncertainties`.
7. Distinguish facts (stated directly in evidence) from synthesis/interpretation.
8. Keep jurisdiction explicit in the answer; never blend jurisdictions silently.
9. Every substantive claim in `key_points` MUST include the evidence_id(s) that support it and an `anchor`: a short phrase copied VERBATIM (not translated) from that evidence.
10. Include a concise, professional summary of the legal rationale in `reasoning_summary` (do not output internal private chain-of-thought).

Respond with ONLY a single valid JSON object matching exactly this shape:
{
  "answer": "<thorough, evidence-grounded answer in the requested language>",
  "key_points": [{"point": "<claim in requested language>", "evidence_ids": ["E001"], "anchor": "<short exact English phrase (3-12 words) copied verbatim from the cited evidence that supports the claim>"}],
  "warnings": ["<conflicts, jurisdiction caveats, etc.>"],
  "uncertainties": ["<things the evidence does not cover>"],
  "reasoning_summary": "<concise explanation of the legal basis for the answer>"
}
"""


# ---------------------------------------------------------------------------
# Prompt budget
# ---------------------------------------------------------------------------
_CHARS_PER_TOKEN = 3.0          # conservative (Indic scripts tokenize worse than English)
_MAX_GRAPH_ENTITIES = 12
_MAX_GRAPH_RELATIONSHIPS = 20
_MIN_ITEM_TEXT_CHARS = 400      # never shrink an evidence item below this


def _estimate_tokens(text: str) -> int:
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def _prompt_token_budget(provider: Optional[BaseLLMProvider] = None) -> int:
    """Max tokens for system+user prompt: provider context minus output reserve and a safety margin."""
    limit = getattr(provider, "max_context_tokens", None)
    if not limit:
        limit = os.environ.get("LLM_MAX_CONTEXT_TOKENS") or os.environ.get("OLLAMA_NUM_CTX") or 32768
    limit = int(limit)
    reserve = int(os.environ.get("LLM_OUTPUT_RESERVE_TOKENS", "4096"))
    return max(1024, int((limit - reserve) * 0.85))


def _compact_graph_context(graph_context: KnowledgeGraphContext,
                           max_entities: int = _MAX_GRAPH_ENTITIES,
                           max_relationships: int = _MAX_GRAPH_RELATIONSHIPS) -> str:
    """Compact, bounded graph facts (names + relations only). The full graph stays internal."""
    names = {e.entity_id: e.name for e in graph_context.entities}
    matched = set(graph_context.matched_entity_ids)
    rels = sorted(
        graph_context.relationships,
        key=lambda r: (0 if (r.source_entity_id in matched or r.target_entity_id in matched) else 1,
                       r.relationship_id),
    )[:max_relationships]
    lines = []
    for r in rels:
        src = names.get(r.source_entity_id, r.source_entity_id)
        dst = names.get(r.target_entity_id, r.target_entity_id)
        lines.append(f"- {src} --{r.relationship_type}--> {dst}")
    matched_names = [names[i] for i in sorted(matched) if i in names][:max_entities]
    out = []
    if matched_names:
        out.append("Matched entities: " + "; ".join(matched_names))
    if lines:
        out.append("Relations:\n" + "\n".join(lines))
    omitted = len(graph_context.relationships) - len(rels)
    if omitted > 0:
        out.append(f"(+{omitted} further relations omitted)")
    return "\n".join(out)


def _build_user_prompt(query: str, evidence_pack: EvidencePack,
                       graph_context: Optional[KnowledgeGraphContext] = None,
                       decision_plan: Optional[DecisionPlan] = None,
                       language: str = "en",
                       max_item_chars: Optional[int] = None,
                       graph_limits: Optional[tuple] = None) -> str:
    lang_name = _LANG_NAMES.get(language, "English")
    lines = [
        f"USER QUERY:\n{query}\n",
        f"REQUESTED ANSWER LANGUAGE:\n{lang_name} ({language}) -- You MUST respond in this language while preserving English statutory names and section numbers.\n",
        "EVIDENCE PACK:",
    ]
    if not evidence_pack.items:
        lines.append("(no evidence retrieved)")
    for item in evidence_pack.items:
        text = item.text
        if max_item_chars is not None and len(text) > max_item_chars:
            text = text[:max_item_chars].rstrip() + " ...[truncated]"
        lines.append(
            f"- [{item.evidence_id}] ({item.source_type}) {item.authority} | {item.jurisdiction} | "
            f"{item.title} {('- ' + item.section) if item.section else ''}\n  TEXT: {text}"
        )
    if graph_context:
        graph_text = _compact_graph_context(graph_context, *(graph_limits or ()))
        if graph_text:
            lines.append("\nGRAPH CONTEXT (context only; do not treat it as evidence):")
            lines.append(graph_text)
    if decision_plan:
        lines.append("\nDETERMINISTIC DECISION PLAN (checks to perform, not legal conclusions):")
        lines.append(decision_plan.model_dump_json())
    lines.append(f"\nREMINDER: the final 'answer' and all key points MUST be written in {lang_name}.")
    return "\n".join(lines)


def _fit_user_prompt(query: str, evidence_pack: EvidencePack,
                     graph_context: Optional[KnowledgeGraphContext],
                     decision_plan: Optional[DecisionPlan],
                     language: str, budget_tokens: int) -> str:
    """Build a prompt within budget: compact graph first, then shrink evidence text
    (longest items first). Evidence items are never dropped, so every evidence_id stays
    citable. Raises LLMProviderError if it still cannot fit (caller falls back)."""
    def size(prompt: str) -> int:
        return _estimate_tokens(SYSTEM_PROMPT) + 120 + _estimate_tokens(prompt)

    prompt = _build_user_prompt(query, evidence_pack, graph_context, decision_plan, language)
    if size(prompt) <= budget_tokens:
        return prompt
    # 1) tighter graph
    prompt = _build_user_prompt(query, evidence_pack, graph_context, decision_plan, language,
                                graph_limits=(5, 8))
    if size(prompt) <= budget_tokens:
        return prompt
    # 2) drop graph entirely (context only, never evidence)
    prompt = _build_user_prompt(query, evidence_pack, None, decision_plan, language)
    if size(prompt) <= budget_tokens:
        return prompt
    # 3) progressively cap per-item evidence text
    longest = max((len(i.text) for i in evidence_pack.items), default=0)
    cap = longest
    while cap > _MIN_ITEM_TEXT_CHARS:
        cap = max(_MIN_ITEM_TEXT_CHARS, int(cap * 0.75))
        prompt = _build_user_prompt(query, evidence_pack, None, decision_plan, language, max_item_chars=cap)
        if size(prompt) <= budget_tokens:
            logger.info(f"Reasoning prompt fitted by capping evidence text at {cap} chars.")
            return prompt
    raise LLMProviderError(
        f"prompt exceeds safe context budget ({size(prompt)} > {budget_tokens} est. tokens)")


class ReasoningEngine:
    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self._explicit_provider = provider

    def _resolve_provider(self, evidence_pack: EvidencePack, language: str = "en") -> BaseLLMProvider:
        if self._explicit_provider is not None:
            return self._explicit_provider
        context = [item.model_dump() for item in evidence_pack.items]
        import os as _os
        if _os.environ.get("LLM_REASONING", "true").strip().lower() in {"false", "0", "no", "off"}:
            return ExtractiveFallbackProvider(context=context, language=language)
        provider = get_default_llm_provider(context=context)
        if isinstance(provider, ExtractiveFallbackProvider):
            provider._language = language
        return provider

    def reason(self, query: str, evidence_pack: EvidencePack,
               graph_context: Optional[KnowledgeGraphContext] = None,
               decision_plan: Optional[DecisionPlan] = None,
               language: str = "en") -> ReasoningOutput:
        provider = self._resolve_provider(evidence_pack, language)
        valid_ids = {item.evidence_id for item in evidence_pack.items}
        user_prompt = ""

        try:
            user_prompt = _fit_user_prompt(query, evidence_pack, graph_context, decision_plan, language,
                                           _prompt_token_budget(provider))
            raw = provider.generate_json(_system_prompt_for(language), user_prompt)
            output = self._coerce_output(raw, provider.name)
        except (LLMProviderError, ValidationError, TypeError, KeyError) as e:
            logger.warning(f"LLM provider '{provider.name}' failed or returned invalid structure ({e}); "
                            "falling back to extractive synthesis.")
            fallback = ExtractiveFallbackProvider(context=[item.model_dump() for item in evidence_pack.items],
                                                  language=language)
            raw = fallback.generate_json(SYSTEM_PROMPT, user_prompt)
            output = self._coerce_output(raw, fallback.name)

        # Never trust cited evidence_ids blindly: drop any id that doesn't
        # actually exist in the evidence pack (defends against hallucinated
        # citations even from a well-behaved-looking JSON response).
        for kp in output.key_points:
            kp.evidence_ids = [eid for eid in kp.evidence_ids if eid in valid_ids]

        return output

    @staticmethod
    def _coerce_output(raw: dict, provider_name: str) -> ReasoningOutput:
        key_points_raw = raw.get("key_points", []) or []
        key_points: List[KeyPoint] = []
        for kp in key_points_raw:
            if isinstance(kp, str):
                key_points.append(KeyPoint(point=kp, evidence_ids=[]))
            else:
                key_points.append(KeyPoint(point=kp.get("point", ""), evidence_ids=kp.get("evidence_ids", []) or [],
                                     anchor=str(kp.get("anchor", "") or "")))

        return ReasoningOutput(
            answer=raw.get("answer", ""),
            key_points=key_points,
            warnings=raw.get("warnings", []) or [],
            uncertainties=raw.get("uncertainties", []) or [],
            missing_information=raw.get("missing_information", []) or [],
            follow_up_questions=raw.get("follow_up_questions", []) or [],
            reasoning_summary=raw.get("reasoning_summary", "") or "",
            should_abstain=bool(raw.get("should_abstain", False)),
            provider=provider_name,
        )
