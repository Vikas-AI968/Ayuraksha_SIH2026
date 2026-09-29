"""
Citation Validation (stage 7).

For every substantive claim (key point) the Reasoning Engine produced,
verify that its cited evidence actually exists AND has meaningful lexical
overlap with the claim -- catching both missing citations and citations
that point at the wrong evidence item. Unsupported claims are not silently
let through: they are rewritten as explicit uncertainty and reported.
"""
from __future__ import annotations

import re
from typing import Dict, List

from models.answer import CitationValidationResult, ClaimValidation, ReasoningOutput
from models.evidence import EvidencePack

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "for", "on", "with",
    "as", "by", "be", "this", "that", "it", "under", "may", "must", "shall", "at", "if",
}


def _stem(w: str) -> str:
    if len(w) > 5:
        for suf in ("ations", "ation", "ings", "ing", "ed", "es", "ly", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                return w[: -len(suf)]
    elif len(w) > 3 and w.endswith("s"):
        return w[:-1]
    return w


def _tokenize(text: str) -> set:
    words = re.findall(r"[\w()§.-]+", text.lower())
    return {_stem(w.strip(".-")) or w for w in words if w not in _STOPWORDS and len(w) > 1}


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[\w()§]+", text.lower()))


_LATIN_RE = re.compile(r"^[a-z0-9()§.-]+$")


def _item_strength(claim: str, claim_tokens: set, anchor: str, ev) -> float:
    """Graded, deterministic support of `claim` by one evidence item, in [0, 1]. 0 = unsupported.

    Signals (best wins): verbatim anchor found in the evidence (language independent),
    lexical overlap of stemmed content words, provision/section reference, Latin-term
    alignment for Indic claims. Nothing here calls a model."""
    if not ev.source_url or not ev.document_id or not ev.provenance:
        return 0.0
    ev_text = f"{ev.title} {ev.authority} {ev.section or ''} {ev.text}"
    best = 0.0

    a = _norm(anchor) if anchor else ""
    if a and len(a.split()) >= 2:
        if a in _norm(ev_text):
            best = 1.0
        else:
            a_tok = _tokenize(anchor)
            if a_tok:
                cov = len(a_tok & _tokenize(ev_text)) / len(a_tok)
                if cov >= 0.8:
                    best = max(best, 0.9 * cov)

    ev_tokens = _tokenize(ev_text)
    if claim_tokens:
        overlap = len(claim_tokens & ev_tokens) / len(claim_tokens)
        if overlap >= MIN_OVERLAP_RATIO:
            best = max(best, min(1.0, 0.4 + overlap))
        lc = {w for w in claim_tokens if _LATIN_RE.match(w)}
        le = {w for w in ev_tokens if _LATIN_RE.match(w)}
        if lc and le:
            lat = len(lc & le) / len(lc)
            if lat >= 0.25:
                best = max(best, min(1.0, 0.4 + lat))
    if ev.section and re.search(r"(?<![\w.])" + re.escape(ev.section.lower()) + r"(?![\w])", claim.lower()):
        best = max(best, 0.5)  # valid support, but too weak on its own to re-link a citation
    return round(best, 4)


MIN_OVERLAP_RATIO = 0.10
_REPAIR_MIN_STRENGTH = 0.6  # stricter than validity: a re-linked citation must clearly support the claim


class CitationValidator:
    MIN_OVERLAP_RATIO = MIN_OVERLAP_RATIO

    def validate(self, reasoning: ReasoningOutput, evidence_pack: EvidencePack) -> CitationValidationResult:
        evidence_by_id = {item.evidence_id: item for item in evidence_pack.items}

        claims: List[ClaimValidation] = []
        unsupported_claims: List[str] = []
        strengths: List[float] = []
        correctness: List[float] = []

        for kp in reasoning.key_points:
            claim_tokens = _tokenize(kp.point)
            cited = [eid for eid in dict.fromkeys(kp.evidence_ids) if eid in evidence_by_id]
            per_id = {eid: _item_strength(kp.point, claim_tokens, kp.anchor, evidence_by_id[eid]) for eid in cited}
            good = [eid for eid, st in per_id.items() if st > 0.0]

            supported = bool(good)
            strength = max((per_id[e] for e in good), default=0.0)
            final_ids = good or cited
            repaired = False
            precision = (len(good) / len(cited)) if cited else 0.0

            # Evidence re-linking: the claim may be right but attached to the wrong evidence id.
            # Re-attach the best-matching item from the same pack (deterministic; validated by text).
            if not supported:
                best_id, best_st = None, 0.0
                for eid, ev in evidence_by_id.items():
                    st = _item_strength(kp.point, claim_tokens, kp.anchor, ev)
                    if st > best_st:
                        best_id, best_st = eid, st
                if best_id and best_st >= _REPAIR_MIN_STRENGTH:
                    supported, strength, repaired = True, best_st, True
                    final_ids = [best_id]

            # Indic claim citing valid retrieved evidence but no verifiable anchor/overlap:
            # accepted (as before) but only at partial strength.
            if cited and not supported and any(ord(c) > 127 for c in kp.point) \
                    and any(evidence_by_id[e].provenance for e in cited):
                supported, strength = True, 0.45
                final_ids = cited

            if supported:
                claim_conf = strength * (0.85 if repaired else (precision if cited else 0.0))
                if not repaired and precision == 0.0:  # supported only via the Indic fallback
                    claim_conf = strength
            else:
                claim_conf = 0.0
                strength = 0.0
            claim_conf = round(max(0.0, min(1.0, claim_conf)), 4)

            claims.append(ClaimValidation(
                claim=kp.point, supported=supported, evidence_ids=final_ids,
                confidence=claim_conf, strength=round(strength, 4), repaired=repaired,
            ))
            strengths.append(strength)
            correctness.append(claim_conf)
            if not supported:
                unsupported_claims.append(kp.point)

        substantive_claims = len(claims)
        supported_claims = sum(1 for c in claims if c.supported)
        citation_coverage = round(supported_claims / substantive_claims, 4) if substantive_claims else 0.0

        return CitationValidationResult(
            claims=claims,
            citation_coverage=citation_coverage,
            unsupported_claims=unsupported_claims,
            citation_correctness=round(sum(correctness) / substantive_claims, 4) if substantive_claims else 0.0,
            faithfulness=round(sum(strengths) / substantive_claims, 4) if substantive_claims else 0.0,
        )

    @staticmethod
    def apply_to_key_points(reasoning: ReasoningOutput, validation: CitationValidationResult) -> List[str]:
        """Produce the final, safe key_points list for the API response:
        unsupported claims are qualified as uncertain rather than dropped
        silently, so the caller can see exactly what was not grounded.
        """
        validation_by_claim: Dict[str, ClaimValidation] = {c.claim: c for c in validation.claims}
        final_points: List[str] = []
        for kp in reasoning.key_points:
            v = validation_by_claim.get(kp.point)
            if v and v.supported:
                final_points.append(kp.point)
            else:
                final_points.append(f"[Unverified against evidence -- treat as uncertain] {kp.point}")
        return final_points


def compute_metrics(evidence_pack: EvidencePack, validation: CitationValidationResult,
                    requested_provisions=None) -> dict:
    """Live, deterministic quality signals derived only from data the query already produced
    (no extra retrieval, embedding or LLM call).

    retrieval    = 0.5 * rank-weighted relevance of the evidence pack
                 + 0.5 * hit quality (requested-provision recall, else rank of the first evidence
                   item that actually grounds a claim).
    citation_correctness / faithfulness come from the claim-level validation above."""
    items = evidence_pack.items
    if not items:
        return {"retrieval": 0.0, "citation_correctness": 0.0, "faithfulness": 0.0}
    weights = [1.0 / (i + 1) for i in range(len(items))]
    rel_w = sum(w * it.relevance_score for w, it in zip(weights, items)) / sum(weights)

    hit = None
    if requested_provisions:
        from retrieval.query_expansion import parse_provision
        found = 0
        total = 0
        for prov in requested_provisions:
            parsed = parse_provision(prov)
            if not parsed:
                continue
            total += 1
            kind, number, sub = parsed
            for it in items:
                field = {"section": it.section, "rule": it.rule, "article": it.article}.get(kind)
                subf = {"section": it.subsection, "rule": it.sub_rule}.get(kind)
                if field and str(field).strip().lower() == str(number).strip().lower() and \
                        (not sub or (subf and str(subf).strip().lower() == str(sub).strip().lower())):
                    found += 1
                    break
        if total:
            hit = found / total
    if hit is None:
        rank_of = {it.evidence_id: i + 1 for i, it in enumerate(items)}
        ranks = [rank_of[e] for c in validation.claims if c.supported for e in c.evidence_ids if e in rank_of]
        hit = 0.0 if not ranks else (1.0 if min(ranks) <= 3 else 0.6)

    return {
        "retrieval": round(0.5 * rel_w + 0.5 * hit, 4),
        "citation_correctness": validation.citation_correctness,
        "faithfulness": validation.faithfulness,
    }
