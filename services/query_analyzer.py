"""
Query Intake (3.1) and Query Analyzer (3.2).

The analyzer is deterministic and rule/keyword based rather than an LLM
call. This is a deliberate choice: the spec says "Do not rely blindly on
LLM output. Validate the structure." The most robust way to guarantee a
validated structure for the routing-critical fields (intent, jurisdiction,
domains) is to not depend on a generative model for them at all -- keyword
rules are fast, free, fully deterministic/testable, and cannot hallucinate
a jurisdiction or domain that was never mentioned. The LLM is reserved for
stage 6 (answer generation over the evidence pack), where its output is
independently validated by the Citation Validator.
"""
from __future__ import annotations

import re
import uuid
from typing import Dict, List, Tuple

from models.query import (
    NormalizedQuery,
    QueryAnalysis,
    QueryIntakeRequest,
    QueryIntent,
    IPDomain,
    RegulatoryDomain,
)

# --- Language detection (lightweight, dependency-free) ---------------------
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")


def detect_language(text: str) -> str:
    """Detect language based on unicode script. Defaults to English for Latin."""
    if _TELUGU_RE.search(text):
        return "te"
    if _DEVANAGARI_RE.search(text):
        return "hi"
    return "en"


class QueryIntakeService:
    """Stage 3.1 -- accepts, normalizes, and prepares a raw query."""

    def intake(self, request: QueryIntakeRequest) -> NormalizedQuery:
        query_id = f"q_{uuid.uuid4().hex[:12]}"
        session_id = request.session_id or f"s_{uuid.uuid4().hex[:12]}"
        normalized_query = " ".join(request.query.strip().split())
        req_lang = (request.language or "auto").strip().lower()
        detected = detect_language(normalized_query)
        if req_lang in ("hi", "te"):
            language = req_lang
        else:
            # "auto"/"en"/unsupported: the query's own script wins. An explicit "en" is
            # the UI default and must not force English onto a Hindi/Telugu question.
            language = detected if detected in ("en", "hi", "te") else "en"

        return NormalizedQuery(
            query_id=query_id,
            session_id=session_id,
            original_query=request.query,
            normalized_query=normalized_query,
            language=language,
            requested_jurisdiction=request.jurisdiction or "auto",
            top_k=request.top_k,
            product_description=request.product_description,
        )


# --- Keyword rule tables (English + Hindi + Telugu) ------------------------

_INTENT_KEYWORDS: Dict[QueryIntent, List[str]] = {
    QueryIntent.PATENTABILITY: [
        "patent", "patentable", "patentability", "section 3(p)", "section 3(d)",
        "prior art", "novelty", "inventive step", "claim allowability",
        # Hindi
        "पेटेंट", "पेटेंटेबिलिटी", "पेटेंट योग्यता", "धारा 3(p)", "धारा 3(d)", "नवीनता", "आविष्कारक कदम",
        # Telugu
        "పేటెంట్", "పేటెంట్ పొందవచ్చా", "సెక్షన్ 3(p)", "సెక్షన్ 3(d)", "నూతనత్వం", "ఆవిష్కరణ",
    ],
    QueryIntent.TRADEMARK: [
        "trademark", "trade mark", "brand name registration", "tm registration", "logo registration",
        "ट्रेडमार्क", "ब्रांड", "ట్రేడ్‌మార్క్", "బ్రాండ్",
    ],
    QueryIntent.COPYRIGHT: [
        "copyright", "authorship", "literary work", "artistic work",
        "कॉपीराइट", "కాపీరైట్",
    ],
    QueryIntent.DESIGN: [
        "industrial design", "design registration", "product design protection",
        "डिजाइन", "డిజైన్",
    ],
    QueryIntent.TRADE_SECRET: [
        "trade secret", "confidential formulation", "undisclosed information", "nda",
        "व्यापार रहस्य", "రహస్య ఫార్ములా",
    ],
    QueryIntent.TRADITIONAL_KNOWLEDGE: [
        "traditional knowledge", "tkdl", "classical text", "ayurvedic classical", "folk medicine", "indigenous knowledge",
        # Hindi
        "पारंपरिक ज्ञान", "पारम्परिक ज्ञान", "टीकेडीएल", "शास्त्रीय ग्रंथ", "लोक चिकित्सा",
        # Telugu
        "సాంప్రదాయ జ్ఞానం", "సాంప్రదాయక జ్ఞానం", "సంప్రదాయ జ్ఞానం", "టీకేడీఎల్", "శాస్త్రీయ గ్రంథాలు",
    ],
    QueryIntent.BIODIVERSITY_ACCESS: [
        "biological resource", "biodiversity access", "bioprospecting", "access to biological",
        "जैविक संसाधन", "जैव विविधता", "జీవ వనరులు", "జీవ వైవిధ్యం",
    ],
    QueryIntent.ABS: [
        "access and benefit sharing", "abs approval", "nba approval", "benefit sharing", "abs fee",
        "लाभ साझाकरण", "एनबीए", "ప్రయోజనాల భాగస్వామ్యం", "ఎన్‌బీఏ",
    ],
    QueryIntent.REGULATORY_CLASSIFICATION: [
        "classify", "classification", "what category", "which category", "regulatory classification", "drug classification",
        "वर्गीकरण", "श्रेणी", "వర్గీకరణ", "ఏ వర్గం",
    ],
    QueryIntent.PRODUCT_COMPLIANCE: [
        "compliance", "regulatory approval", "license to manufacture", "approval required", "compliant",
        "अनुपालन", "लाइसेंस", "स्वीकृति", "సమ్మతి", "లైసెన్స్", "అనుమతి",
    ],
    QueryIntent.LABELING: [
        "label", "labeling", "labelling", "packaging requirement",
        "लेबल", "लेबलिंग", "లేబుల్", "లేబులింగ్",
    ],
    QueryIntent.IMPORT_EXPORT: [
        "import", "export", "customs", "cross-border", "destination market",
        "आयात", "निर्यात", "దిగుమతి", "ఎగుమతి",
    ],
    QueryIntent.LICENSING: [
        "licensing", "license agreement", "royalty", "technology transfer",
        "लाइसेंसिंग", "లైసెన్సింగ్",
    ],
    QueryIntent.PRIOR_ART: [
        "prior art search", "existing patents", "already patented",
        "पूर्व कला", "ముందస్తు పేటెంట్లు",
    ],
}

_IP_DOMAIN_KEYWORDS: Dict[IPDomain, List[str]] = {
    IPDomain.PATENT: [
        "patent", "section 3(", "novelty", "inventive step", "patentability",
        "पेटेंट", "धारा 3(", "పేటెంట్", "సెక్షన్ 3(",
    ],
    IPDomain.TRADEMARK: ["trademark", "trade mark", "brand", "ट्रेडमार्क", "ట్రేడ్‌మార్క్"],
    IPDomain.COPYRIGHT: ["copyright", "कॉपीराइट", "కాపీరైట్"],
    IPDomain.DESIGN: ["industrial design", "design registration", "डिजाइन", "డిజైన్"],
    IPDomain.TRADE_SECRET: ["trade secret", "confidential formulation", "व्यापार रहस्य"],
    IPDomain.TRADITIONAL_KNOWLEDGE: [
        "traditional knowledge", "tkdl", "classical text",
        "पारंपरिक ज्ञान", "पारम्परिक ज्ञान", "సాంప్రదాయ జ్ఞానం", "సంప్రదాయ జ్ఞానం",
    ],
}

_REGULATORY_DOMAIN_KEYWORDS: Dict[RegulatoryDomain, List[str]] = {
    RegulatoryDomain.AYURVEDA: [
        "ayurveda", "ayurvedic", "classical formulation", "shastriya",
        "आयुर्वेद", "आयुर्वेदिक", "शास्त्रीय", "ఆయుర్వేద", "ఆయుర్వేదం", "శాస్త్రీయ",
    ],
    RegulatoryDomain.DRUG: [
        "drug", "medicine", "pharmaceutical", "dosage form",
        "दवा", "औषध", "ఔషధం", "మందు",
    ],
    RegulatoryDomain.FOOD: [
        "food", "aahara", "aahar", "dietary", "nutraceutical",
        "आहार", "भोजन", "ఆహార", "ఆహారం",
    ],
    RegulatoryDomain.COSMETIC: [
        "cosmetic", "skin cream", "topical", "external use",
        "प्रसाधन", "सौंदర్య प्रसाधन", "సౌందర్య సాధనం",
    ],
    RegulatoryDomain.BIODIVERSITY: [
        "biodiversity", "biological resource", "bioprospecting",
        "जैविक संसाधन", "जैव विविधता", "జీవ వనరులు", "జీవ వైవిధ్యం",
    ],
    RegulatoryDomain.ABS: [
        "access and benefit sharing", "abs", "nba",
        "लाभ साझाकरण", "ఎన్‌బీఏ",
    ],
    RegulatoryDomain.IMPORT_EXPORT: ["import", "export", "customs", "आयात", "निर्यात", "దిగుమతి", "ఎగుమతి"],
}

_AYURVEDA_KEYWORDS = [
    "ayurveda", "ayurvedic", "ashwagandha", "brahmi", "tulsi", "neem", "triphala",
    "churna", "aahar", "aahara", "classical text", "charaka", "sushruta",
    # Hindi
    "आयुर्वेद", "आयुर्वेदिक", "अश्वगंधा", "ब्राह्मी", "तुलसी", "नीम", "त्रिफला",
    "चूर्ण", "आहार", "चरक", "सुश्रुत",
    # Telugu
    "ఆయుర్వేద", "ఆయుర్వేదం", "ఆయుర్వేదిక్", "అశ్వగంధ", "బ్రాహ్మి", "తులసి", "వేప", "త్రిఫల",
    "చూర్ణం", "ఆహార", "చరక", "సుశ్రుత",
]
_TK_KEYWORDS = [
    "traditional knowledge", "tkdl", "folk", "indigenous", "classical formulation",
    # Hindi
    "पारंपरिक ज्ञान", "पारम्परिक ज्ञान", "टीकेडीएल", "पारंपरिक", "पारम्परिक",
    # Telugu
    "సాంప్రదాయ జ్ఞానం", "సాంప్రదాయ", "సంప్రదాయ", "సాంప్రదాయక",
]
_BIO_RESOURCE_KEYWORDS = [
    "biological resource", "plant extract", "herb", "biodiversity", "bioprospecting",
    "genetic resource", "medicinal plant",
    # Hindi
    "जैविक संसाधन", "पौधे का अर्क", "जड़ी बूटी", "जैव विविधता", "औषधीय पौधा",
    # Telugu
    "జీవ వనరులు", "మొక్కల సారం", "మూలిక", "జీవ వైవిధ్యం", "ఔషధ మొక్క",
]

_INDIA_KEYWORDS = [
    "india", "indian", "nba", "ipindia", "tkdl", "fssai", "ayush", "cgpdtm",
    # Hindi
    "भारत", "भारतीय",
    # Telugu
    "భారతదేశం", "భారతదేశంలో", "భారతీయ", "ఇండియా",
]
_DESTINATION_MARKET_PATTERNS = [
    ("united states", "United States"), ("usa", "United States"), ("u.s.", "United States"),
    ("european union", "European Union"), (" eu ", "European Union"), ("europe", "European Union"),
    ("united kingdom", "United Kingdom"), ("uk", "United Kingdom"),
    ("japan", "Japan"), ("australia", "Australia"), ("canada", "Canada"),
    ("singapore", "Singapore"), ("uae", "United Arab Emirates"), ("china", "China"),
]
_INTERNATIONAL_KEYWORDS = ["international", "wipo", "pct", "trips", "wto", "nagoya", "cbd treaty", "madrid system"]


def _match_keywords(text: str, table: Dict) -> List:
    hits = []
    for key, kws in table.items():
        if any(kw in text for kw in kws):
            hits.append(key)
    return hits


class QueryAnalyzer:
    """Stage 3.2 -- deterministic structured analysis of the normalized query."""

    def analyze(self, normalized: NormalizedQuery) -> QueryAnalysis:
        text = f"{normalized.normalized_query} {normalized.product_description or ''}".lower()

        # Intent: pick keyword table with the most hits; ties broken by table order.
        intent_scores: Dict[QueryIntent, int] = {}
        for intent, kws in _INTENT_KEYWORDS.items():
            score = sum(1 for kw in kws if kw in text)
            if score:
                intent_scores[intent] = score
        if intent_scores:
            intent = max(intent_scores.items(), key=lambda kv: kv[1])[0]
        elif len(normalized.normalized_query.split()) <= 2:
            intent = QueryIntent.UNKNOWN
        else:
            intent = QueryIntent.GENERAL_INFORMATION

        ip_domains = _match_keywords(text, _IP_DOMAIN_KEYWORDS)
        regulatory_domains = _match_keywords(text, _REGULATORY_DOMAIN_KEYWORDS)

        ayurveda_relevance = any(kw in text for kw in _AYURVEDA_KEYWORDS)
        tk_relevance = any(kw in text for kw in _TK_KEYWORDS) or intent == QueryIntent.TRADITIONAL_KNOWLEDGE
        bio_relevance = any(kw in text for kw in _BIO_RESOURCE_KEYWORDS) or intent in (
            QueryIntent.BIODIVERSITY_ACCESS, QueryIntent.ABS,
        )

        jurisdiction, jurisdiction_candidates, destination_market = self._resolve_jurisdiction(
            text, normalized.requested_jurisdiction
        )

        missing_information: List[str] = []
        if intent == QueryIntent.UNKNOWN:
            missing_information.append(
                "Query intent is unclear -- please specify what you need (e.g. patentability, "
                "trademark, regulatory classification, ABS/biodiversity access)."
            )
        if jurisdiction in ("unknown", "ambiguous"):
            missing_information.append(
                "Jurisdiction could not be confidently determined -- please specify a jurisdiction "
                "(e.g. 'India', or a destination market such as 'European Union')."
            )
        if not ip_domains and not regulatory_domains and intent != QueryIntent.GENERAL_INFORMATION:
            missing_information.append("No specific IP or regulatory domain could be identified in the query.")

        # Confidence: proportion of signals found, deterministic and explainable.
        signal_hits = sum([
            1 if intent not in (QueryIntent.UNKNOWN,) else 0,
            1 if jurisdiction not in ("unknown", "ambiguous") else 0,
            1 if ip_domains else 0,
            1 if regulatory_domains else 0,
        ])
        confidence = round(min(1.0, 0.15 + 0.20 * signal_hits + (0.05 if ayurveda_relevance else 0.0)), 2)

        return QueryAnalysis(
            intent=intent,
            jurisdiction=jurisdiction,
            jurisdiction_candidates=jurisdiction_candidates,
            ip_domains=ip_domains,
            regulatory_domains=regulatory_domains,
            ayurveda_relevance=ayurveda_relevance,
            traditional_knowledge_relevance=tk_relevance,
            biological_resource_relevance=bio_relevance,
            destination_market=destination_market,
            missing_information=missing_information,
            confidence=confidence,
        )

    @staticmethod
    def _resolve_jurisdiction(text: str, requested: str) -> Tuple[str, List[str], str | None]:
        if requested and requested.lower() not in ("auto",):
            return requested, [requested], None

        candidates: List[str] = []
        if any(kw in text for kw in _INDIA_KEYWORDS):
            candidates.append("India")
        if any(kw in text for kw in _INTERNATIONAL_KEYWORDS):
            candidates.append("International")

        destination_market = None
        for pattern, label in _DESTINATION_MARKET_PATTERNS:
            if pattern in f" {text} ":
                destination_market = label
                if label not in candidates:
                    candidates.append(label)
                break

        if len(candidates) == 0:
            # No explicit jurisdiction signal at all: this is a genuinely
            # unknown jurisdiction, not silently defaulted to India, so
            # downstream routing/abstention can ask the user to clarify.
            return "unknown", [], None
        if len(candidates) == 1:
            return candidates[0], candidates, destination_market
        # Multiple distinct jurisdictions mentioned -> never silently mix.
        return "ambiguous", candidates, destination_market
