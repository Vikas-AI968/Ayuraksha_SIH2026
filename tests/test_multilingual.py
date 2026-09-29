"""Tests for Multilingual Query Pipeline, Retrieval, and Capabilities (English, Hindi, Telugu)."""
from models.query import QueryIntakeRequest, QueryIntent
from services.query_analyzer import QueryAnalyzer, QueryIntakeService, detect_language
from retrieval.query_expansion import extract_provisions, build_search_signals
from models.answer import ReasoningOutput, KeyPoint
from models.evidence import EvidenceItem, EvidencePack
from services.citation_validator import CitationValidator
from fastapi.testclient import TestClient
from api.main import app


def test_language_detection():
    # English
    assert detect_language("Can traditional Ayurvedic knowledge be patented in India?") == "en"
    assert detect_language("What is Section 3(p) of the Patents Act?") == "en"

    # Hindi (Devanagari)
    assert detect_language("क्या पारंपरिक आयुर्वेदिक ज्ञान का भारत में पेटेंट कराया जा सकता है?") == "hi"
    assert detect_language("पेटेंट अधिनियम की धारा 3(p) क्या है?") == "hi"

    # Telugu
    assert detect_language("సాంప్రదాయ ఆయుర్వేద జ్ఞానానికి భారతదేశంలో పేటెంట్ పొందవచ్చా?") == "te"
    assert detect_language("పేటెంట్ చట్టంలోని సెక్షన్ 3(p) ఏమిటి?") == "te"


def test_query_intake_explicit_language():
    intake_svc = QueryIntakeService()

    req_en = QueryIntakeRequest(query="Can I patent this formulation?", language="en")
    norm_en = intake_svc.intake(req_en)
    assert norm_en.language == "en"

    req_hi = QueryIntakeRequest(query="क्या पारंपरिक ज्ञान का पेटेंट हो सकता है?", language="auto")
    norm_hi = intake_svc.intake(req_hi)
    assert norm_hi.language == "hi"

    req_te = QueryIntakeRequest(query="సాంప్రదాయ ఆయుర్వేదం", language="auto")
    norm_te = intake_svc.intake(req_te)
    assert norm_te.language == "te"

    # Explicit override
    req_override = QueryIntakeRequest(query="Ashwagandha formulation in India", language="hi")
    norm_override = intake_svc.intake(req_override)
    assert norm_override.language == "hi"


def test_hindi_query_analysis():
    intake_svc = QueryIntakeService()
    analyzer = QueryAnalyzer()

    req = QueryIntakeRequest(query="क्या पारंपरिक आयुर्वेदिक ज्ञान का भारत में पेटेंट कराया जा सकता है?")
    norm = intake_svc.intake(req)
    analysis = analyzer.analyze(norm)

    assert analysis.intent == QueryIntent.PATENTABILITY
    assert analysis.jurisdiction == "India"
    assert analysis.ayurveda_relevance is True
    assert analysis.traditional_knowledge_relevance is True


def test_telugu_query_analysis():
    intake_svc = QueryIntakeService()
    analyzer = QueryAnalyzer()

    req = QueryIntakeRequest(query="సాంప్రదాయ ఆయుర్వేద జ్ఞానానికి భారతదేశంలో పేటెంట్ పొందవచ్చా?")
    norm = intake_svc.intake(req)
    analysis = analyzer.analyze(norm)

    assert analysis.intent == QueryIntent.PATENTABILITY
    assert analysis.jurisdiction == "India"
    assert analysis.ayurveda_relevance is True
    assert analysis.traditional_knowledge_relevance is True


def test_multilingual_provision_extraction():
    # English
    en_provisions = extract_provisions("What does Section 3(p) of the Patents Act say?")
    assert "Section 3(p)" in en_provisions

    # Hindi धारा 3(p)
    hi_provisions = extract_provisions("पेटेंट अधिनियम की धारा 3(p) के तहत क्या नियम हैं?")
    assert "Section 3(p)" in hi_provisions

    # Telugu సెక్షన్ 3(p)
    te_provisions = extract_provisions("పేటెంట్ చట్టంలోని సెక్షన్ 3(p) ఏమిటి?")
    assert "Section 3(p)" in te_provisions


def test_multilingual_search_signals_generation():
    intake_svc = QueryIntakeService()
    analyzer = QueryAnalyzer()

    # Hindi query
    req_hi = QueryIntakeRequest(query="क्या पारंपरिक आयुर्वेदिक ज्ञान का भारत में पेटेंट कराया जा सकता है?")
    norm_hi = intake_svc.intake(req_hi)
    analysis_hi = analyzer.analyze(norm_hi)
    signals_hi = build_search_signals(norm_hi.normalized_query, analysis_hi)

    # Should contain cross-lingual English search terms for BM25/semantic retrieval
    assert any("traditional knowledge" in s.lower() for s in signals_hi)
    assert any("ayurvedic" in s.lower() for s in signals_hi)
    assert any("patent" in s.lower() for s in signals_hi)

    # Telugu query
    req_te = QueryIntakeRequest(query="సాంప్రదాయ ఆయుర్వేద జ్ఞానానికి భారతదేశంలో పేటెంట్ పొందవచ్చా?")
    norm_te = intake_svc.intake(req_te)
    analysis_te = analyzer.analyze(norm_te)
    signals_te = build_search_signals(norm_te.normalized_query, analysis_te)

    assert any("traditional knowledge" in s.lower() for s in signals_te)
    assert any("ayurvedic" in s.lower() for s in signals_te)
    assert any("patent" in s.lower() for s in signals_te)


def test_languages_endpoint():
    with TestClient(app) as client:
        resp = client.get("/api/v1/languages")
        assert resp.status_code == 200
        data = resp.json()
        assert "languages" in data
        codes = [lang["code"] for lang in data["languages"]]
        assert "en" in codes
        assert "hi" in codes
        assert "te" in codes
        assert all(lang["stt"] and lang["tts"] for lang in data["languages"])


def test_citation_validator_multilingual_support():
    validator = CitationValidator()
    evidence_item = EvidenceItem(
        evidence_id="E001",
        chunk_id="chunk_1",
        document_id="IPINDIA-PATENTS-ACT-1970",
        source_id="ip-india",
        title="The Patents Act, 1970",
        authority="Office of the Controller General of Patents, Designs & Trade Marks (IP India)",
        jurisdiction="India",
        section="3(p)",
        source_type="statute",
        text="an invention which in effect, is traditional knowledge or which is an aggregation or duplication of known properties of traditionally known component or components.",
        source_url="https://ipindia.gov.in/patents-act.pdf",
    )
    evidence_pack = EvidencePack(query="query", items=[evidence_item])

    # Keypoint in Hindi citing E001
    reasoning_hi = ReasoningOutput(
        answer="भारतीय पेटेंट अधिनियम 1970 की धारा 3(p) के अनुसार पारंपरिक ज्ञान पेटेंट योग्य नहीं है।",
        key_points=[
            KeyPoint(
                point="धारा 3(p) के तहत पारंपरिक ज्ञान या उसके घटकों का दोहराव पेटेंट योग्य नहीं है।",
                evidence_ids=["E001"]
            )
        ]
    )

    validation_result = validator.validate(reasoning_hi, evidence_pack)
    assert validation_result.citation_coverage > 0.0
    assert validation_result.claims[0].supported is True
