# Implementation Plan — IP-SAKTI Sahayak (SIH26045) MVP

Make the existing authoritative RAG backend fully functioning end-to-end with Ollama Cloud, multilingual support (English, Hindi, Telugu), FastEmbed multilingual embeddings, STT/TTS, and a minimal HTML/CSS/JS test frontend.

## Proposed Architecture & Workflow

```
User Query (EN / HI / TE via Text or Browser STT)
        ↓
Query Intake & Language Detection (Lightweight: Devanagari -> HI, Telugu -> TE, Latin -> EN)
        ↓
Deterministic Analysis & Hindi/Telugu Keywords (Intent, Jurisdiction, Ayurveda/TK relevance)
        ↓
Multilingual Query Expansion & Cross-Language Retrieval Representation
        ↓
Exact Legal Provision Boost (Supports Section, धारा, సెక్షన్)
        ↓
Hybrid Retrieval (Multilingual FastEmbed MiniLM-L12 + BM25 Lexical)
        ↓
Lightweight Reranking & Evidence Pack Assembly
        ↓
Pre-Reasoning Abstention Check (Feature, not failure)
        ↓
Ollama Cloud LLM Reasoning (gpt-oss:120b-cloud in user's language, faithful citations)
        ↓
Multilingual Citation Validation & Coverage
        ↓
Post-Reasoning Abstention & Gated Deterministic Confidence
        ↓
Response Payload (includes language, answer, citations, evidence, confidence, warnings)
        ↓
Minimal Frontend (Displays Answer, Citations, Confidence + Browser TTS [🔊 Speak Answer])
```

## User Review Required

> [!IMPORTANT]
> **Embedding Model Transition**: FastEmbed's proven multilingual model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions, quantized ONNX) is already verified and cached on this machine. We will rebuild the authoritative Qdrant collection cleanly using this multilingual model so that Hindi, Telugu, and English queries share a single cross-lingual semantic embedding space. All raw PDFs are cached locally in `data/authoritative/raw/`, so re-ingestion will take ~15-30 seconds with no external network required.

> [!NOTE]
> **Ollama Cloud Configuration**: We verified `https://ollama.com/api/chat` with model `gpt-oss:120b-cloud` using the API key provided in `.env.example`. The response status is 200 OK. We will ensure `.env` is populated with this configuration while keeping secrets safe from frontend exposure or health endpoints.

## Proposed Changes

### 1. Configuration & Dependencies
- `.env` & `.env.example`: Set `EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, `LLM_PROVIDER=ollama`, `OLLAMA_BASE_URL=https://ollama.com`, `OLLAMA_MODEL=gpt-oss:120b-cloud`, `OLLAMA_API_KEY`. Clean up stale Anthropic references.

---

### 2. Embeddings & Ingestion
#### [MODIFY] [embeddings/provider.py](file:///d:/BABA/embeddings/provider.py)
- Change default model to `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Dynamically detect dimension from FastEmbed model rather than hardcoding.
- Re-run `python -m scripts.ingest_authoritative_corpus` to populate the Qdrant collection and BM25 index with multilingual embeddings.

---

### 3. Multilingual Query Pipeline & Analysis
#### [MODIFY] [services/query_analyzer.py](file:///d:/BABA/services/query_analyzer.py)
- Extend `detect_language` to support Telugu script (`\u0C00-\u0C7F`) alongside Devanagari (`\u0900-\u097F`) and English Latin script.
- Add Hindi and Telugu keyword tables for intent (patentability, traditional knowledge, biodiversity/ABS, regulatory classification), domains, Ayurveda keywords, and India jurisdiction.
- Enable deterministic analysis to correctly identify intent, jurisdiction, and Ayurveda/TK relevance for Hindi and Telugu queries.

#### [MODIFY] [retrieval/query_expansion.py](file:///d:/BABA/retrieval/query_expansion.py)
- Expand `_PROVISION_RE` and `extract_provisions` to match Hindi terms ("धारा", "दफ़ा") and Telugu terms ("సెక్షన్", "నిబంధన"), rendering them to canonical "Section X(y)".
- In `build_search_signals`, add multilingual concept expansion: if the query is in Hindi or Telugu, include mapped English concept terms so BM25 and semantic search find the English legal text directly.

---

### 4. Orchestration, Reasoning & Validation
#### [MODIFY] [models/answer.py](file:///d:/BABA/models/answer.py)
- Add `language: str = "en"` to `QueryResponse`.

#### [MODIFY] [services/orchestrator.py](file:///d:/BABA/services/orchestrator.py)
- Optimize product classification: only run when explicit `product_description` is provided or intent is regulatory/compliance sensitive, avoiding unnecessary overhead for pure patentability questions.
- Pass resolved language to the reasoning engine and populate `language` in `QueryResponse`.

#### [MODIFY] [services/reasoning.py](file:///d:/BABA/services/reasoning.py)
- Update `SYSTEM_PROMPT` and `_build_user_prompt` with language directives: respond in the user's requested language (Hindi, Telugu, or English) while strictly preserving official document names, statute titles, and section numbers in their authentic authoritative form.

#### [MODIFY] [services/citation_validator.py](file:///d:/BABA/services/citation_validator.py)
- Update `_tokenize` to recognize unicode word characters across scripts (`\w+`).
- Add robust claim validation for multilingual answers against English evidence: check cited evidence IDs against the evidence pack, statutory provisions, and numerical/source identifiers so non-English claims are reliably verified.

#### [MODIFY] [services/confidence.py](file:///d:/BABA/services/confidence.py)
- Ensure low retrieval quality bounds confidence appropriately so irrelevant documents cannot produce high confidence.

#### [MODIFY] [services/llm_provider.py](file:///d:/BABA/services/llm_provider.py)
- Remove stale Anthropic provider. Keep Ollama Cloud as primary and Extractive as zero-network fallback.

---

### 5. API Endpoints
#### [MODIFY] [api/main.py](file:///d:/BABA/api/main.py)
- Add `GET /api/v1/languages` endpoint returning supported MVP languages (`en`, `hi`, `te`) with STT/TTS metadata.
- Ensure health endpoint exposes multilingual status and loaded index stats without leaking secrets.

---

### 6. Minimal Frontend & Speech (STT / TTS)
#### [NEW] [frontend/index.html](file:///d:/BABA/frontend/index.html)
- Clean, minimal test interface: Language dropdown (`auto`, `en`, `hi`, `te`), Jurisdiction dropdown, query textarea, `[🎤 Speak]` microphone button for STT, `[Ask IP-SAKTI]` button, quick sample queries (EN, HI, TE), Answer section with `[🔊 Speak Answer]` and `[⏹ Stop]` buttons, Confidence badge, Citation Coverage progress, Citations list, collapsible Evidence items, and Warnings/Abstention alerts.

#### [NEW] [frontend/style.css](file:///d:/BABA/frontend/style.css)
- Lightweight, responsive styling with clean modern aesthetics, accessible status colors, and readable typography.

#### [NEW] [frontend/app.js](file:///d:/BABA/frontend/app.js)
- Communicates with `http://127.0.0.1:8000`.
- Implements Web Speech API (`SpeechRecognition` / `webkitSpeechRecognition`) with language awareness (`en-IN`, `hi-IN`, `te-IN`).
- Implements SpeechSynthesis API for TTS, selecting voices matching response language.
- Renders answers, confidence, citations, evidence, and abstentions clearly.

---

### 7. Tests & Verification
#### [NEW] [tests/test_multilingual.py](file:///d:/BABA/tests/test_multilingual.py)
- Tests for language detection (EN, HI, TE).
- Tests for Hindi/Telugu query analysis (intent, jurisdiction, Ayurveda relevance).
- Tests for Hindi ("धारा 3(p)") and Telugu ("సెక్షన్ 3(p)") provision extraction.
- Tests for multilingual retrieval of English evidence (Patents Act Section 3(p)).
- Tests for `GET /api/v1/languages`.
- Tests for citation validation on multilingual outputs.

#### [MODIFY] [data/benchmark/authoritative_evaluation.json](file:///d:/BABA/data/benchmark/authoritative_evaluation.json)
- Update `multilingual-01` and add multilingual benchmark questions for Hindi and Telugu.

---

## Verification Plan

### Automated Tests
- `python -m compileall -q .`
- `pytest -q` (ensure all 49 existing tests continue to pass + new multilingual tests pass)
- Run `python scripts/smoke_test.py`
- Run `python -m scripts.evaluate_authoritative`

### Manual Verification
- Start FastAPI backend: `uvicorn api.main:app --host 127.0.0.1 --port 8000`
- Check `GET /health` and `GET /api/v1/languages`
- Start minimal frontend on `http://127.0.0.1:5173`
- Execute queries:
  1. English: "Can traditional Ayurvedic knowledge be patented in India?"
  2. Hindi: "क्या पारंपरिक आयुर्वेदिक ज्ञान का भारत में पेटेंट कराया जा सकता है?"
  3. Telugu: "సాంప్రదాయ ఆయుర్వేద జ్ఞానానికి భారతదేశంలో పేటెంట్ పొందవచ్చా?"
  4. Explicit Section: "What is Section 3(p) of the Patents Act?"
  5. Hindi Section: "पेटेंट अधिनियम की धारा 3(p) क्या है?"
  6. Telugu Section: "పేటెంట్ చట్టంలోని సెక్షన్ 3(p) ఏమిటి?"
  7. Abstention test: "What FSSAI license is required for this product?"
- Test STT listening and transcription.
- Test TTS answer speech and stop button.
- Verify persistence after backend restart without re-ingesting.
