# IP-SAKTI Sahayak – Authoritative Citation-Grounded Regulatory RAG

**Problem Statement ID:** 26045  
**Title:** IP-SAKTI Sahayak – Multilingual RAG-based, source-cited AI assistant for Intellectual Property and regulatory guidance in Ayurveda.

---

## Overview

This repository contains the local FastAPI backend for IP-SAKTI Sahayak. It supports explicit authoritative-source ingestion, versioned archives, SQLite knowledge graph provenance, Qdrant/BM25 hybrid retrieval, citation validation, confidence scoring, and safe abstention. The bundled synthetic corpus is development data only.

It provides a modular, production-ready pipeline that accepts documents across multiple formats (PDF, HTML, TXT, JSON, URLs), parses and normalizes content, extracts structured metadata, chunks along legal boundaries, generates vector embeddings, indexes points in **Qdrant**, maintains **BM25 lexical search**, performs **hybrid candidate fusion**, and applies **Top-K reranking** to return verifiable, citation-grounded evidence chunks for downstream LLM generation.

---

## Key Features

1. **Synthetic Development Corpus:** 32 comprehensive, authoritative-style synthetic legal/regulatory documents covering Patents (Section 3(p), 3(d), Phytopharmaceuticals), Geographical Indications, Trademarks, Copyright, Industrial Designs, Traditional Knowledge (TKDL), Biodiversity/ABS (Biological Diversity Act, NBA), Drug Classification (Classical, Proprietary, Phytopharmaceutical, Ayurveda-Aahar, Cosmetic), and International Frameworks (TRIPS, PCT, Nagoya Protocol, WIPO GRATK).
2. **Explicit Synthetic Markers:** All sample documents are strictly tagged with `source_type = "synthetic"` and URI prefix `synthetic://` to prevent misrepresentation as genuine legal authorities.
3. **Structured Metadata Model:** Preserves document-level and chunk-level metadata (`title`, `authority`, `jurisdiction`, `domain`, `document_type`, `effective_date`, `source_url`, `version`, `document_hash`).
4. **Structure-Aware Legal Chunker:** Splits along Chapter, Section, Article, Rule, and Clause boundaries with sliding-window fallback.
5. **Abstract Embedding Provider:** `FastEmbedProvider` (using ONNX runtime with `BAAI/bge-small-en-v1.5`) with automatic fallback to deterministic `HashEmbeddingProvider`.
6. **Qdrant Vector Database Integration:** Vector storage and payload-indexed metadata filtering across jurisdiction, domain, authority, and document type.
7. **BM25 Lexical Retrieval:** Exact-match retrieval for legal sections (e.g. `Section 3(p)`), statute names (`Biological Diversity Act`), and treaties (`PCT`, `Nagoya Protocol`).
8. **Hybrid Candidate Merger:** Weighted score normalization combining vector semantic scores and BM25 scores (`semantic_weight = 0.6`, `bm25_weight = 0.4`).
9. **Top-K Reranker:** `LightweightScoringReranker` rescores candidates using exact phrase matches, section identifier boosts, title alignment, and term density.
10. **Full Citation Traceability:** Ensures every retrieved chunk maps back to its parent document (`Claim -> Evidence Chunk -> Source Document -> Source URL`).
11. **REST APIs (FastAPI):** Complete OpenAPI endpoints for ingestion (`POST /api/v1/ingest`) and retrieval (`POST /api/v1/retrieve`, `GET /api/v1/grounding`).
12. **Benchmarking & Evaluation:** Automated benchmark suite measuring `Recall@K`, `Precision@K`, `MRR`, `Citation Correctness`, and `Latency (ms)` comparing Vector-only, BM25-only, Hybrid, and Hybrid + Reranker.

---

## Directory Structure

```text
ip_sakti_rag/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── synthetic/           # 32 Synthetic JSON & TXT test corpus files
│   └── benchmark/
├── ingestion/
│   ├── fetcher.py           # Stage 1: Load local files (JSON, TXT, HTML, PDF) & URLs
│   ├── parser.py            # Stage 2: Extract text & structural headings
│   ├── normalizer.py        # Stage 3: Clean whitespace & line breaks, preserving legal terms
│   ├── metadata.py          # Stage 4: Extract/validate metadata & compute SHA256 hashes
│   ├── chunker.py           # Stage 5: Legal structure-aware chunking
│   ├── deduplicator.py      # Deduplication via content hashes
│   ├── versioning.py        # Version management & effective dates
│   ├── validator.py         # Data validation before indexing
│   └── pipeline.py          # Full ingestion stage orchestrator
├── embeddings/
│   └── provider.py          # Abstract EmbeddingProvider interface (FastEmbed & fallback)
├── vector_store/
│   └── qdrant_store.py      # Qdrant vector database integration & payload filtering
├── retrieval/
│   ├── semantic.py          # Vector semantic search
│   ├── bm25.py              # BM25 lexical keyword search
│   ├── hybrid.py            # Weighted score normalization & candidate merger
│   ├── reranker.py          # Top-K rescoring reranker
│   └── citation.py          # Citation traceability service
├── models/
│   ├── document.py          # Document & Metadata Pydantic models
│   └── chunk.py             # Chunk, RetrievalResult, & CitationTrace models
├── api/
│   ├── ingest.py            # POST /api/v1/ingest
│   ├── query.py             # POST /api/v1/retrieve & GET /api/v1/grounding
│   └── main.py              # Main FastAPI Application
├── evaluation/
│   ├── benchmark.py         # Gold-standard benchmark queries
│   └── metrics.py           # Recall@K, Precision@K, MRR, Citation correctness metrics
├── scripts/
│   ├── generate_synthetic_data.py # Generate synthetic test corpus
│   ├── ingest_dataset.py          # Ingest corpus into Qdrant & BM25
│   └── evaluate_retrieval.py      # Run retrieval benchmark evaluation
├── tests/
│   ├── test_ingestion.py    # Ingestion unit tests
│   ├── test_retrieval.py    # Retrieval unit tests
│   └── test_api.py          # FastAPI endpoint integration tests
├── requirements.txt
├── .env.example
└── README.md
```

---

The complete flow of the Backend :

```
USER QUERY

↓

QUERY INTAKE

↓

QUERY ANALYSIS

↓

QUERY UNDERSTANDING

↓

CLASSIFICATION + ROUTING

↓

KNOWLEDGE GRAPH

↓

SEMANTIC + BM25 RETRIEVAL

↓

HYBRID FUSION

↓

RERANKING

↓

EVIDENCE PACK

↓

ABSTENTION CHECK

↓

LLM REASONING

↓

CITATION VALIDATION

↓

CONFIDENCE

↓

FINAL ANSWER

```

---

## Quick Start & Setup

### 1. Create Virtual Environment & Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Synthetic Development Corpus

```bash
python scripts/generate_synthetic_data.py
```

### Authoritative corpus

Explicit official document records live in `data/authoritative/configured_documents.json`. The ingestion command fetches only those configured URLs, verifies redirects remain on the official host, archives raw bytes and normalized text, computes hashes, indexes chunks, and writes `data/authoritative/manifests/corpus_manifest.json`:

```powershell
$env:EMBEDDING_PROVIDER="hash"  # optional deterministic local fallback
python -m scripts.ingest_authoritative_corpus
python -m scripts.validate_authoritative_corpus
```

The current manifest contains six verified official documents: four IP India documents and two WIPO traditional-knowledge documents. Drafts remain marked `draft`; guidance and WIPO technical studies are not Indian legislation. FSSAI, AYUSH, and NBA documents remain un-ingested when their document indexes were unavailable or could not be verified. Restricted sources such as TKDL remain metadata-only.

Legal chunks preserve chapter, section, subsection, rule, regulation, article, schedule, paragraph, and page context where the source text supports it. The benchmark cases in `data/benchmark/authoritative_evaluation.json` cover patents, traditional knowledge, Ayurveda, FSSAI, ABS, WIPO, current/draft versions, ambiguity, and multilingual input.

Production startup loads only successfully archived authoritative snapshots. Set `ALLOW_SYNTHETIC_CORPUS=true` only for development tests. Configure `AUTHORITATIVE_DOCUMENTS_CONFIG`, `AUTHORITATIVE_MANIFEST`, `QDRANT_LOCATION`, `QDRANT_COLLECTION`, `OLLAMA_BASE_URL`, and `OLLAMA_MODEL` through the environment.

The MVP supports English (`en`), Hindi (`hi`), and Telugu (`te`). Set
`OLLAMA_API_KEY` in `.env` for Ollama Cloud; the key is used only by the
backend and is never returned by health or query endpoints. Without a usable
LLM connection, the backend falls back to deterministic extractive synthesis.

Corpus status is available at `GET /api/v1/corpus/health`. Query responses expose source URLs, document IDs, versions, status, sections, provenance, decision checks, confidence factors, and abstention reasons. The LLM receives only the structured evidence pack and cannot create citations from model memory.

### 3. Run Ingestion Script

```bash
python scripts/ingest_dataset.py
```

### 4. Run Retrieval Benchmark Evaluation

```bash
python scripts/evaluate_retrieval.py
```

### 5. Run Unit & Integration Tests

```bash
pytest tests/
```

### 6. Launch FastAPI Server

```bash
uvicorn api.main:app --reload --port 8000
```

Open Swagger API Documentation at: `http://localhost:8000/docs`

Open `frontend/index.html` directly in a browser, or serve `frontend/` with
`python -m http.server 5173 --directory frontend`. The page sends `language`,
`jurisdiction`, and `query` to the API and provides browser-dependent
SpeechRecognition STT (`en-IN`, `hi-IN`, `te-IN`) and SpeechSynthesis TTS.
Browser support and installed voices determine whether those controls are
available. Startup loads the persisted Qdrant collection and BM25 index; it
does not ingest the corpus automatically. Use `GET /api/v1/health`,
`GET /api/v1/languages`, and `POST /api/v1/query` for a quick check.

---

## API Examples

### POST /api/v1/retrieve

Request:
```json
{
  "query": "Can I patent a traditional Ayurvedic formulation under Section 3(p)?",
  "top_k": 3,
  "jurisdiction": "India",
  "domain": "Patents",
  "retrieval_method": "hybrid",
  "use_reranker": true
}
```

Response:
```json
{
  "query": "Can I patent a traditional Ayurvedic formulation under Section 3(p)?",
  "total_candidates": 20,
  "top_k": 3,
  "results": [
    {
      "chunk_id": "SYN-PAT-001_chk_002",
      "document_id": "SYN-PAT-001",
      "score": 1.285,
      "retrieval_method": "reranked",
      "text": "Under Section 3(p) of the Patents Act, 1970, an invention which in effect is traditional knowledge or which is an aggregation or duplication of known properties of traditionally known component or components is not an patentable invention.",
      "metadata": {
        "title": "Guidelines for Patent Applications in Traditional Knowledge & Ayurveda",
        "authority": "IP-SAKTI Synthetic Patent Office",
        "jurisdiction": "India",
        "domain": "Patents",
        "document_type": "Guidance",
        "effective_date": "2025-01-15",
        "source_url": "synthetic://patent-office/guidelines-tk-ayurveda-2025",
        "source_type": "synthetic"
      },
      "citation": {
        "citation_text": "[IP-SAKTI Synthetic Patent Office] Guidelines for Patent Applications in Traditional Knowledge & Ayurveda (Guidance, India, Eff. 2025-01-15) - synthetic://patent-office/guidelines-tk-ayurveda-2025"
      }
    }
  ]
}
```
