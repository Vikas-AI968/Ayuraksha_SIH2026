import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import bootstrap_env  # noqa: F401  # loads .env before any other project import touches os.environ

import os
import glob
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ingestion.pipeline import IngestionPipeline
from vector_store.qdrant_store import QdrantVectorStore, get_qdrant_settings
from embeddings.provider import get_default_embedding_provider
from retrieval.bm25 import BM25Retriever, get_bm25_index_path
from retrieval.semantic import SemanticRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import LightweightScoringReranker
from services.orchestrator import QueryOrchestrator
import api.ingest as ingest_router
import api.query as query_router
import api.classify as classify_router
import api.sources as sources_router

logger = logging.getLogger("main_app")

# Initialize global components
embedding_provider = get_default_embedding_provider()
_qdrant_location, _qdrant_collection = get_qdrant_settings()
logger.info("[QDRANT] location=%s collection=%s", _qdrant_location, _qdrant_collection)
vector_store = QdrantVectorStore(
    collection_name=_qdrant_collection,
    location=_qdrant_location,
    embedding_provider=embedding_provider,
)
bm25_retriever = BM25Retriever()
semantic_retriever = SemanticRetriever(vector_store=vector_store)
hybrid_retriever = HybridRetriever(semantic_retriever=semantic_retriever, bm25_retriever=bm25_retriever, semantic_weight=0.6, bm25_weight=0.4)
reranker = LightweightScoringReranker()
ingestion_pipeline = IngestionPipeline(registry_file=":memory:")
orchestrator = QueryOrchestrator(
    hybrid_retriever=hybrid_retriever,
    reranker=reranker,
    bm25_retriever=bm25_retriever,
)

# Wire references into router modules
ingest_router.ingestion_pipeline = ingestion_pipeline
ingest_router.vector_store = vector_store
ingest_router.bm25_retriever = bm25_retriever

query_router.semantic_retriever = semantic_retriever
query_router.bm25_retriever = bm25_retriever
query_router.hybrid_retriever = hybrid_retriever
query_router.reranker = reranker
query_router.orchestrator = orchestrator

sources_router.bm25_retriever = bm25_retriever
sources_router.vector_store = vector_store
sources_router.orchestrator = orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup is a READ operation: it LOADS whatever index the
    explicit ingestion command (`python -m scripts.ingest_authoritative_corpus`)
    already built and persisted to disk. It must never re-fetch, re-parse,
    re-chunk, or re-embed the corpus -- that is exactly the "32/869 ... 869/869"
    startup-ingestion regression this fixes. Qdrant's on-disk collection is
    already open (constructed above, module scope) with zero extra work; the
    only other in-memory structure that needs restoring is BM25, which has no
    persistence of its own, so it is pickled by the ingestion script and
    loaded verbatim here.
    """
    logger.info("[STARTUP] Loading existing indexes...")
    bm25_path = get_bm25_index_path()
    loaded = bm25_retriever.load(bm25_path)
    qdrant_points = vector_store.count_points()
    if loaded:
        logger.info(
            "[STARTUP] Loaded existing BM25 index (%d chunks) from %s; Qdrant collection '%s' has %d points. "
            "No re-ingestion performed.",
            len(bm25_retriever.chunks), bm25_path, vector_store.collection_name, qdrant_points,
        )
    else:
        logger.warning(
            "[STARTUP] No persisted BM25 index found at %s (Qdrant collection '%s' has %d points). "
            "Starting with an EMPTY retrieval index -- this is expected on a fresh checkout. "
            "Run `python -m scripts.ingest_authoritative_corpus` to build it; startup will NOT "
            "auto-ingest.", bm25_path, vector_store.collection_name, qdrant_points,
        )
        if os.environ.get("ALLOW_SYNTHETIC_CORPUS", "false").lower() in {"1", "true", "yes"}:
            synthetic_files = glob.glob("data/synthetic/*.json")
            if synthetic_files:
                logger.info(
                    "[STARTUP] ALLOW_SYNTHETIC_CORPUS=true: explicitly bootstrapping %d synthetic "
                    "documents (development/test mode only, not the authoritative corpus).",
                    len(synthetic_files),
                )
                for json_path in synthetic_files:
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        res = ingestion_pipeline.process(data)
                        if res["status"] == "success" and res["chunks"]:
                            vector_store.upsert_chunks(res["chunks"])
                            bm25_retriever.index_chunks(res["chunks"])
                    except Exception as e:
                        logger.error(f"Failed to ingest synthetic file {json_path}: {e}")
                logger.info(f"[STARTUP] Synthetic bootstrap complete. Indexed {len(bm25_retriever.chunks)} total chunks.")
    yield
    # Shutdown logic
    logger.info("Shutting down IP-SAKTI RAG Engine.")


app = FastAPI(
    title="IP-SAKTI Sahayak RAG Foundation API",
    description="Multilingual citation-grounded RAG backend ingestion and retrieval foundation for IP & Regulatory guidance in Ayurveda.",
    version="1.0.0",
    lifespan=lifespan
)

def _configured_cors_origins() -> list[str]:
    """Reads comma-separated CORS_ORIGINS from the environment. Falls back to
    a permissive wildcard ONLY if the operator has not configured anything
    (e.g. a quick local experiment), so a real deployment that sets
    CORS_ORIGINS is never silently overridden by a hardcoded '*'."""
    raw = os.environ.get("CORS_ORIGINS", "").strip()
    if not raw:
        return ["*"]
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["*"]


_cors_origins = _configured_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    # Wildcard origins and credentialed requests are mutually exclusive per
    # the CORS spec (browsers reject the combination); only allow credentials
    # once specific origins are actually configured.
    allow_credentials=_cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router.router)
app.include_router(query_router.router)
app.include_router(classify_router.router)
app.include_router(sources_router.router)


@app.get("/api/v1/languages", tags=["Languages"])
def get_supported_languages():
    """Returns supported languages for multilingual RAG, STT, and TTS."""
    return {
        "languages": [
            {
                "code": "en",
                "name": "English",
                "native_name": "English",
                "stt": True,
                "tts": True,
                "browser_dependent": True,
                "support_note": "STT/TTS availability depends on browser Web Speech API support.",
                "stt_locale": "en-IN",
                "tts_locale": "en-IN",
            },
            {
                "code": "hi",
                "name": "Hindi",
                "native_name": "हिन्दी",
                "stt": True,
                "tts": True,
                "browser_dependent": True,
                "support_note": "STT/TTS availability depends on browser Web Speech API support.",
                "stt_locale": "hi-IN",
                "tts_locale": "hi-IN",
            },
            {
                "code": "te",
                "name": "Telugu",
                "native_name": "తెలుగు",
                "stt": True,
                "tts": True,
                "browser_dependent": True,
                "support_note": "STT/TTS availability depends on browser Web Speech API support.",
                "stt_locale": "te-IN",
                "tts_locale": "te-IN",
            },
        ],
        "default": "en",
        "detection_supported": True,
    }


def _readiness_status() -> str:
    """Truthful status: 'healthy' only once the retrieval index actually has
    content; 'not_ready' if a critical component never initialized. Never
    unconditionally report healthy -- an empty index means every query would
    retrieve nothing, which is not a healthy backend."""
    if vector_store is None or bm25_retriever is None or orchestrator is None:
        return "not_ready"
    if len(bm25_retriever.chunks) == 0:
        return "degraded"
    return "healthy"


@app.get("/")
def root():
    return {
        "system": "IP-SAKTI Sahayak RAG Backend",
        "phase": "Authoritative Knowledge + Hybrid Retrieval Intelligence",
        "corpus_mode": "development_synthetic" if os.environ.get("ALLOW_SYNTHETIC_CORPUS", "false").lower() in {"1", "true", "yes"} else "authoritative",
        "indexed_chunks": len(bm25_retriever.chunks),
        "status": _readiness_status(),
        "multilingual_languages": ["en", "hi", "te"],
        "embedding_model": os.environ.get("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        "llm_provider": os.environ.get("LLM_PROVIDER", "ollama"),
    }


@app.get("/health")
@app.get("/api/v1/health")
def health():
    """Liveness/readiness endpoint with system health and index metrics."""
    return {
        "status": _readiness_status(),
        "indexed_chunks": len(bm25_retriever.chunks),
        "qdrant_collection": _qdrant_collection,
        "qdrant_location": _qdrant_location,
        "multilingual_languages": ["en", "hi", "te"],
        "embedding_model": os.environ.get("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        "llm_provider": os.environ.get("LLM_PROVIDER", "ollama"),
    }
