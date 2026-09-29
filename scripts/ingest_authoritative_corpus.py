"""Ingest explicitly configured official documents into the local corpus."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import bootstrap_env  # noqa: F401  # loads .env before any other project import touches os.environ

import json
import os

from embeddings.provider import get_default_embedding_provider
from ingestion.authoritative import AuthoritativeCorpusService
from ingestion.pipeline import IngestionPipeline
from retrieval.bm25 import BM25Retriever
from services.knowledge_graph import KnowledgeGraph
from sources.registry import SourceRegistry
from vector_store.qdrant_store import QdrantVectorStore, get_qdrant_settings


def main() -> int:
    registry = SourceRegistry()
    if not registry.documents():
        print("No configured authoritative documents found.")
        print("Set AUTHORITATIVE_DOCUMENTS_CONFIG to a JSON document registry.")
        return 2
    provider = get_default_embedding_provider()
    location, collection = get_qdrant_settings()
    print(f"[QDRANT] location={location} collection={collection}")
    vector_store = QdrantVectorStore(collection_name=collection, location=location, embedding_provider=provider)
    bm25 = BM25Retriever()
    graph = KnowledgeGraph()
    service = AuthoritativeCorpusService(
        IngestionPipeline(dedup_strategy="update"), vector_store=vector_store,
        bm25_retriever=bm25, graph=graph,
    )
    entries = service.ingest_configured(registry)
    bm25.save()
    print("Authoritative corpus ingestion")
    print("------------------------------")
    print(f"Documents configured: {len(entries)}")
    print(f"Documents ingested: {sum(entry.fetch_status == 'ingested' for entry in entries)}")
    print(f"Documents failed: {sum(entry.fetch_status == 'failed' for entry in entries)}")
    print(f"Chunks created: {sum(entry.chunk_count for entry in entries)}")
    print(f"BM25 documents: {len(bm25.chunks)}")
    print(f"BM25 index persisted: data/processed/bm25_index.pkl")
    print(f"Failures: {json.dumps([entry.model_dump() for entry in entries if entry.fetch_status == 'failed'], indent=2)}")
    return 0 if all(entry.fetch_status == "ingested" for entry in entries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
