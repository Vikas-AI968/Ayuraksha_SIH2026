"""
Ingestion Script for IP-SAKTI Sahayak.
Ingests all synthetic documents from data/synthetic/ into Qdrant vector store and BM25 index.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import bootstrap_env  # noqa: F401  # loads .env before any other project import touches os.environ

import glob
import json
import time
from ingestion.pipeline import IngestionPipeline
from vector_store.qdrant_store import QdrantVectorStore
from embeddings.provider import get_default_embedding_provider
from retrieval.bm25 import BM25Retriever


def main():
    print("=" * 70)
    print(" IP-SAKTI SAHAYAK - PHASE-1 DATASET INGESTION SCRIPT")
    print("=" * 70)

    embedding_provider = get_default_embedding_provider()
    vector_store = QdrantVectorStore(collection_name="ip_sakti_chunks", location=":memory:", embedding_provider=embedding_provider)
    bm25_retriever = BM25Retriever()
    pipeline = IngestionPipeline(dedup_strategy="update")

    synthetic_files = glob.glob("data/synthetic/*.json")
    print(f"Found {len(synthetic_files)} synthetic documents to ingest.\n")

    total_chunks = 0
    start_time = time.time()

    for idx, filepath in enumerate(synthetic_files, 1):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"[{idx}/{len(synthetic_files)}] Ingesting Document: '{data.get('title')}' ({data.get('document_id')})...")
        res = pipeline.process(data)

        if res["status"] == "success" and res["chunks"]:
            # Upsert into Qdrant
            vector_store.upsert_chunks(res["chunks"])
            # Index into BM25
            bm25_retriever.index_chunks(res["chunks"])
            total_chunks += len(res["chunks"])
            print(f"   -> Status: SUCCESS | Chunks: {len(res['chunks'])} | Time: {res['duration_seconds']:.3f}s")
        elif res["status"] == "skipped":
            print(f"   -> Status: SKIPPED ({res.get('reason')})")
        else:
            print(f"   -> Status: FAILED | Error: {res.get('error')}")

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"INSPECTED SUMMARY:")
    print(f"  Total Documents Ingested: {len(synthetic_files)}")
    print(f"  Total Chunks Indexed:     {total_chunks}")
    print(f"  Total Time Elapsed:       {elapsed:.2f} seconds")
    print("=" * 70)

if __name__ == "__main__":
    main()
