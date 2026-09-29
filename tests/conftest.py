"""Test mode explicitly opts into the synthetic regression corpus.

Also isolates the vector store location: without this, tests would upsert
into the SAME persistent ./data/qdrant directory the real API/ingestion
script uses (vector_store/qdrant_store.py's DEFAULT_QDRANT_LOCATION), silently
polluting the authoritative corpus with test documents on every `pytest` run.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import bootstrap_env  # noqa: F401  # loads .env (explicit overrides below still win)

import os

os.environ["ALLOW_SYNTHETIC_CORPUS"] = "true"
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["QDRANT_LOCATION"] = ":memory:"
os.environ["QDRANT_COLLECTION"] = "ip_sakti_test"
# A path that never exists, so BM25Retriever.load() always returns False in
# tests and falls back to the explicit synthetic-corpus bootstrap below --
# never silently loading whatever the real ingestion script has persisted.
os.environ["BM25_INDEX_PATH"] = "data/processed/_test_bm25_index_never_persisted.pkl"
