"""
BM25 Lexical Retriever for IP-SAKTI Sahayak.
Provides keyword-exact retrieval for legal citations, section numbers, statute names, and technical terms.
"""
import logging
import os
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from models.chunk import Chunk, RetrievalResult
from models.document import DocumentMetadata
from ingestion.paths import resolve_repo_path

logger = logging.getLogger("bm25_retriever")

# BM25Okapi has no persistence of its own and rebuilding it means re-tokenizing
# every chunk's text. That is cheap in isolation, but the spec (SIH26045
# stabilization) draws a hard line: API startup must LOAD an existing index,
# never REBUILD one from source material -- so this index is pickled to disk
# by the ingestion script and loaded verbatim at API startup instead.
DEFAULT_BM25_INDEX_PATH = "data/processed/bm25_index.pkl"


def get_bm25_index_path() -> str:
    """Mirrors vector_store.qdrant_store.get_qdrant_settings(): a single
    source of truth for where the persisted BM25 index lives, so ingestion
    and the API can never silently drift onto different files. Tests
    override BM25_INDEX_PATH to a path that never exists, keeping them
    isolated from whatever the real ingestion script has persisted."""
    return os.environ.get("BM25_INDEX_PATH", DEFAULT_BM25_INDEX_PATH)


class BM25Retriever:
    """Lexical keyword index supporting structured metadata filtering and BM25 scoring."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25_index: Optional[BM25Okapi] = None

    def _tokenize(self, text: str) -> List[str]:
        # Extract alphanumeric words and section identifiers
        tokens = re.findall(r"\w+", text.lower())
        return tokens

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Indexes chunks for BM25 search."""
        self.chunks.extend(chunks)
        for chunk in chunks:
            # Combine text, title, section, and authority for rich lexical indexing
            full_text = f"{chunk.metadata.title} {chunk.metadata.authority} {chunk.section or ''} {chunk.text}"
            tokens = self._tokenize(full_text)
            self.corpus_tokens.append(tokens)

        if self.corpus_tokens:
            self.bm25_index = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 10, **filters) -> List[RetrievalResult]:
        """Performs BM25 search with metadata filtering."""
        if not self.bm25_index or not self.chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25_index.get_scores(query_tokens)

        # Filter candidate chunks based on metadata filters
        results = []
        for idx, score in enumerate(scores):
            # Include all non-negative candidates matching token overlap
            chunk = self.chunks[idx]
            meta = chunk.metadata

            # Apply metadata filters
            if filters.get("jurisdiction") and meta.jurisdiction != filters["jurisdiction"]:
                continue
            if filters.get("domain") and meta.domain != filters["domain"]:
                continue
            if filters.get("authority") and meta.authority != filters["authority"]:
                continue
            if filters.get("document_type") and meta.document_type != filters["document_type"]:
                continue
            if filters.get("status") and meta.status != filters["status"]:
                continue
            if filters.get("source_priority_min") is not None and meta.source_priority < filters["source_priority_min"]:
                continue
            if filters.get("authoritative") is not None and meta.authoritative != filters["authoritative"]:
                continue

            # Check if any query token overlaps corpus token
            token_overlap = set(query_tokens).intersection(set(self.corpus_tokens[idx]))
            if not token_overlap and score <= 0:
                continue

            adjusted_score = float(score) if score > 0 else float(len(token_overlap))

            res = RetrievalResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                score=adjusted_score,
                bm25_score=adjusted_score,
                retrieval_method="bm25",
                text=chunk.text,
                metadata=meta
            )
            res.build_citation()
            results.append(res)

        # Sort by BM25 score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        self.chunks = []
        self.corpus_tokens = []
        self.bm25_index = None

    def save(self, path: Optional[str] = None) -> None:
        """Persists the built index (chunks + tokens + BM25Okapi state) to disk.
        Called by the explicit ingestion script -- never by API startup."""
        target = resolve_repo_path(path or get_bm25_index_path())
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump({"chunks": self.chunks, "corpus_tokens": self.corpus_tokens, "bm25_index": self.bm25_index}, f)
        logger.info("Saved BM25 index (%d chunks) -> %s", len(self.chunks), target)

    def load(self, path: Optional[str] = None) -> bool:
        """Loads a previously-saved index from disk. Returns False (leaving
        this retriever empty) if no saved index exists yet -- callers must
        NOT fall back to rebuilding from source documents; that is the
        ingestion script's job, run explicitly."""
        target = resolve_repo_path(path or get_bm25_index_path())
        if not target.is_file():
            return False
        with open(target, "rb") as f:
            state = pickle.load(f)
        self.chunks = state["chunks"]
        self.corpus_tokens = state["corpus_tokens"]
        self.bm25_index = state["bm25_index"]
        logger.info("Loaded existing BM25 index (%d chunks) <- %s", len(self.chunks), target)
        return True
