"""Deterministic hashing helpers used across ingestion, embeddings, and the vector store.

Python's built-in ``hash()`` is intentionally NOT used anywhere in this codebase for
anything that must be stable across process restarts. As of Python 3.3, str/bytes
hashing is randomized per-process (``PYTHONHASHSEED``) for security (hash-flooding
protection), so ``hash("some text")`` returns a *different* integer every time the
interpreter starts. That is fine for in-memory dict/set lookups within a single
process, but it is fatal for:

  * Qdrant point IDs -- computed once at ingestion time and again at query/upsert
    time in a completely different ``python -m uvicorn`` process. If the IDs drift,
    re-running ingestion silently creates duplicate points instead of upserting.
  * The deterministic hash-embedding fallback -- the ingestion process and the API
    process must derive bit-for-bit identical vectors for the same text, or
    retrieval silently returns garbage after a restart.

Everything below is backed by ``hashlib.sha256``, which is stable across processes,
machines, and Python versions. Centralized here so no call site re-implements (and
potentially re-breaks) this logic.
"""
from __future__ import annotations

import hashlib
import uuid

# Fixed, deterministically-derived namespace (never random) so uuid5() output only
# ever depends on the input string. This value must never change once data has
# been ingested -- doing so would change every point ID and orphan existing
# Qdrant points on next upsert.
POINT_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "ip-sakti-sahayak.local")


def stable_point_id(chunk_id: str) -> str:
    """Deterministic Qdrant point ID for a given chunk_id.

    Returns a UUID string (a valid Qdrant ExtendedPointId) that is identical for
    the same ``chunk_id`` across processes, restarts, and machines, so repeated
    ingestion safely upserts in place instead of creating duplicate points.
    """
    return str(uuid.uuid5(POINT_ID_NAMESPACE, chunk_id))


def stable_document_id(*parts: str) -> str:
    """Deterministic short document ID derived from arbitrary text.

    Used only where the caller has no explicit ``document_id`` (e.g. raw text or
    ad-hoc URL ingestion) and previously relied on the process-dependent
    ``hash()`` builtin.
    """
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return f"doc_{digest[:12]}"


def stable_bucket_hash(token: str) -> int:
    """Deterministic non-negative integer hash of a token.

    Used to bucket tokens into a fixed-size vector (the hash-embedding fallback
    provider). Replaces the process-dependent ``hash()`` builtin so the same
    token always lands in the same bucket with the same sign/magnitude,
    regardless of which process computed it.
    """
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
