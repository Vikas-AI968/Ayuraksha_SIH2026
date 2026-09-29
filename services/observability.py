"""
Observability (stage 12): structured per-stage logging, plus a lightweight
SQLite-backed store for query state so GET /api/v1/query/{query_id} works.

Uses only the Python standard library (sqlite3, logging, json) -- no new
infrastructure/dependency is introduced, consistent with "Do not introduce
PostgreSQL. Do not add unnecessary databases."
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("observability")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

STAGES = [
    "QUERY_INTAKE", "QUERY_ANALYSIS", "CLASSIFICATION", "ROUTING", "RETRIEVAL",
    "RERANKING", "EVIDENCE_BUILD", "LLM_REASONING", "CITATION_VALIDATION",
    "CONFIDENCE", "ABSTENTION", "RESPONSE",
]


@dataclass
class QueryTrace:
    """Accumulates structured, non-sensitive stage timing/metrics for one query."""

    query_id: str
    session_id: str
    stages: List[Dict[str, Any]] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str, **extra: Any):
        start = time.perf_counter()
        error = None
        try:
            yield
        except Exception as e:  # re-raised after logging
            error = str(e)
            raise
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            record = {"stage": name, "latency_ms": latency_ms, **extra}
            if error:
                record["error"] = error
            self.stages.append(record)
            log_fn = logger.error if error else logger.info
            # Never log raw user query text here -- only ids/metrics.
            log_fn(f"query_id={self.query_id} stage={name} latency_ms={latency_ms} "
                   f"{'error=' + error if error else ''} {extra}")


class QueryLogStore:
    """SQLite-backed store for query_id -> final structured response + trace."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get("QUERY_LOG_DB_PATH", "data/processed/query_log.sqlite3")
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS query_log (
                query_id TEXT PRIMARY KEY,
                session_id TEXT,
                status TEXT,
                created_at REAL,
                response_json TEXT,
                trace_json TEXT
            )
            """
        )
        self._conn.commit()

    def save(self, query_id: str, session_id: str, status: str, response: Dict[str, Any], trace: List[Dict[str, Any]]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO query_log (query_id, session_id, status, created_at, response_json, trace_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (query_id, session_id, status, time.time(), json.dumps(response), json.dumps(trace)),
        )
        self._conn.commit()

    def get(self, query_id: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT query_id, session_id, status, created_at, response_json, trace_json FROM query_log WHERE query_id = ?",
            (query_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "query_id": row[0],
            "session_id": row[1],
            "status": row[2],
            "created_at": row[3],
            "response": json.loads(row[4]),
            "trace": json.loads(row[5]),
        }

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM query_log")
        return cur.fetchone()[0]


_default_store: Optional[QueryLogStore] = None


def get_default_query_log_store() -> QueryLogStore:
    global _default_store
    if _default_store is None:
        _default_store = QueryLogStore()
    return _default_store
