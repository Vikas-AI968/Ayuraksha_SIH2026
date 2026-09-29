"""SQLite ledger for preserving document versions across ingestion runs."""
from __future__ import annotations

import os
import sqlite3
from typing import Optional

from models.document import DocumentMetadata


class DocumentVersionStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get("DOCUMENT_VERSION_DB_PATH", "data/processed/document_versions.sqlite3")
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS document_versions (
                document_id TEXT NOT NULL, title TEXT NOT NULL, authority TEXT NOT NULL,
                version TEXT NOT NULL, effective_date TEXT NOT NULL, status TEXT NOT NULL,
                content_hash TEXT, parent_document_id TEXT, metadata_json TEXT NOT NULL,
                PRIMARY KEY (document_id, version, effective_date)
            )"""
        )
        self._connection.commit()

    def find_related(self, title: str, authority: str) -> list[DocumentMetadata]:
        rows = self._connection.execute(
            "SELECT metadata_json FROM document_versions WHERE title = ? AND authority = ? ORDER BY effective_date",
            (title, authority),
        ).fetchall()
        return [DocumentMetadata.model_validate_json(row[0]) for row in rows]

    def record(self, document_id: str, metadata: DocumentMetadata) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO document_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (document_id, metadata.title, metadata.authority, metadata.version, metadata.effective_date,
             metadata.status, metadata.document_hash, metadata.parent_document_id, metadata.model_dump_json()),
        )
        self._connection.commit()

    def all_for(self, document_id: str) -> list[DocumentMetadata]:
        rows = self._connection.execute(
            "SELECT metadata_json FROM document_versions WHERE document_id = ? ORDER BY effective_date",
            (document_id,),
        ).fetchall()
        return [DocumentMetadata.model_validate_json(row[0]) for row in rows]
