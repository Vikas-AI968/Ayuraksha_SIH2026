"""Explicitly configured source adapters; no arbitrary website scraping."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from sources.adapters.base import BaseSourceAdapter, SourceAdapterNotConfigured


class LocalSourceAdapter(BaseSourceAdapter):
    """Loads operator-supplied JSON documents from a configured local path."""

    def __init__(self, source_id: str, paths: Iterable[str]):
        self.source_id = source_id
        self.paths = [Path(path) for path in paths]

    def fetch_documents(self) -> Iterable[dict[str, Any]]:
        import json

        found = False
        for path in self.paths:
            if not path.is_file():
                continue
            found = True
            if path.suffix.lower() != ".json":
                raise ValueError(f"Configured source file must be JSON: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            documents = payload if isinstance(payload, list) else [payload]
            for document in documents:
                if not isinstance(document, dict):
                    raise ValueError(f"Configured source document must be an object: {path}")
                document.setdefault("source_id", self.source_id)
                document.setdefault("source_type", "official")
                document.setdefault("authority_level", "primary")
                yield document
        if not found:
            raise SourceAdapterNotConfigured(
                f"No configured local documents were found for source '{self.source_id}'."
            )


class RestrictedSourceAdapter(BaseSourceAdapter):
    """Metadata-only adapter for restricted sources such as TKDL."""

    def __init__(self, source_id: str):
        self.source_id = source_id

    def fetch_documents(self) -> Iterable[dict[str, Any]]:
        raise SourceAdapterNotConfigured(
            f"Source '{self.source_id}' is restricted; only legitimately supplied metadata may be registered."
        )
