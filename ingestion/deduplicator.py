"""
Deduplicator Module - Prevents duplicate document indexing.
"""
from typing import Dict, Any, Optional
import os
import json


class Deduplicator:
    """Tracks ingested document content hashes and handles deduplication."""

    def __init__(self, registry_file: Optional[str] = "data/processed/hash_registry.json"):
        self.registry_file = registry_file
        self.registry: Dict[str, Dict[str, Any]] = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        if self.registry_file and self.registry_file != ":memory:" and os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_registry(self) -> None:
        if not self.registry_file or self.registry_file == ":memory:":
            return
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2)

    def check_duplicate(self, doc_id: str, doc_hash: str) -> Optional[str]:
        """
        Returns:
        - 'exact_duplicate' if same doc_id and same content hash
        - 'hash_collision' if different doc_id has exact same content hash
        - None if unique
        """
        if doc_id in self.registry:
            existing_hash = self.registry[doc_id].get("doc_hash")
            if existing_hash == doc_hash:
                return "exact_duplicate"

        for existing_id, meta in self.registry.items():
            if meta.get("doc_hash") == doc_hash and existing_id != doc_id:
                return "hash_collision"

        return None

    def register(self, doc_id: str, doc_hash: str, version: str, effective_date: str) -> None:
        self.registry[doc_id] = {
            "doc_hash": doc_hash,
            "version": version,
            "effective_date": effective_date
        }
        self._save_registry()

    def clear(self) -> None:
        self.registry = {}
        if self.registry_file and self.registry_file != ":memory:" and os.path.exists(self.registry_file):
            os.remove(self.registry_file)
