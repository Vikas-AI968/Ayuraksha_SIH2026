"""Small SQLite-backed knowledge graph for local development and tests."""
from __future__ import annotations

import json
import os
import sqlite3
import re
from typing import Iterable, Optional

from models.knowledge import KnowledgeDocument, KnowledgeEntity, KnowledgeGraphContext, KnowledgeRelationship


class KnowledgeGraph:
    """Graph interface backed by SQLite; the query layer does not depend on SQLite details."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get("KNOWLEDGE_DB_PATH", "data/processed/knowledge.sqlite3")
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._ensure_schema()
        self._seed_reference_taxonomy()

    def _ensure_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS knowledge_entities (
                entity_id TEXT PRIMARY KEY, name TEXT NOT NULL, entity_type TEXT NOT NULL,
                description TEXT, metadata_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS knowledge_documents (
                document_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS knowledge_relationships (
                relationship_id TEXT PRIMARY KEY, source_entity_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL, target_entity_id TEXT NOT NULL,
                source_document_id TEXT, evidence_id TEXT, metadata_json TEXT NOT NULL
            );
            """
        )
        self._connection.commit()

    def _seed_reference_taxonomy(self) -> None:
        """Seed concept relationships only; this is not a substitute for source ingestion."""
        if self._connection.execute("SELECT 1 FROM knowledge_entities LIMIT 1").fetchone():
            return
        entities = [
            KnowledgeEntity(entity_id="patents-act", name="Patents Act", entity_type="statute"),
            KnowledgeEntity(entity_id="section-3p", name="Section 3(p)", entity_type="provision"),
            KnowledgeEntity(entity_id="traditional-knowledge", name="Traditional Knowledge", entity_type="concept"),
            KnowledgeEntity(entity_id="tkdl", name="TKDL", entity_type="database_registry"),
            KnowledgeEntity(entity_id="ayush-guidance", name="AYUSH patent guidance", entity_type="guideline"),
            KnowledgeEntity(entity_id="classical-medicine", name="Classical Medicine", entity_type="product_category"),
            KnowledgeEntity(entity_id="biological-resource", name="Biological Resource", entity_type="concept"),
            KnowledgeEntity(entity_id="nba", name="National Biodiversity Authority", entity_type="authority"),
            KnowledgeEntity(entity_id="ayurveda-aahara", name="Ayurveda Aahara", entity_type="product_category"),
            KnowledgeEntity(entity_id="fssai-regulations", name="FSSAI Regulations", entity_type="regulation"),
        ]
        relationships = [
            ("patents-act", "contains", "section-3p"),
            ("section-3p", "concerns", "traditional-knowledge"),
            ("traditional-knowledge", "related_to", "tkdl"),
            ("ayush-guidance", "interprets_or_implements", "patents-act"),
            ("classical-medicine", "may_trigger", "traditional-knowledge"),
            ("biological-resource", "may_require", "nba"),
            ("ayurveda-aahara", "regulated_by", "fssai-regulations"),
        ]
        self.add_many(
            entities,
            [KnowledgeRelationship(
                relationship_id=f"taxonomy-{source}-{relation}-{target}",
                source_entity_id=source, relationship_type=relation, target_entity_id=target,
                metadata={"origin": "reference_taxonomy", "authoritative_content_ingested": False},
            ) for source, relation, target in relationships],
        )

    def add_entity(self, entity: KnowledgeEntity) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO knowledge_entities VALUES (?, ?, ?, ?, ?)",
            (entity.entity_id, entity.name, entity.entity_type, entity.description, json.dumps(entity.metadata)),
        )
        self._connection.commit()

    def add_document(self, document: KnowledgeDocument) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO knowledge_documents VALUES (?, ?)",
            (document.document_id, document.model_dump_json()),
        )
        self._connection.commit()

    def add_relationship(self, relationship: KnowledgeRelationship) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO knowledge_relationships VALUES (?, ?, ?, ?, ?, ?, ?)",
            (relationship.relationship_id, relationship.source_entity_id, relationship.relationship_type,
             relationship.target_entity_id, relationship.source_document_id, relationship.evidence_id,
             json.dumps(relationship.metadata)),
        )
        self._connection.commit()

    # Words too generic to identify an entity on their own.
    _GENERIC_TERMS = frozenset({
        "section", "sections", "act", "acts", "rule", "rules", "article", "articles", "patent", "patents",
        "regulation", "regulations", "the", "and", "for", "law", "india", "indian", "what", "which",
        "document", "guideline", "guidelines", "order", "clause", "chapter", "part", "code", "product",
        "drug", "medicine", "authority", "national", "under", "with", "from", "that", "this", "about",
    })
    _PROVISION_RE = re.compile(r"\b(section|sec|rule|article|regulation|clause)\s*(\d+[a-z]?)((?:\s*\(\s*\w+\s*\))*)", re.I)
    MAX_MATCHED_ENTITIES = 12
    MAX_RELATIONSHIPS = 200

    @staticmethod
    def _norm(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    @classmethod
    def _provision_keys(cls, text: str) -> set[str]:
        """Normalized provision references, e.g. 'Section 3(p)' -> 'section3p'."""
        keys = set()
        for kind, num, subs in cls._PROVISION_RE.findall(text):
            kind = "section" if kind.lower() == "sec" else kind.lower()
            keys.add(kind + num.lower() + cls._norm(subs))
        return keys

    def find_entities(self, text: str) -> list[KnowledgeEntity]:
        lowered = text.lower()
        query_norm = self._norm(text)
        query_provisions = self._provision_keys(text)
        query_tokens = {t for t in re.findall(r"[a-z0-9]+", lowered) if len(t) > 2}
        rows = self._connection.execute(
            "SELECT entity_id, name, entity_type, description, metadata_json FROM knowledge_entities"
        ).fetchall()
        scored = []
        for entity_id, name, entity_type, description, metadata_json in rows:
            name_lower = name.lower().strip()
            name_norm = self._norm(name)
            if not name_norm:
                continue
            score = 0.0
            entity_provisions = self._provision_keys(name)
            if entity_provisions:
                # Provision-style entity: match only on the exact provision reference.
                if entity_provisions & query_provisions:
                    score = 3.0
            else:
                if re.search(r"(?<![a-z0-9])" + re.escape(name_lower) + r"(?![a-z0-9])", lowered):
                    score = 2.0
                else:
                    name_tokens = {t for t in re.findall(r"[a-z0-9]+", name_lower) if len(t) > 2}
                    specific = name_tokens - self._GENERIC_TERMS
                    # Multi-token names need most of their specific tokens present;
                    # never match on generic words alone.
                    if len(specific) >= 2 and len(specific & query_tokens) >= max(2, len(specific) - 1):
                        score = 1.0
                    elif len(specific) == 1 and len(name_tokens) == 1 and specific <= query_tokens:
                        score = 1.5  # single-word specific names/acronyms, e.g. "TKDL"
            if score:
                scored.append((score, KnowledgeEntity(entity_id=entity_id, name=name, entity_type=entity_type,
                                                      description=description, metadata=json.loads(metadata_json))))
        scored.sort(key=lambda pair: (-pair[0], pair[1].entity_id))
        return [entity for _, entity in scored[: self.MAX_MATCHED_ENTITIES]]

    def context_for(self, text: str, max_depth: int = 2) -> KnowledgeGraphContext:
        max_depth = max(0, min(max_depth, 2))
        entities = self.find_entities(text)
        matched_ids = {entity.entity_id for entity in entities}
        if not matched_ids:
            return KnowledgeGraphContext()
        visited = set(matched_ids)
        frontier = set(matched_ids)
        relationships: list[KnowledgeRelationship] = []
        seen_rels: set[str] = set()
        for _ in range(max_depth):
            if not frontier:
                break
            placeholders = ",".join("?" for _ in frontier)
            rows = self._connection.execute(
                f"SELECT relationship_id, source_entity_id, relationship_type, target_entity_id, "
                f"source_document_id, evidence_id, metadata_json FROM knowledge_relationships "
                f"WHERE source_entity_id IN ({placeholders}) OR target_entity_id IN ({placeholders})",
                tuple(frontier) + tuple(frontier),
            ).fetchall()
            next_frontier = set()
            for row in rows:
                relationship = KnowledgeRelationship(
                    relationship_id=row[0], source_entity_id=row[1], relationship_type=row[2],
                    target_entity_id=row[3], source_document_id=row[4], evidence_id=row[5],
                    metadata=json.loads(row[6]),
                )
                if relationship.relationship_id not in seen_rels and len(relationships) < self.MAX_RELATIONSHIPS:
                    seen_rels.add(relationship.relationship_id)
                    relationships.append(relationship)
                next_frontier.update({relationship.source_entity_id, relationship.target_entity_id} - visited)
            visited.update(next_frontier)
            frontier = next_frontier
        related_ids = visited | {r.source_entity_id for r in relationships} | {r.target_entity_id for r in relationships}
        entity_rows = self._connection.execute(
            f"SELECT entity_id, name, entity_type, description, metadata_json FROM knowledge_entities "
            f"WHERE entity_id IN ({','.join('?' for _ in related_ids)})", tuple(related_ids)
        ).fetchall()
        all_entities = [KnowledgeEntity(entity_id=row[0], name=row[1], entity_type=row[2],
                                         description=row[3], metadata=json.loads(row[4])) for row in entity_rows]
        document_ids = {r.source_document_id for r in relationships if r.source_document_id}
        documents = []
        for document_id in document_ids:
            row = self._connection.execute(
                "SELECT payload_json FROM knowledge_documents WHERE document_id = ?", (document_id,)
            ).fetchone()
            if row:
                documents.append(KnowledgeDocument.model_validate_json(row[0]))
        return KnowledgeGraphContext(entities=all_entities, relationships=relationships, documents=documents,
                         matched_entity_ids=sorted(matched_ids), traversal_depth=max_depth)

    def add_many(self, entities: Iterable[KnowledgeEntity] = (), relationships: Iterable[KnowledgeRelationship] = ()) -> None:
        for entity in entities:
            self.add_entity(entity)
        for relationship in relationships:
            self.add_relationship(relationship)
