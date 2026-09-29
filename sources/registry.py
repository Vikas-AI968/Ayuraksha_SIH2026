"""
Authoritative Source Registry for IP-SAKTI Sahayak.

This registry describes WHICH bodies and explicitly configured documents are authoritative for which
jurisdiction/domain combinations. It intentionally does NOT contain
fabricated document-level URLs. `homepage_url` values below are the
verified public homepages of the real organizations. A homepage is never
treated as an ingested document; document records are loaded separately
from explicit configuration.

`ingestion_status` (per source body) and per-document status are DERIVED
from `data/authoritative/manifests/corpus_manifest.json` -- the ONE
truthful record of what has actually been fetched, parsed, and indexed
(matched via each manifest entry's `source_id`). This file used to
hand-maintain every entry as "not_ingested" regardless of what the
manifest said, which meant the registry claimed the 6 real IPINDIA/WIPO
documents (1219 chunks) were unavailable even after they were ingested.
That duplicate, drift-prone state is gone: the manifest is now the single
source of truth, and this module only formats/derives from it.
"""
from typing import List, Optional, Dict, Any
import json
import logging
import os
from pydantic import BaseModel, Field
from models.source import ConfiguredDocument, CorpusManifest

logger = logging.getLogger("sources.registry")

DEFAULT_MANIFEST_PATH = "data/authoritative/manifests/corpus_manifest.json"

# One of these is the truthful status for any given document_id.
DOC_STATUS_AUTHORITATIVE_CURRENT = "authoritative_current"
DOC_STATUS_AUTHORITATIVE_DRAFT = "authoritative_draft"
DOC_STATUS_AUTHORITATIVE_HISTORICAL = "authoritative_historical"
DOC_STATUS_REFERENCE = "reference"
DOC_STATUS_SYNTHETIC = "synthetic_development"
DOC_STATUS_NOT_INGESTED = "not_ingested"


class SourceRegistryEntry(BaseModel):
    source_id: str
    source_name: str
    source_type: str = Field(..., description="official|treaty|regulation|guideline")
    authority: str
    jurisdiction: str  # "India" | "International" | specific destination market
    domains: List[str] = Field(default_factory=list, description="Domain taxonomy values this body is authoritative for")
    homepage_url: Optional[str] = Field(
        default=None,
        description="Verified organizational homepage. NOT a claim about where a specific document lives."
    )
    ingestion_status: str = Field(default="not_ingested", description="not_ingested|synthetic_only|adapter_pending")
    source_priority: float = Field(default=0.7, ge=0.0, le=1.0)
    access_status: str = Field(default="public", description="public|restricted|metadata_only|unavailable")
    notes: Optional[str] = None


_DOCUMENTS: List[ConfiguredDocument] = []


# --- India ---------------------------------------------------------------
_INDIA_SOURCES: List[SourceRegistryEntry] = [
    SourceRegistryEntry(
        source_id="india-code",
        source_name="India Code (Ministry of Law & Justice)",
        source_type="official",
        authority="Legislative Department, Government of India",
        jurisdiction="India",
        domains=["Patents", "Trademarks", "Copyright", "Designs", "Biodiversity/ABS", "Drug Classification"],
        homepage_url="https://www.indiacode.nic.in",
        ingestion_status="not_ingested",
        source_priority=1.0,
        notes="Canonical repository of central and state Acts/Rules of India.",
    ),
    SourceRegistryEntry(
        source_id="ip-india",
        source_name="Office of the Controller General of Patents, Designs & Trade Marks (IP India)",
        source_type="official",
        authority="Department for Promotion of Industry and Internal Trade (DPIIT), Government of India",
        jurisdiction="India",
        domains=["Patents", "Trademarks", "Designs", "Geographical Indications"],
        homepage_url="https://ipindia.gov.in",
        ingestion_status="not_ingested",
        source_priority=1.0,
    ),
    SourceRegistryEntry(
        source_id="ministry-of-ayush",
        source_name="Ministry of AYUSH",
        source_type="official",
        authority="Government of India",
        jurisdiction="India",
        domains=["Drug Classification"],
        homepage_url="https://ayush.gov.in",
        ingestion_status="not_ingested",
        source_priority=1.0,
        notes="Administers TKDL and Ayurveda drug/regulatory policy.",
    ),
    SourceRegistryEntry(
        source_id="tkdl",
        source_name="Traditional Knowledge Digital Library (TKDL)",
        source_type="official",
        authority="CSIR / Ministry of AYUSH",
        jurisdiction="India",
        domains=["Traditional Knowledge"],
        homepage_url="https://www.tkdl.res.in",
        ingestion_status="not_ingested",
        source_priority=0.8,
        access_status="restricted",
        notes="Prior-art database of codified Indian traditional medicine knowledge; access is restricted/paid for bulk use.",
    ),
    SourceRegistryEntry(
        source_id="national-biodiversity-authority",
        source_name="National Biodiversity Authority (NBA)",
        source_type="official",
        authority="Ministry of Environment, Forest and Climate Change, Government of India",
        jurisdiction="India",
        domains=["Biodiversity/ABS"],
        homepage_url="https://nbaindia.org",
        ingestion_status="not_ingested",
        source_priority=1.0,
        notes="Administers the Biological Diversity Act, 2002 and ABS approvals.",
    ),
    SourceRegistryEntry(
        source_id="fssai",
        source_name="Food Safety and Standards Authority of India (FSSAI)",
        source_type="official",
        authority="Ministry of Health and Family Welfare, Government of India",
        jurisdiction="India",
        domains=["Drug Classification"],
        homepage_url="https://fssai.gov.in",
        ingestion_status="not_ingested",
        source_priority=1.0,
        notes="Regulates 'Ayurveda Aahara' food category and general food safety compliance.",
    ),
]

# --- International ---------------------------------------------------------
_INTERNATIONAL_SOURCES: List[SourceRegistryEntry] = [
    SourceRegistryEntry(
        source_id="wipo",
        source_name="World Intellectual Property Organization (WIPO)",
        source_type="treaty",
        authority="WIPO",
        jurisdiction="International",
        domains=["Patents", "Trademarks", "Designs", "Traditional Knowledge"],
        homepage_url="https://www.wipo.int",
        ingestion_status="not_ingested",
        source_priority=0.8,
    ),
    SourceRegistryEntry(
        source_id="wto-trips",
        source_name="WTO / TRIPS Agreement",
        source_type="treaty",
        authority="World Trade Organization",
        jurisdiction="International",
        domains=["Patents", "Trademarks", "Copyright", "Designs"],
        homepage_url="https://www.wto.org",
        ingestion_status="not_ingested",
        source_priority=0.8,
    ),
    SourceRegistryEntry(
        source_id="cbd",
        source_name="Convention on Biological Diversity (CBD)",
        source_type="treaty",
        authority="UN CBD Secretariat",
        jurisdiction="International",
        domains=["Biodiversity/ABS"],
        homepage_url="https://www.cbd.int",
        ingestion_status="not_ingested",
        source_priority=0.8,
    ),
    SourceRegistryEntry(
        source_id="nagoya-protocol",
        source_name="Nagoya Protocol on Access and Benefit-Sharing",
        source_type="treaty",
        authority="UN CBD Secretariat",
        jurisdiction="International",
        domains=["Biodiversity/ABS"],
        homepage_url="https://www.cbd.int/abs",
        ingestion_status="not_ingested",
        source_priority=0.8,
    ),
    SourceRegistryEntry(
        source_id="pct",
        source_name="Patent Cooperation Treaty (PCT)",
        source_type="treaty",
        authority="WIPO",
        jurisdiction="International",
        domains=["Patents"],
        homepage_url="https://www.wipo.int/pct/en",
        ingestion_status="not_ingested",
        source_priority=0.8,
    ),
    SourceRegistryEntry(
        source_id="madrid-system",
        source_name="Madrid System for International Trademark Registration",
        source_type="treaty",
        authority="WIPO",
        jurisdiction="International",
        domains=["Trademarks"],
        homepage_url="https://www.wipo.int/madrid/en",
        ingestion_status="not_ingested",
        source_priority=0.8,
    ),
    SourceRegistryEntry(
        source_id="hague-system",
        source_name="Hague System for International Design Registration",
        source_type="treaty",
        authority="WIPO",
        jurisdiction="International",
        domains=["Designs"],
        homepage_url="https://www.wipo.int/hague/en",
        ingestion_status="not_ingested",
        source_priority=0.8,
    ),
]

# --- Synthetic (Phase-1 test corpus, always marked as such) ---------------
_SYNTHETIC_SOURCE = SourceRegistryEntry(
    source_id="synthetic-phase1-corpus",
    source_name="IP-SAKTI Synthetic Development Corpus (Phase 1)",
    source_type="synthetic",
    authority="IP-SAKTI Sahayak (internal test authoring)",
    jurisdiction="India",
    domains=[
        "Patents", "Trademarks", "Copyright", "Designs", "Geographical Indications",
        "Traditional Knowledge", "Biodiversity/ABS", "Drug Classification",
    ],
    homepage_url=None,
    ingestion_status="synthetic_only",
    source_priority=0.0,
    access_status="development_only",
    notes="Representative synthetic documents for development/testing ONLY. Never present as real legal authority.",
)


class SourceRegistry:
    """In-memory registry of authoritative source categories."""

    def __init__(self) -> None:
        self._entries: List[SourceRegistryEntry] = (
            _INDIA_SOURCES + _INTERNATIONAL_SOURCES + [_SYNTHETIC_SOURCE]
        )
        self._documents = self._load_documents()
        self._manifest_entries = self._load_manifest()
        self._apply_manifest_truth()

    @staticmethod
    def _load_documents() -> List[ConfiguredDocument]:
        path = os.environ.get("AUTHORITATIVE_DOCUMENTS_CONFIG", "data/authoritative/configured_documents.json")
        if not os.path.isfile(path):
            return []
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return [ConfiguredDocument.model_validate(item) for item in payload]

    @staticmethod
    def _load_manifest() -> List[Dict[str, Any]]:
        """Loads the corpus manifest -- the single source of truth for what has
        actually been ingested (as opposed to merely configured)."""
        path = os.environ.get("AUTHORITATIVE_CORPUS_MANIFEST", DEFAULT_MANIFEST_PATH)
        if not os.path.isfile(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            manifest = CorpusManifest.model_validate(payload)
            return [entry.model_dump() for entry in manifest.entries]
        except Exception as e:
            logger.warning(f"Could not load corpus manifest at {path}: {e}")
            return []

    def _apply_manifest_truth(self) -> None:
        """Overrides each source body's ingestion_status based on the manifest,
        instead of the hand-maintained default. A source is only ever reported
        as ingested if the manifest shows an actually-ingested chunk_count>0
        document pointing at it."""
        ingested_by_source: Dict[str, List[Dict[str, Any]]] = {}
        for entry in self._manifest_entries:
            if entry.get("fetch_status") == "ingested" and entry.get("chunk_count", 0) > 0:
                ingested_by_source.setdefault(entry["source_id"], []).append(entry)

        for source_entry in self._entries:
            if source_entry.source_type == "synthetic":
                continue  # always "synthetic_only", never conflated with real ingestion
            docs = ingested_by_source.get(source_entry.source_id, [])
            if not docs:
                source_entry.ingestion_status = "not_ingested"
                continue
            statuses = {d.get("status") for d in docs}
            if "current" in statuses:
                source_entry.ingestion_status = "ingested_current"
            elif "draft" in statuses:
                source_entry.ingestion_status = "ingested_draft"
            else:
                source_entry.ingestion_status = "ingested_historical"

    def document_status(self, document_id: str) -> str:
        """Truthful per-document status, one of DOC_STATUS_* constants above."""
        for entry in self._manifest_entries:
            if entry.get("document_id") == document_id:
                if entry.get("fetch_status") != "ingested" or not entry.get("chunk_count", 0):
                    return DOC_STATUS_NOT_INGESTED
                status = entry.get("status")
                if status == "current":
                    return DOC_STATUS_AUTHORITATIVE_CURRENT
                if status == "draft":
                    return DOC_STATUS_AUTHORITATIVE_DRAFT
                if status in ("historical", "superseded"):
                    return DOC_STATUS_AUTHORITATIVE_HISTORICAL
                return DOC_STATUS_REFERENCE
        if document_id == _SYNTHETIC_SOURCE.source_id:
            return DOC_STATUS_SYNTHETIC
        return DOC_STATUS_NOT_INGESTED

    def ingested_documents(self) -> List[Dict[str, Any]]:
        """All manifest entries that are actually indexed right now, annotated
        with their truthful document_status."""
        result = []
        for entry in self._manifest_entries:
            if entry.get("fetch_status") == "ingested" and entry.get("chunk_count", 0) > 0:
                result.append({**entry, "document_status": self.document_status(entry["document_id"])})
        return result

    def all(self) -> List[SourceRegistryEntry]:
        return list(self._entries)

    def documents(self) -> List[ConfiguredDocument]:
        """Return explicitly configured document records, never inferred homepage URLs."""
        return list(self._documents)

    def document(self, document_id: str) -> Optional[ConfiguredDocument]:
        return next((document for document in self._documents if document.document_id == document_id), None)

    def get(self, source_id: str) -> Optional[SourceRegistryEntry]:
        for entry in self._entries:
            if entry.source_id == source_id:
                return entry
        return None

    def for_jurisdiction(self, jurisdiction: str) -> List[SourceRegistryEntry]:
        if not jurisdiction or jurisdiction.lower() in ("auto", "unknown", "ambiguous"):
            return list(self._entries)
        return [e for e in self._entries if e.jurisdiction.lower() == jurisdiction.lower()]

    def for_domain(self, domain: str) -> List[SourceRegistryEntry]:
        return [e for e in self._entries if domain in e.domains]

    def select(self, jurisdiction: Optional[str], domains: List[str]) -> List[SourceRegistryEntry]:
        """Select the source categories relevant to a routing decision.

        Always includes the synthetic corpus as a reference entry (clearly
        marked, never conflated with authoritative data), plus any real
        authoritative bodies matching the jurisdiction/domain -- their
        `ingestion_status` (see _apply_manifest_truth) truthfully reflects
        whether documents from that body are actually indexed right now.
        """
        candidates = self.for_jurisdiction(jurisdiction) if jurisdiction else list(self._entries)
        if domains:
            candidates = [e for e in candidates if any(d in e.domains for d in domains) or e.source_type == "synthetic"]
        # De-duplicate while preserving order, always keep the synthetic corpus present.
        seen = set()
        result = []
        for e in candidates:
            if e.source_id not in seen:
                seen.add(e.source_id)
                result.append(e)
            result.sort(key=lambda entry: entry.source_priority, reverse=True)
        if _SYNTHETIC_SOURCE.source_id not in seen:
            result.append(_SYNTHETIC_SOURCE)
        return result

    def summary(self) -> Dict[str, Any]:
        ingested_docs = self.ingested_documents()
        return {
            "total_sources": len(self._entries),
            "india_sources": len(self.for_jurisdiction("India")),
            "international_sources": len(self.for_jurisdiction("International")),
            "ingested_sources": [e.source_id for e in self._entries if e.ingestion_status.startswith("ingested")],
            "ingested_document_count": len(ingested_docs),
            "ingested_document_ids": [d["document_id"] for d in ingested_docs],
            "total_ingested_chunks": sum(d.get("chunk_count", 0) for d in ingested_docs),
        }


_default_registry: Optional[SourceRegistry] = None


def get_default_registry() -> SourceRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = SourceRegistry()
    return _default_registry
