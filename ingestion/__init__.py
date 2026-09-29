from ingestion.fetcher import DocumentFetcher
from ingestion.parser import DocumentParser
from ingestion.normalizer import DocumentNormalizer
from ingestion.metadata import MetadataExtractor
from ingestion.chunker import LegalChunker
from ingestion.deduplicator import Deduplicator
from ingestion.versioning import VersionManager
from ingestion.pipeline import IngestionPipeline

__all__ = [
    "DocumentFetcher",
    "DocumentParser",
    "DocumentNormalizer",
    "MetadataExtractor",
    "LegalChunker",
    "Deduplicator",
    "VersionManager",
    "IngestionPipeline",
]
