"""Structure-aware legal chunking with inherited section and page context."""
from __future__ import annotations

from typing import List

from ingestion.legal_structure import LegalContext, iter_structured_segments
from models.chunk import Chunk
from models.document import Document, DocumentMetadata


class LegalChunker:
    """Chunk legal text at structural boundaries, then use bounded windows."""

    def __init__(self, target_chunk_size: int = 500, chunk_overlap: int = 100):
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: Document) -> List[Chunk]:
        doc_metadata = doc.metadata or DocumentMetadata(
            title=doc.title, authority=doc.authority, jurisdiction=doc.jurisdiction,
            domain=doc.domain, document_type=doc.document_type, effective_date=doc.effective_date,
            source_url=doc.source_url,
            source_type="synthetic" if doc.source_url.startswith("synthetic://") else "official",
        )
        chunks: List[Chunk] = []
        counter = 1
        for text, context in iter_structured_segments(doc.content):
            pieces = [text] if len(text) <= self.target_chunk_size * 4 else self._sliding_window_split(text, context)
            for piece in pieces:
                chunk_id = f"{doc.document_id}_chk_{counter:03d}"
                metadata = doc_metadata.model_copy(update={
                    "chapter": context.chapter, "section": context.section,
                    "subsection": context.subsection, "rule": context.rule,
                    "sub_rule": context.sub_rule, "regulation": context.regulation,
                    "sub_regulation": context.sub_regulation, "article": context.article,
                    "schedule": context.schedule, "paragraph": context.paragraph,
                    "page_number": context.page_number, "chunk_id": chunk_id,
                })
                chunks.append(Chunk(
                    chunk_id=chunk_id, document_id=doc.document_id, text=piece,
                    metadata=metadata, section=context.section, subsection=context.subsection,
                    rule=context.rule, sub_rule=context.sub_rule, regulation=context.regulation,
                    sub_regulation=context.sub_regulation, article=context.article,
                    chapter=context.chapter, schedule=context.schedule, paragraph=counter,
                ))
                counter += 1
        return chunks

    def _sliding_window_split(self, text: str, context: LegalContext) -> List[str]:
        words = text.split()
        step = max(1, self.target_chunk_size - self.chunk_overlap)
        prefix = self._context_prefix(context)
        pieces = []
        for start in range(0, len(words), step):
            window = " ".join(words[start:start + self.target_chunk_size])
            if not window:
                break
            pieces.append(f"[{prefix}] {window}" if prefix else window)
            if start + self.target_chunk_size >= len(words):
                break
        return pieces

    @staticmethod
    def _context_prefix(context: LegalContext) -> str:
        location = []
        if context.chapter:
            location.append(f"Chapter {context.chapter}")
        if context.section:
            location.append(f"Section {context.section}")
        if context.subsection:
            location.append(f"({context.subsection})")
        if context.rule:
            location.append(f"Rule {context.rule}")
        if context.regulation:
            location.append(f"Regulation {context.regulation}")
        return " ".join(location)
