"""
Document Normalizer Module - Stage 3 of Ingestion.
Normalizes extracted text while strictly preserving legal terms and semantics.
"""
import re
from typing import Dict, Any


class DocumentNormalizer:
    """Cleans whitespace, headers/footers, broken lines while retaining legal content structure."""

    def normalize(self, parsed_doc: Dict[str, Any]) -> Dict[str, Any]:
        text = parsed_doc.get("content", "")
        
        # 1. Normalize line endings (CRLF -> LF)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Keep page markers; they are provenance needed for citations.

        # 3. Replace multiple horizontal whitespace spaces/tabs with single space
        text = re.sub(r"[ \t]+", " ", text)

        # 4. Join broken lines within sentences (e.g. single word split across line break without paragraph break)
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                cleaned_lines.append(stripped)
            else:
                cleaned_lines.append("") # Preserve paragraph boundaries

        normalized_text = "\n".join(cleaned_lines)
        
        # 5. Normalize excessive blank lines (more than 2 consecutive newlines) to 2 newlines
        normalized_text = re.sub(r"\n{3,}", "\n\n", normalized_text).strip()

        normalized_doc = dict(parsed_doc)
        normalized_doc["content"] = normalized_text
        return normalized_doc
