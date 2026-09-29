import re
from typing import Dict, Any, List


class DocumentParser:
    """Parses raw text and preserves headings/structural boundaries."""

    def parse(self, raw_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses document text, extracts section headings, page markers, and returns cleaned parsed doc.
        """
        content = raw_doc.get("content", "")
        if not content:
            raise ValueError(f"Document '{raw_doc.get('document_id')}' has empty content.")

        # Identify structural headings (Chapter X, Section Y, Article Z, Rule N, Clause K)
        heading_pattern = re.compile(
            r"^(CHAPTER\s+[IVXLCDM\d]+|Section\s+\d+\w*|Article\s+\d+\w*|Rule\s+\d+\w*|Clause\s+\d+\w*|SCENARIO\s+ANALYSIS|Case\s+Description)",
            re.IGNORECASE | re.MULTILINE
        )

        headings = heading_pattern.findall(content)
        parsed_doc = dict(raw_doc)
        parsed_doc["headings_found"] = headings
        parsed_doc["raw_content"] = content
        return parsed_doc
