"""
Data Validation Module - Stage 18 of Ingestion.
Validates document schema completeness before indexing. Quarantines invalid documents.
"""
from typing import Dict, Any, Tuple, List
import re


class DocumentValidator:
    """Validates document completeness and schema requirements."""

    REQUIRED_FIELDS = [
        "title",
        "authority",
        "jurisdiction",
        "domain",
        "document_type",
        "effective_date",
        "source_url",
        "content"
    ]

    DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    def validate(self, raw_doc: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates document payload. Returns (is_valid, list_of_errors).
        """
        errors = []

        for field in self.REQUIRED_FIELDS:
            val = raw_doc.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                errors.append(f"Missing or empty required field: '{field}'")

        eff_date = raw_doc.get("effective_date", "")
        if eff_date and not self.DATE_PATTERN.match(str(eff_date)):
            errors.append(f"Invalid effective_date format: '{eff_date}'. Must be YYYY-MM-DD.")

        content = raw_doc.get("content", "")
        if content and len(content.strip()) < 10:
            errors.append("Document content too short (<10 characters).")

        return len(errors) == 0, errors
