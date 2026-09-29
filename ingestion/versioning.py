"""
Versioning Module - Manages document versions and effective date awareness.
"""
from typing import Dict, Any, List
from models.document import DocumentMetadata


class VersionManager:
    """Manages document versions and effective dates."""

    def resolve_version(self, new_meta: DocumentMetadata, existing_metas: List[DocumentMetadata]) -> Dict[str, Any]:
        """
        Compares new document metadata against existing versions for the same title/authority.
        Determines whether it is a new version or supersedes an older version.
        """
        matching_versions = [
            m for m in existing_metas if m.title == new_meta.title and m.authority == new_meta.authority
        ]

        if not matching_versions:
            return {"status": "new_document", "active_version": new_meta.version}

        # Check effective dates
        newer_versions = [m for m in matching_versions if m.effective_date > new_meta.effective_date]
        if newer_versions:
            return {
                "status": "historical_version",
                "message": f"Ingested version ({new_meta.effective_date}) is historical relative to current version ({newer_versions[0].effective_date})"
            }

        return {
            "status": "latest_version",
            "message": f"Version {new_meta.version} (eff. {new_meta.effective_date}) is now the active latest version."
        }
