"""Cross-platform path helpers for the authoritative manifest.

The authoritative manifest (data/authoritative/manifests/corpus_manifest.json)
persists ``raw_path`` / ``normalized_path`` strings that may have been written
on Windows (backslash separators, e.g. ``data\\authoritative\\raw\\x.pdf``) and
later read on POSIX (or vice versa). Python's ``pathlib.Path`` does not treat
``\\`` as a separator on POSIX, so those paths silently fail ``is_file()``
checks instead of raising -- this is exactly the "missing raw archive" /
"0 documents loaded" failure mode this module fixes.

It also avoids depending on the current working directory: manifest paths are
resolved relative to the repository root (this file's grandparent directory),
not whatever directory the process happened to be launched from.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Union

# ingestion/paths.py -> ingestion/ -> <repo root>
REPO_ROOT = Path(__file__).resolve().parent.parent


def to_portable_relpath(path: Union[str, Path]) -> str:
    """Return a POSIX-style path string, relative to REPO_ROOT when possible.

    Use this whenever persisting a path into the manifest so future reads
    (on any OS) can resolve it deterministically.
    """
    normalized = str(path).replace("\\", "/")
    p = Path(normalized)
    if p.is_absolute():
        try:
            p = p.relative_to(REPO_ROOT)
        except ValueError:
            return PurePosixPath(normalized).as_posix()
    return PurePosixPath(p).as_posix()


def resolve_repo_path(path_str: Union[str, Path, None]) -> Path:
    """Resolve a manifest-stored path (possibly Windows-style separators,
    possibly written by a process with a different CWD) to a real filesystem
    Path, anchored at the repository root rather than the current CWD.
    """
    if not path_str:
        return Path(str(path_str or ""))
    normalized = str(path_str).replace("\\", "/")
    p = Path(normalized)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def path_exists(path_str: Union[str, Path, None]) -> bool:
    """True if the manifest-stored path resolves to an existing file."""
    if not path_str:
        return False
    return resolve_repo_path(path_str).is_file()
