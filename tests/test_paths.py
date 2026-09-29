"""Regression tests for ingestion/paths.py.

These pin down the exact bug found in the authoritative manifest: paths
persisted with Windows backslash separators (or an absolute path from a
different machine) must still resolve correctly when read back on POSIX,
regardless of the process's current working directory.
"""
import os

from ingestion.paths import REPO_ROOT, resolve_repo_path, to_portable_relpath


def test_windows_style_relative_path_resolves_under_repo_root():
    windows_path = r"data\authoritative\raw\example-abc123.pdf"
    resolved = resolve_repo_path(windows_path)
    assert resolved == (REPO_ROOT / "data" / "authoritative" / "raw" / "example-abc123.pdf").resolve()


def test_posix_style_relative_path_resolves_the_same_as_windows_style():
    posix_path = "data/authoritative/raw/example-abc123.pdf"
    windows_path = "data\\authoritative\\raw\\example-abc123.pdf"
    assert resolve_repo_path(posix_path) == resolve_repo_path(windows_path)


def test_resolution_is_independent_of_current_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # simulate uvicorn/pytest launched from an unrelated CWD
    resolved = resolve_repo_path("data/authoritative/manifests/corpus_manifest.json")
    assert resolved == (REPO_ROOT / "data" / "authoritative" / "manifests" / "corpus_manifest.json").resolve()


def test_to_portable_relpath_always_emits_forward_slashes():
    p = to_portable_relpath(REPO_ROOT / "data" / "authoritative" / "raw" / "x.pdf")
    assert "\\" not in p
    assert p == "data/authoritative/raw/x.pdf"


def test_round_trip_write_then_read_regardless_of_separator_style():
    original = REPO_ROOT / "data" / "authoritative" / "normalized" / "doc.txt"
    portable = to_portable_relpath(original)
    assert resolve_repo_path(portable) == original.resolve()
    # A manifest entry written on Windows would contain backslashes even
    # though our writer now emits forward slashes -- both must resolve.
    assert resolve_repo_path(portable.replace("/", "\\")) == original.resolve()


def test_real_authoritative_manifest_entries_resolve_to_existing_files():
    """Guards against the exact regression: validate_authoritative_corpus
    reported 'missing raw archive' for all 6 entries because Path(entry.raw_path)
    silently failed on backslash-separated paths."""
    import json

    manifest_path = REPO_ROOT / "data" / "authoritative" / "manifests" / "corpus_manifest.json"
    if not manifest_path.is_file():
        return  # nothing to check in environments without the sample corpus
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    ingested = [e for e in data.get("entries", []) if e.get("fetch_status") == "ingested"]
    assert ingested, "expected at least one ingested manifest entry"
    for entry in ingested:
        assert resolve_repo_path(entry["raw_path"]).is_file(), f"missing raw archive for {entry['document_id']}"
        assert resolve_repo_path(entry["normalized_path"]).is_file(), f"missing normalized text for {entry['document_id']}"
