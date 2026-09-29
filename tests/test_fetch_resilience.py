"""Regression tests for the ingestion "hang" fix (Section 7 of the audit spec).

Pins down two independent behaviors:
1. A transient network error during authoritative fetch is retried a bounded
   number of times, then raised (never retried forever, never silently
   swallowed).
2. A non-transient error (redirect-to-unexpected-host) is never retried.

Network calls are mocked -- these tests must not require internet access.
"""
import httpx
import pytest

from ingestion.authoritative import AuthoritativeCorpusService
from ingestion.pipeline import IngestionPipeline
from models.source import ConfiguredDocument


def _make_document(tmp_path, official_url="https://example-gov.test/doc.pdf"):
    return ConfiguredDocument(
        document_id="TEST-DOC-1",
        source_id="test-source",
        source_name="Test Source",
        issuing_authority="Test Authority",
        official_url=official_url,
        document_title="Test Document",
        document_type="statute",
        jurisdiction="India",
        domain="Patents",
        version="1.0",
        status="current",
        authoritative=True,
    )


def _service(tmp_path):
    return AuthoritativeCorpusService(
        IngestionPipeline(registry_file=":memory:"),
        vector_store=None, bm25_retriever=None, graph=None,
        root=str(tmp_path / "authoritative"),
    )


def test_transient_network_error_is_retried_then_succeeds(tmp_path, monkeypatch):
    document = _make_document(tmp_path)
    service = _service(tmp_path)

    calls = {"n": 0}

    def flaky_get(url, headers=None, timeout=None, follow_redirects=True):
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ReadTimeout("simulated stall", request=httpx.Request("GET", url))
        # Non-PDF content so this test exercises only the retry path, not the
        # PDF parser.
        return httpx.Response(
            200, content=b"<html><body>ok</body></html>",
            headers={"content-type": "text/html"},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", flaky_get)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)  # don't actually wait in tests

    entry = service.ingest_document(document)

    assert calls["n"] == 2, "expected exactly one retry before success"
    assert entry.fetch_status == "ingested"


def test_retries_are_bounded_and_final_failure_is_reported(tmp_path, monkeypatch):
    document = _make_document(tmp_path)
    service = _service(tmp_path)

    calls = {"n": 0}

    def always_times_out(url, headers=None, timeout=None, follow_redirects=True):
        calls["n"] += 1
        raise httpx.ReadTimeout("simulated permanent stall", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", always_times_out)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    entry = service.ingest_document(document)

    # AUTHORITATIVE_FETCH_MAX_RETRIES defaults to 2 -> 1 initial + 2 retries = 3 calls.
    assert calls["n"] == 3
    assert entry.fetch_status == "failed"
    assert entry.error  # a reason must be recorded, never silently dropped
    assert document.document_id in str((tmp_path / "authoritative" / "failed" / f"{document.document_id}.json"))


def test_redirect_to_unexpected_host_is_not_retried(tmp_path, monkeypatch):
    document = _make_document(tmp_path)
    service = _service(tmp_path)

    calls = {"n": 0}

    def redirected_get(url, headers=None, timeout=None, follow_redirects=True):
        calls["n"] += 1
        # Simulate the final response landing on a different host than the
        # one that was requested (e.g. a hijacked/misconfigured redirect).
        return httpx.Response(
            200, content=b"not the real doc",
            request=httpx.Request("GET", "https://attacker.example/doc.pdf"),
        )

    monkeypatch.setattr(httpx, "get", redirected_get)

    entry = service.ingest_document(document)

    assert calls["n"] == 1, "host-mismatch is a hard validation failure, never retried"
    assert entry.fetch_status == "failed"
    assert "redirected" in entry.error.lower()
