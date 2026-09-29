"""
API Integration Tests using FastAPI TestClient.

`TestClient(app)` used as a bare object (not a `with` block) does NOT run
FastAPI's lifespan handler, so the retrieval indexes are never populated and
`/` correctly reports "degraded" (an honest, empty-index state) -- this is a
test-harness gap, not a production health-check bug (see api/main.py's
`_readiness_status`, which is deliberately truthful about an empty index).
Using a module-scoped `with TestClient(app) as client:` runs startup once for
the whole test module, matching how uvicorn actually runs the app.
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

_client_cm = TestClient(app)
client = _client_cm.__enter__()


def teardown_module(module):
    _client_cm.__exit__(None, None, None)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_ingest_api_endpoint():
    payload = {
        "title": "API Test Document",
        "authority": "API Test Authority",
        "jurisdiction": "India",
        "domain": "Patents",
        "document_type": "Guidance",
        "effective_date": "2026-01-01",
        "source_url": "synthetic://api-test-doc",
        "content": "Section 1. API Ingestion Test Content for citation RAG verification."
    }
    response = client.post("/api/v1/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunk_count"] > 0


def test_retrieve_api_endpoint():
    payload = {
        "query": "What is Section 3(p)?",
        "top_k": 3,
        "jurisdiction": "India",
        "domain": "Patents",
        "retrieval_method": "hybrid",
        "use_reranker": True
    }
    response = client.post("/api/v1/retrieve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert len(data["results"]) > 0
    first_res = data["results"][0]
    assert "citation" in first_res
    assert first_res["citation"]["source_url"] is not None
