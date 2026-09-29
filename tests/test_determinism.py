"""Cross-process determinism and Qdrant persistence tests.

These are the tests the engineering brief calls out explicitly as mandatory
(Tests A-D): a single Python process agreeing with itself proves nothing --
the whole point of the bugs being fixed here is that they only show up
*across* process boundaries. Each test below spawns real, separate `python`
subprocesses (not just separate function calls) to prove that.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(code: str) -> str:
    """Runs `code` in a brand-new python subprocess and returns its stdout.
    Raises if the subprocess errors, so failures show the real traceback."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout


# --- Test A: deterministic Qdrant point IDs across processes -------------

def test_stable_point_id_across_processes():
    code = (
        "from determinism import stable_point_id\n"
        "print(stable_point_id('patents_act_1970_chk_042'))\n"
    )
    out1 = _run(code)
    out2 = _run(code)
    assert out1 == out2
    assert out1.strip(), "point id must not be empty"


def test_stable_point_id_is_a_valid_uuid():
    import uuid
    from determinism import stable_point_id
    # Must not raise -- Qdrant point IDs must be an unsigned int or a UUID.
    uuid.UUID(stable_point_id("some_chunk_id"))


# --- Test B: deterministic fallback embeddings across processes ----------

def test_stable_hash_embedding_across_processes():
    code = (
        "from embeddings.provider import HashEmbeddingProvider\n"
        "p = HashEmbeddingProvider()\n"
        "v = p.embed_text('Section 3(p) traditional knowledge patent application')\n"
        "print(v)\n"
    )
    out1 = _run(code)
    out2 = _run(code)
    assert out1 == out2
    assert out1.strip() != "[]"


def test_stable_bucket_hash_across_processes():
    code = (
        "from determinism import stable_bucket_hash\n"
        "print(stable_bucket_hash('ayurveda'))\n"
    )
    out1 = _run(code)
    out2 = _run(code)
    assert out1 == out2


# --- Test C: persistent Qdrant storage survives a process restart -------

def test_qdrant_persistence_across_processes():
    qdrant_client = pytest.importorskip("qdrant_client")
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = str(Path(tmpdir) / "qdrant_data")
        write_code = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
from vector_store.qdrant_store import QdrantVectorStore
from embeddings.provider import HashEmbeddingProvider
from models.chunk import Chunk
from models.document import DocumentMetadata

meta = DocumentMetadata(
    title="Test Doc", authority="Test Authority", jurisdiction="India",
    domain="Patents", document_type="Act", effective_date="2020-01-01",
    source_url="synthetic://test", source_type="synthetic",
)
provider = HashEmbeddingProvider()
store = QdrantVectorStore(collection_name="persistence_test", location={store_path!r}, embedding_provider=provider)
chunk = Chunk(chunk_id="doc1_chk_001", document_id="doc1", text="patent traditional knowledge", metadata=meta)
store.upsert_chunks([chunk])
print("wrote", store.client.count(collection_name="persistence_test").count)
"""
        out1 = _run(write_code)
        assert "wrote 1" in out1

        read_code = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
from vector_store.qdrant_store import QdrantVectorStore
from embeddings.provider import HashEmbeddingProvider

provider = HashEmbeddingProvider()
store = QdrantVectorStore(collection_name="persistence_test", location={store_path!r}, embedding_provider=provider)
print("read", store.client.count(collection_name="persistence_test").count)
"""
        out2 = _run(read_code)
        assert "read 1" in out2, (
            "Qdrant did not persist across process restart -- "
            "a fresh process opening the same location/collection must see the same points."
        )


# --- Test D: idempotent ingestion (repeated upsert does not duplicate) ---

def test_idempotent_upsert_stable_point_count():
    qdrant_client = pytest.importorskip("qdrant_client")
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = str(Path(tmpdir) / "qdrant_data")
        code = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
from vector_store.qdrant_store import QdrantVectorStore
from embeddings.provider import HashEmbeddingProvider
from models.chunk import Chunk
from models.document import DocumentMetadata

meta = DocumentMetadata(
    title="Test Doc", authority="Test Authority", jurisdiction="India",
    domain="Patents", document_type="Act", effective_date="2020-01-01",
    source_url="synthetic://test", source_type="synthetic",
)
provider = HashEmbeddingProvider()
store = QdrantVectorStore(collection_name="idempotent_test", location={store_path!r}, embedding_provider=provider)
chunks = [
    Chunk(chunk_id=f"doc1_chk_{{i:03d}}", document_id="doc1", text=f"clause number {{i}}", metadata=meta)
    for i in range(5)
]
store.upsert_chunks(chunks)
count_after_first = store.client.count(collection_name="idempotent_test").count

# Re-run ingestion of the identical chunk set (simulates re-running the
# ingestion script, or the API re-loading the manifest on a fresh start).
store.upsert_chunks(chunks)
count_after_second = store.client.count(collection_name="idempotent_test").count

print("first", count_after_first)
print("second", count_after_second)
assert count_after_first == count_after_second == 5, (count_after_first, count_after_second)
"""
        out = _run(code)
        assert "first 5" in out and "second 5" in out
