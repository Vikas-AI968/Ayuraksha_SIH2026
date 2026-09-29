"""
Embedding Provider Interface & Implementations.
Abstracts embedding generation to support model swapping without pipeline changes.
"""
from abc import ABC, abstractmethod
import threading
from typing import List, Optional
import logging
import os
import numpy as np

from determinism import stable_bucket_hash


class _FastEmbedInitTimeout(Exception):
    """Raised when FastEmbed model initialization exceeds the configured budget."""

logger = logging.getLogger("embedding_provider")

# FastEmbed downloads ONNX model weights from Hugging Face on first use with
# no built-in timeout of its own. On a slow, blocked, or stalled network this
# can hang the ingestion/API startup process indefinitely -- exactly the
# "process appears to hang after the PDF fetch log line" symptom. Bound the
# initialization in a worker thread so a stall degrades to the deterministic
# hash-based fallback instead of hanging forever.
_FASTEMBED_INIT_TIMEOUT_SECONDS = float(os.environ.get("FASTEMBED_INIT_TIMEOUT_SECONDS", "60"))

# Root cause of the ingestion hang this batching/timeout guards against: the
# original code sent the *entire* chunk list (869 chunks for the Patents Act
# alone) to FastEmbed/ONNX Runtime in a single `model.embed(texts)` call, with
# no progress output and no bound on how long inference could take. If ONNX
# is pathological on a given machine (thread oversubscription, a slow/no-AVX2
# CPU fallback kernel, etc.) that single call can run for tens of minutes
# with zero visibility and no way to recover short of killing the process.
# Splitting into small batches turns that into: observable progress, and a
# hard per-batch time budget after which we give up on FastEmbed for the rest
# of *this* call and use the existing deterministic hash fallback instead --
# for the whole call, not just the remaining batches, so a single document's
# chunks are never split across two incompatible embedding spaces.
_EMBED_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))
_EMBED_BATCH_TIMEOUT_SECONDS = float(os.environ.get("EMBEDDING_BATCH_TIMEOUT_SECONDS", "120"))
# Optional explicit ONNX Runtime thread count. Left unset by default (None ->
# fastembed/onnxruntime auto-detects), but exposed because runaway CPU time
# with little wall-clock progress is a classic symptom of ONNX Runtime
# auto-detecting a pathological thread count on some Windows/CPU combos.
_FASTEMBED_THREADS = os.environ.get("FASTEMBED_THREADS")


def _call_with_timeout(fn, timeout_seconds: float):
    """Runs ``fn()`` on a daemon thread bounded by ``timeout_seconds``.

    Raises ``TimeoutError`` if it does not complete in time (the thread is
    abandoned -- it is a daemon, so it can never block process exit), or
    re-raises whatever exception ``fn()`` raised. Returns ``fn()``'s result
    on success. Shared by FastEmbed model init and batch inference so both
    degrade the same way instead of hanging forever.
    """
    result: dict = {}

    def _run():
        try:
            result["value"] = fn()
        except Exception as exc:  # noqa: BLE001 - surfaced via result dict
            result["error"] = exc

    thread = threading.Thread(target=_run, name="bounded-call", daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    if thread.is_alive():
        raise TimeoutError(f"operation did not complete within {timeout_seconds}s")
    if "error" in result:
        raise result["error"]
    return result.get("value")


class BaseEmbeddingProvider(ABC):
    """Abstract Base Class for Embedding Providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a float vector."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings into float vectors."""
        pass


class FastEmbedProvider(BaseEmbeddingProvider):
    """FastEmbed Provider using ONNX runtime (multilingual MiniLM-L12-v2, 384 dimensions)."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self._dim = 384
        try:
            from fastembed import TextEmbedding
            for m in TextEmbedding.list_supported_models():
                if m.get("model") == model_name:
                    self._dim = m.get("dim", 384)
                    break
        except Exception:
            pass
        self.model = None
        try:
            self.model = self._init_model_with_timeout(model_name, _FASTEMBED_INIT_TIMEOUT_SECONDS)
            if self.model is not None:
                logger.info(f"Initialized FastEmbedProvider with model {model_name} (dim={self._dim})")
        except _FastEmbedInitTimeout:
            logger.warning(
                "FastEmbed initialization exceeded %.0fs (likely a stalled model download); "
                "falling back to HashEmbeddingProvider. Set FASTEMBED_INIT_TIMEOUT_SECONDS to "
                "adjust, or EMBEDDING_PROVIDER=hash to skip FastEmbed entirely.",
                _FASTEMBED_INIT_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.warning(f"FastEmbed initialization failed: {e}. Falling back to TFIDFEmbeddingProvider.")

    @staticmethod
    def _init_model_with_timeout(model_name: str, timeout_seconds: float):
        """Loads the FastEmbed model bounded by ``timeout_seconds`` (see
        ``_call_with_timeout``)."""
        def _load():
            import warnings
            from fastembed import TextEmbedding
            kwargs = {}
            if _FASTEMBED_THREADS:
                kwargs["threads"] = int(_FASTEMBED_THREADS)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*mean pooling.*",
                    category=UserWarning,
                )
                return TextEmbedding(model_name=model_name, **kwargs)

        try:
            return _call_with_timeout(_load, timeout_seconds)
        except TimeoutError:
            raise _FastEmbedInitTimeout(
                f"FastEmbed model load did not complete within {timeout_seconds}s"
            )

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        if self.model is None:
            raise RuntimeError("FastEmbed model is not initialized.")
        embeddings = list(self.model.embed([text]))
        return embeddings[0].tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.model is None:
            raise RuntimeError("FastEmbed model is not initialized.")

        total = len(texts)
        results: List[Optional[List[float]]] = [None] * total
        for start in range(0, total, _EMBED_BATCH_SIZE):
            batch = texts[start:start + _EMBED_BATCH_SIZE]

            def _run_batch(_batch=batch):
                return [emb.tolist() for emb in self.model.embed(_batch)]

            try:
                batch_vectors = _call_with_timeout(_run_batch, _EMBED_BATCH_TIMEOUT_SECONDS)
            except Exception as exc:
                logger.warning(
                    "[EMBEDDING] FastEmbed inference stalled/failed at batch %d-%d of %d (%s). "
                    "Falling back to the deterministic hash embedding provider for this entire "
                    "call, so chunks aren't split across two incompatible embedding spaces. "
                    "If this keeps happening, set EMBEDDING_PROVIDER=hash, lower "
                    "EMBEDDING_BATCH_SIZE, or set FASTEMBED_THREADS to a small fixed value, "
                    "then re-ingest so all chunks share one embedding space.",
                    start, min(start + _EMBED_BATCH_SIZE, total), total, exc,
                )
                return HashEmbeddingProvider(dim=self._dim).embed_batch(texts)

            for i, vec in enumerate(batch_vectors):
                results[start + i] = vec
            done = min(start + _EMBED_BATCH_SIZE, total)
            logger.info("[EMBEDDING] embedded %d/%d", done, total)

        return results


class HashEmbeddingProvider(BaseEmbeddingProvider):
    """Fallback Hash/TF-IDF based vectorizer generating normalized 384-dim embeddings."""

    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_vector(self, text: str) -> List[float]:
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return vec.tolist()

        for token in tokens:
            # Deterministic hash to bucket (stable across processes/restarts)
            h = stable_bucket_hash(token) & 0x7fffffff
            idx = h % self._dim
            val = ((h >> 7) % 100) / 100.0 - 0.5
            vec[idx] += val

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_text(self, text: str) -> List[float]:
        return self._hash_vector(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_vector(t) for t in texts]


def get_default_embedding_provider() -> BaseEmbeddingProvider:
    """Factory method to load primary FastEmbed provider or fallback."""
    if os.environ.get("EMBEDDING_PROVIDER", "").lower() in {"hash", "deterministic"}:
        logger.info("[EMBEDDING] provider=hash-fallback (explicitly configured)")
        return HashEmbeddingProvider()
    try:
        model_name = os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        provider = FastEmbedProvider(model_name=model_name)
        if provider.model is not None:
            logger.info("[EMBEDDING] provider=fastembed model=%s dim=%d", model_name, provider.dimension)
            return provider
    except Exception as e:
        logger.warning(f"Could not load FastEmbedProvider: {e}")

    logger.info("[EMBEDDING] provider=hash-fallback")
    return HashEmbeddingProvider()
