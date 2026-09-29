"""Regression tests for the FastEmbed initialization hang guard.

Root cause pinned down here: FastEmbed downloads model weights from
Hugging Face with no timeout of its own, so a stalled/blocked network
previously hung the entire ingestion process forever right after the
"fetch succeeded" log line. These tests verify:

(a) `_init_model_with_timeout` gives up after the configured budget instead
    of hanging, by exercising the *real* method against a stubbed slow
    `fastembed.TextEmbedding`.
(b) the loader thread is a daemon thread, so an abandoned/still-running load
    never blocks interpreter exit (this specifically guards against
    reintroducing the bug via `concurrent.futures.ThreadPoolExecutor`, whose
    worker threads are non-daemon and get joined by an atexit hook).
(c) `FastEmbedProvider.__init__` degrades to `model=None` (-> caller falls
    back to `HashEmbeddingProvider`) rather than propagating the timeout.
"""
import sys
import threading
import time
import types

import pytest

from embeddings.provider import FastEmbedProvider, _FastEmbedInitTimeout


def _install_slow_fastembed(monkeypatch, sleep_seconds):
    """Replaces the `fastembed` module's TextEmbedding with a stub that
    blocks for `sleep_seconds` before returning, without touching the network."""
    class SlowTextEmbedding:
        def __init__(self, model_name):
            time.sleep(sleep_seconds)
            self.model_name = model_name

    fake_module = types.ModuleType("fastembed")
    fake_module.TextEmbedding = SlowTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake_module)


def test_init_gives_up_after_timeout_instead_of_hanging(monkeypatch):
    _install_slow_fastembed(monkeypatch, sleep_seconds=5)
    start = time.time()
    with pytest.raises(_FastEmbedInitTimeout):
        FastEmbedProvider._init_model_with_timeout("stub-model", timeout_seconds=0.2)
    elapsed = time.time() - start
    assert elapsed < 2.0, "caller must not wait anywhere near the full stalled duration"


def test_init_succeeds_when_load_completes_within_budget(monkeypatch):
    _install_slow_fastembed(monkeypatch, sleep_seconds=0.05)
    model = FastEmbedProvider._init_model_with_timeout("stub-model", timeout_seconds=2.0)
    assert model is not None
    assert model.model_name == "stub-model"


def test_provider_init_falls_back_to_none_model_on_timeout(monkeypatch):
    _install_slow_fastembed(monkeypatch, sleep_seconds=1.0)
    monkeypatch.setattr("embeddings.provider._FASTEMBED_INIT_TIMEOUT_SECONDS", 0.1)
    provider = FastEmbedProvider(model_name="stub-model")
    # __init__ must swallow the timeout and leave model=None, exactly like the
    # pre-existing "unavailable" fallback path, so get_default_embedding_provider()
    # falls back to HashEmbeddingProvider instead of raising or hanging.
    assert provider.model is None


def test_loader_thread_is_daemon_so_it_never_blocks_process_exit(monkeypatch):
    """The specific bug this guards against: concurrent.futures.ThreadPoolExecutor
    registers a NON-daemon worker with an atexit hook that JOINS abandoned
    workers on interpreter shutdown -- so 'giving up' after the timeout would
    still hang the whole process (e.g. the one-shot ingestion script) until
    the stalled download's own internal retries finished. A plain daemon
    thread must be used instead."""
    _install_slow_fastembed(monkeypatch, sleep_seconds=5)

    captured = {}
    real_thread = threading.Thread

    def spying_thread(*args, **kwargs):
        t = real_thread(*args, **kwargs)
        captured["thread"] = t
        return t

    monkeypatch.setattr(threading, "Thread", spying_thread)

    with pytest.raises(_FastEmbedInitTimeout):
        FastEmbedProvider._init_model_with_timeout("stub-model", timeout_seconds=0.2)

    assert captured.get("thread") is not None
    assert captured["thread"].daemon is True
