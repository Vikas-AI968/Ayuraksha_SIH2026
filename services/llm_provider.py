"""
Abstract LLM Provider for the Reasoning Engine.

The Phase-1 repository did not have any LLM provider configured (no API
keys, no google-generativeai / anthropic dependency, no .env). Per RULE 16
("if a dependency or external service is unavailable, implement a clean
abstraction/fallback rather than breaking the architecture") and RULE 11
("keep the project runnable locally"), this module:

1. Defines an abstract `BaseLLMProvider` so the reasoning engine never talks
   to a concrete SDK directly.
2. Implements `AnthropicProvider` (Claude, via the plain REST API over
   `httpx` -- no extra dependency) and `GeminiProvider` (via the optional
   `google-generativeai` package) that activate ONLY if the corresponding
   API key environment variable is set.
3. Implements `ExtractiveFallbackProvider`, a zero-dependency, zero-network
   provider that deterministically composes an answer strictly from the
   Evidence Pack it is given. This is the default so the system is fully
   runnable end-to-end with no external service and can never hallucinate
   facts not present in the evidence.

Selection is controlled by the LLM_PROVIDER env var ("ollama" | "anthropic" | "gemini"
| "extractive"), defaulting to auto-detection based on which API key is
present, falling back to "extractive".
"""
from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger("llm_provider")


class LLMProviderError(RuntimeError):
    pass


class BaseLLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Generate a response and return it parsed as a JSON-compatible dict.

        Implementations must raise LLMProviderError on any failure (network,
        auth, malformed JSON) so the reasoning engine can fall back safely.
        """
        raise NotImplementedError


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Best-effort extraction of a single JSON object from model output."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise LLMProviderError("Could not parse a JSON object from LLM output.")


class GeminiProvider(BaseLLMProvider):
    """Calls Google's Gemini API via the google-genai SDK (or legacy google-generativeai)."""

    name = "gemini"

    def __init__(self, model: Optional[str] = None):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise LLMProviderError("GOOGLE_API_KEY is not set.")
        self.model_name = model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        # Try the new google-genai SDK first, fall back to legacy google-generativeai
        try:
            from google import genai
            from google.genai import types as genai_types
            client = genai.Client(api_key=self.api_key)
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = client.models.generate_content(
                model=self.model_name,
                contents=combined_prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            return _extract_json_object(response.text)
        except ImportError:
            pass
        except LLMProviderError:
            raise
        except Exception as e:
            raise LLMProviderError(f"Gemini API call failed (google-genai): {e}")

        # Fallback to legacy google-generativeai SDK
        try:
            import google.generativeai as genai_legacy
        except ImportError as e:
            raise LLMProviderError(
                f"Neither google-genai nor google-generativeai is installed; cannot use GeminiProvider: {e}"
            )
        try:
            genai_legacy.configure(api_key=self.api_key)
            model = genai_legacy.GenerativeModel(self.model_name, system_instruction=system_prompt)
            response = model.generate_content(user_prompt)
            return _extract_json_object(response.text)
        except LLMProviderError:
            raise
        except Exception as e:
            raise LLMProviderError(f"Gemini API call failed (google-generativeai): {e}")


_UNSET = object()


class OllamaProvider(BaseLLMProvider):
    """Ollama provider using the JSON-capable /api/chat endpoint.

    Works against either a local Ollama daemon (no API key, default
    http://localhost:11434) or Ollama Cloud (OLLAMA_API_KEY set, base URL
    https://ollama.com, model e.g. "gpt-oss:120b-cloud") -- the wire
    protocol is identical, so a single implementation covers both. Ollama
    Cloud is treated as an external service: bearer-token auth, configurable
    connect/total timeouts, and bounded retry-with-backoff on transient
    failures (429 / 5xx / timeout / connection error). Auth failures (401/403)
    and permanent client errors (4xx other than 429) are not retried. The
    API key is never logged or included in any exception message.
    """

    name = "ollama"

    # Status codes worth retrying: rate limit + server-side failures.
    _RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Any = _UNSET,
        timeout_seconds: Optional[float] = None,
        connect_timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        self.model = model or os.environ.get("OLLAMA_MODEL") or os.environ.get("LLM_MODEL", "qwen3:latest")
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        # Explicit None means caller explicitly requests "no API key" (suppress env var).
        # The sentinel _UNSET (not provided at all) falls back to the environment variable.
        if api_key is _UNSET:
            self.api_key = os.environ.get("OLLAMA_API_KEY") or None
        else:
            self.api_key = api_key if api_key else None
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None
            else float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120"))
        )
        self.connect_timeout_seconds = (
            connect_timeout_seconds if connect_timeout_seconds is not None
            else float(os.environ.get("OLLAMA_CONNECT_TIMEOUT_SECONDS", "10"))
        )
        self.max_retries = (
            max_retries if max_retries is not None
            else int(os.environ.get("OLLAMA_MAX_RETRIES", "2"))
        )

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _sleep_backoff(self, attempt: int) -> None:
        import time
        time.sleep(min(2 ** attempt * 0.5, 8.0))

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise LLMProviderError(f"httpx is required for OllamaProvider: {exc}")

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        timeout = httpx.Timeout(self.timeout_seconds, connect=self.connect_timeout_seconds)

        attempts = max(1, self.max_retries + 1)
        last_error: Optional[LLMProviderError] = None

        for attempt in range(attempts):
            try:
                response = httpx.post(url, json=payload, headers=self._headers(), timeout=timeout)
            except httpx.TimeoutException:
                last_error = LLMProviderError(
                    f"Ollama request timed out after {self.timeout_seconds}s "
                    f"(model={self.model})."
                )
            except httpx.ConnectError as exc:
                last_error = LLMProviderError(
                    f"Could not connect to Ollama at {self.base_url} (network unavailable?): {exc}"
                )
            except Exception as exc:
                # Any other httpx-level failure -- never retry unknown errors
                # indefinitely, but do give transient-looking failures one
                # more chance within the configured retry budget.
                last_error = LLMProviderError(f"Ollama request failed: {exc}")
            else:
                status = response.status_code
                if status == 401:
                    raise LLMProviderError(
                        "Ollama authentication failed (401 Unauthorized). Check that OLLAMA_API_KEY "
                        "is set and valid for Ollama Cloud."
                    )
                if status == 403:
                    raise LLMProviderError(
                        "Ollama request forbidden (403). The API key may lack access to the "
                        f"requested model ({self.model})."
                    )
                if status == 404:
                    raise LLMProviderError(
                        f"Ollama model '{self.model}' was not found (404) at {self.base_url}."
                    )
                if status in self._RETRYABLE_STATUS:
                    last_error = LLMProviderError(
                        f"Ollama returned a transient error (HTTP {status})."
                    )
                elif status >= 400:
                    raise LLMProviderError(
                        f"Ollama rejected the request (HTTP {status}): {response.text[:200]!r}"
                    )
                else:
                    try:
                        body = response.json()
                    except Exception as exc:
                        raise LLMProviderError(f"Ollama returned a non-JSON response body: {exc}")
                    message = body.get("message") or {}
                    content = message.get("content", "")
                    if not content:
                        raise LLMProviderError("Ollama returned an empty message body.")
                    return _extract_json_object(content)

            if attempt < attempts - 1:
                self._sleep_backoff(attempt)
                continue

        raise last_error or LLMProviderError("Ollama request failed for an unknown reason.")


class ExtractiveFallbackProvider(BaseLLMProvider):
    """Deterministic, dependency-free provider.

    Composes an answer strictly by extracting and lightly stitching together
    the supplied evidence text -- it never invents facts, and it is always
    available (no network, no API key). The reasoning engine passes the
    evidence pack contents inside `user_prompt`; this provider looks for the
    `__evidence_items__` structure the reasoning engine also attaches
    out-of-band via `context` (see services/reasoning.py) rather than trying
    to re-parse free text, so it stays perfectly grounded.
    """

    name = "extractive"

    _I18N = {
        "en": {
            "none": "There is not enough retrieved evidence to answer this query. Please refine the query or provide more context.",
            "nowarn": "No evidence was available for extractive synthesis.", "unc": "Insufficient evidence.",
            "intro": "Based on the retrieved evidence (extractive synthesis, no generative LLM available):",
            "acc": "According to {a} ({j}): {s}",
            "warn": "This answer was produced by deterministic extractive synthesis because no generative LLM was available. It quotes retrieved evidence directly rather than reasoning over it.",
        },
        "hi": {
            "none": "इस प्रश्न का उत्तर देने के लिए पर्याप्त साक्ष्य प्राप्त नहीं हुआ। कृपया प्रश्न को स्पष्ट करें या अधिक संदर्भ दें।",
            "nowarn": "निष्कर्षण के लिए कोई साक्ष्य उपलब्ध नहीं था।", "unc": "साक्ष्य अपर्याप्त है।",
            "intro": "प्राप्त साक्ष्य के आधार पर (निष्कर्षण-आधारित सारांश; मूल अंग्रेज़ी पाठ उद्धृत):",
            "acc": "{a} ({j}) के अनुसार: {s}",
            "warn": "यह उत्तर निर्धारित निष्कर्षण विधि से बनाया गया क्योंकि कोई जनरेटिव LLM उपलब्ध नहीं था। इसमें प्राप्त साक्ष्य का मूल अंग्रेज़ी पाठ सीधे उद्धृत है।",
        },
        "te": {
            "none": "ఈ ప్రశ్నకు సమాధానం ఇవ్వడానికి తగిన ఆధారాలు లభించలేదు. దయచేసి ప్రశ్నను స్పష్టం చేయండి లేదా మరింత సందర్భం ఇవ్వండి.",
            "nowarn": "సంగ్రహణకు ఎలాంటి ఆధారాలు అందుబాటులో లేవు.", "unc": "ఆధారాలు సరిపోవు.",
            "intro": "లభించిన ఆధారాల ప్రకారం (సంగ్రహణ ఆధారిత సారాంశం; అసలు ఆంగ్ల పాఠ్యం ఉటంకించబడింది):",
            "acc": "{a} ({j}) ప్రకారం: {s}",
            "warn": "జనరేటివ్ LLM అందుబాటులో లేనందున ఈ సమాధానం నిర్ణీత సంగ్రహణ పద్ధతిలో రూపొందించబడింది. ఇందులో లభించిన ఆధారాల అసలు ఆంగ్ల పాఠ్యం నేరుగా ఉటంకించబడింది.",
        },
    }

    def __init__(self, context: Optional[List[Dict[str, Any]]] = None, language: str = "en"):
        self._context = context
        self._language = language

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        evidence_items = self._context or []
        t = self._I18N.get(self._language, self._I18N["en"])
        if not evidence_items:
            return {
                "answer": t["none"],
                "key_points": [],
                "warnings": [t["nowarn"]],
                "uncertainties": [t["unc"]],
            }

        key_points = []
        answer_lines = []
        for item in evidence_items[:5]:
            snippet = item["text"].strip().replace("\n", " ")
            if len(snippet) > 320:
                snippet = snippet[:317].rstrip() + "..."
            key_points.append({"point": snippet, "evidence_ids": [item["evidence_id"]], "anchor": snippet})
            answer_lines.append(t["acc"].format(a=item["authority"], j=item["jurisdiction"], s=snippet))

        answer = t["intro"] + "\n\n" + "\n\n".join(answer_lines)
        return {
            "answer": answer,
            "key_points": key_points,
            "warnings": [t["warn"]],
            "uncertainties": [],
        }


def get_default_llm_provider(context: Optional[List[Dict[str, Any]]] = None) -> BaseLLMProvider:
    """Selects a provider based on LLM_PROVIDER env var / available configuration.

    Ollama Cloud (`gpt-oss:120b-cloud` via OLLAMA_API_KEY + OLLAMA_BASE_URL=
    https://ollama.com) is the primary, recommended provider for this
    deployment. A local Ollama daemon (OLLAMA_ENABLED=true, no API key) and
    Google Gemini are supported secondary options. Anthropic is intentionally
    NEVER auto-selected and is not offered as an explicit preference either;
    AnthropicProvider's class still exists in this module for completeness/
    testability, but is not reachable through this selection function.

    LLM_PROVIDER: "ollama" | "gemini" | "extractive" | "auto" (default).
    In "auto" mode: Ollama Cloud (if OLLAMA_API_KEY is set) or a local Ollama
    daemon (if OLLAMA_ENABLED=true) is tried first, then Gemini (if
    GOOGLE_API_KEY is set), then the dependency-free extractive fallback.
    Constructing a provider never raises here -- any failure to construct
    (missing config) is caught and logged, and selection falls through to
    the next option, ending at the extractive provider which always works.
    """
    preference = os.environ.get("LLM_PROVIDER", "auto").lower()

    def _try(name: str) -> Optional[BaseLLMProvider]:
        try:
            if name == "ollama":
                return OllamaProvider()
            if name == "gemini":
                return GeminiProvider()
        except LLMProviderError as e:
            logger.info(f"Provider '{name}' unavailable: {e}")
        return None

    if preference in ("ollama", "gemini"):
        provider = _try(preference)
        if provider:
            return provider
        logger.warning(f"Requested LLM_PROVIDER='{preference}' unavailable, falling back to extractive.")
        return ExtractiveFallbackProvider(context=context)

    if preference == "extractive":
        return ExtractiveFallbackProvider(context=context)

    # auto-detection, in priority order: Ollama Cloud -> local Ollama -> Gemini -> extractive.
    ollama_cloud_configured = bool(os.environ.get("OLLAMA_API_KEY"))
    ollama_local_enabled = os.environ.get("OLLAMA_ENABLED", "").lower() in {"1", "true", "yes"}
    if ollama_cloud_configured or ollama_local_enabled:
        provider = _try("ollama")
        if provider:
            return provider

    provider = _try("gemini")
    if provider:
        return provider

    return ExtractiveFallbackProvider(context=context)
