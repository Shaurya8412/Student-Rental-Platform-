"""LLM access layer: one provider-agnostic JSON call, disk-cached.

Design goals (per project spec):
- Temperature 0, JSON output.
- Every response is cached to disk, keyed by a SHA-256 hash of
  provider + model + prompts, so repeat runs are free and identical.
- Demo mode: if ``DEMO_MODE=1`` or no API key is configured, no network
  call is made. The cache is consulted first; on a miss, a caller-supplied
  deterministic ``fallback`` callable produces the response instead.
  Fallback ("stub") responses are cached too but tagged ``source="stub"``;
  when a real API key is present they are ignored and recomputed via the
  API, so demo answers never silently poison real runs.
- Robustness: retries with backoff, tolerant JSON extraction (code
  fences / surrounding prose), and a single ``LLMError`` type that
  callers can catch. Nothing in here should crash the UI.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from . import config

# Pick up a project-root `.env` (API keys) for CLI/test/evaluation use;
# the Streamlit app loads it too. Shell variables take precedence.
try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


class LLMError(Exception):
    """Raised when an LLM call fails and no fallback was provided."""


class LLMClient:
    """Minimal JSON-in/JSON-out LLM client with a disk cache."""

    def __init__(
        self,
        provider: Optional[str] = None,
        demo_mode: Optional[bool] = None,
        cache_path: Optional[Path] = None,
        model: Optional[str] = None,
    ) -> None:
        self.provider = (provider or os.getenv("LLM_PROVIDER")
                         or config.DEFAULT_PROVIDER).lower()
        if self.provider not in config.LLM_PROVIDERS:
            raise ValueError(f"Unknown LLM provider: {self.provider!r}")
        prov = config.LLM_PROVIDERS[self.provider]
        self.model = model or os.getenv("LLM_MODEL") or prov["model"]
        self.api_key = os.getenv(prov["env_key"], "")
        if demo_mode is None:
            demo_mode = os.getenv("DEMO_MODE", "0") == "1" or not self.api_key
        self.demo_mode = bool(demo_mode)
        self.cache_path = Path(cache_path or config.CACHE_PATH)
        self._cache: dict[str, Any] = self._load_cache()
        self.stats = {"api_calls": 0, "cache_hits": 0, "stub_calls": 0}

    # ------------------------------------------------------------------ API
    def call_json(
        self,
        system: str,
        user: str,
        fallback: Optional[Callable[[], dict]] = None,
    ) -> dict:
        """Return a parsed JSON object for (system, user).

        Order of resolution: disk cache -> live API (unless demo mode)
        -> ``fallback`` callable. Raises LLMError only if all fail and no
        fallback was given.
        """
        key = self._cache_key(system, user)
        entry = self._cache.get(key)
        if entry and (self.demo_mode or entry.get("source") == "api"):
            self.stats["cache_hits"] += 1
            return copy.deepcopy(entry["response"])

        if not self.demo_mode:
            try:
                response = self._call_api_with_retries(system, user)
                self._store(key, response, source="api")
                return copy.deepcopy(response)
            except LLMError:
                if fallback is None:
                    raise
        if fallback is not None:
            response = fallback()
            if not isinstance(response, dict):
                raise LLMError("Fallback did not return a dict")
            self.stats["stub_calls"] += 1
            self._store(key, response, source="stub")
            return copy.deepcopy(response)
        raise LLMError("LLM unavailable (demo mode / no API key) and no "
                       "fallback was provided")

    # -------------------------------------------------------------- caching
    def _cache_key(self, system: str, user: str) -> str:
        payload = json.dumps(
            {"provider": self.provider, "model": self.model,
             "system": system, "user": user},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _store(self, key: str, response: dict, source: str) -> None:
        self._cache[key] = {"response": response, "source": source}
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self._cache, indent=1), encoding="utf-8")
        except OSError:
            pass  # cache is an optimisation; never fail the call over it

    # ---------------------------------------------------------------- calls
    def _call_api_with_retries(self, system: str, user: str) -> dict:
        last_error: Optional[Exception] = None
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                text = self._call_api_once(system, user)
                self.stats["api_calls"] += 1
                return self._extract_json(text)
            except (LLMError, urllib.error.URLError, TimeoutError,
                    OSError) as exc:  # network + parse failures are retried
                last_error = exc
                if attempt < config.LLM_MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        raise LLMError(f"LLM call failed after "
                       f"{config.LLM_MAX_RETRIES} attempts: {last_error}")

    def _call_api_once(self, system: str, user: str) -> str:
        if self.provider == "gemini":
            url = ("https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{self.model}:generateContent?key={self.api_key}")
            body = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": config.LLM_TEMPERATURE,
                    "responseMimeType": "application/json",
                },
            }
            headers = {"Content-Type": "application/json"}
            extract = lambda d: d["candidates"][0]["content"]["parts"][0]["text"]  # noqa: E731
        elif self.provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            body = {
                "model": self.model,
                "temperature": config.LLM_TEMPERATURE,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            }
            headers = {"Content-Type": "application/json",
                       "Authorization": f"Bearer {self.api_key}"}
            extract = lambda d: d["choices"][0]["message"]["content"]  # noqa: E731
        else:  # anthropic
            url = "https://api.anthropic.com/v1/messages"
            body = {
                "model": self.model,
                "temperature": config.LLM_TEMPERATURE,
                "max_tokens": 1024,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            headers = {"Content-Type": "application/json",
                       "x-api-key": self.api_key,
                       "anthropic-version": "2023-06-01"}
            extract = lambda d: d["content"][0]["text"]  # noqa: E731

        request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"),
            headers=headers, method="POST")
        with urllib.request.urlopen(
                request, timeout=config.LLM_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        try:
            return extract(payload)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected API response shape: {exc}") from exc

    # -------------------------------------------------------------- parsing
    @staticmethod
    def _extract_json(text: str) -> dict:
        """Parse a JSON object out of raw model text, tolerating fences."""
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned,
                         flags=re.IGNORECASE)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        raise LLMError("Could not parse a JSON object from the LLM response")
