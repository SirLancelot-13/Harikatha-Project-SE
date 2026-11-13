"""Client for the local, self-hosted servers: the TTS server and an optional LLM server.

Best-effort by design: every failure raises LocalError, and callers fall back to Sarvam when
the provider is "auto". The Sarvam API key is never sent to these servers.
"""
import time

import httpx

from .config import settings

# Tests inject an httpx.MockTransport here.
transport: httpx.BaseTransport | None = None

HEALTH_TTL_SEC = 15.0
_health_cache: tuple[float, dict | None] | None = None


class LocalError(RuntimeError):
    pass


def _client(base_url: str, read_timeout: float) -> httpx.Client:
    timeout = httpx.Timeout(read_timeout, connect=settings.local_connect_timeout_sec)
    return httpx.Client(base_url=base_url, timeout=timeout, transport=transport)


def _detail(r: httpx.Response) -> str:
    try:
        body = r.json()
        return str(body.get("detail") or body.get("error") or body)[:300]
    except ValueError:
        return r.text[:300]


def reset_health_cache() -> None:
    global _health_cache
    _health_cache = None


def health() -> dict | None:
    """The TTS server's /health (engine, sample_rate, languages), or None if it is not usable.

    Cached briefly so a dead server costs one short probe, not one per request."""
    global _health_cache
    now = time.monotonic()
    if _health_cache and now - _health_cache[0] < HEALTH_TTL_SEC:
        return _health_cache[1]
    result = None
    try:
        with _client(settings.local_tts_url, 5.0) as c:
            r = c.get("/health")
        data = r.json() if r.status_code == 200 else {}
        if data.get("ok") and isinstance(data.get("languages"), list) and data.get("sample_rate"):
            result = data
    except (httpx.HTTPError, ValueError, AttributeError):
        result = None
    _health_cache = (now, result)
    return result


def synthesize(text: str, language: str, pace: float, kind: str) -> bytes:
    """One chunk of text -> WAV bytes (PCM16 mono at the server's sample rate)."""
    body = {"text": text, "language_code": language, "pace": pace, "kind": kind}
    try:
        with _client(settings.local_tts_url, settings.local_tts_timeout_sec) as c:
            r = c.post("/synthesize", json=body)
    except httpx.HTTPError as e:
        raise LocalError(f"local TTS network error: {type(e).__name__}") from e
    if r.status_code != 200:
        raise LocalError(f"local TTS -> {r.status_code}: {_detail(r)}")
    if not r.content:
        raise LocalError("local TTS returned no audio")
    return r.content


def chat(body: dict) -> dict:
    """POST an OpenAI-style chat completion to the local LLM server (Ollama, llama.cpp, vLLM...)."""
    if not settings.local_llm_url:
        raise LocalError("LOCAL_LLM_URL is not set")
    headers = {"Authorization": f"Bearer {settings.local_llm_api_key}"} if settings.local_llm_api_key else {}
    try:
        with _client(settings.local_llm_url.rstrip("/"), settings.local_llm_timeout_sec) as c:
            r = c.post("/chat/completions", json=body, headers=headers)
    except httpx.HTTPError as e:
        raise LocalError(f"local LLM network error: {type(e).__name__}") from e
    if r.status_code != 200:
        raise LocalError(f"local LLM -> {r.status_code}: {_detail(r)}")
    try:
        return r.json()
    except ValueError as e:
        raise LocalError("local LLM returned invalid JSON") from e
