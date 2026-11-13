"""Validated JSON generation through a locally hosted Ollama model."""
import hashlib
import json
import threading
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import settings

MAX_ATTEMPTS = 2
T = TypeVar("T", bound=BaseModel)
_log_lock = threading.Lock()
_calls: Counter[str] = Counter()


class LLMError(RuntimeError):
    """The local model could not produce validated output."""


class SoftCheck(ValueError):
    """A recoverable quality warning; accept the last answer if retry is exhausted."""


def calls_for(pid: str) -> int:
    return _calls[pid]


def _log(entry: dict) -> None:
    d = settings.data_dir / "runtime" / "llm_log"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{datetime.now(UTC).date()}.jsonl"
    with _log_lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in response")
    return json.loads(text[start : end + 1])


def _chat(messages: list[dict]) -> tuple[str, dict]:
    try:
        with httpx.Client(base_url=settings.ollama_base_url, timeout=settings.request_timeout_sec) as client:
            response = client.post("/api/chat", json={
                "model": settings.llm_model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {"temperature": settings.llm_temperature, "num_predict": settings.llm_max_tokens},
            })
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise LLMError(f"Ollama request failed: {type(exc).__name__}: {exc}") from exc
    data = response.json()
    return data.get("message", {}).get("content", ""), {
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
    }


def complete_json(
    *, pid: str, stage: str, prompt: str, model: type[T], mock_fn: Callable[[], T],
    system: str = "", check: Callable[[T], None] | None = None, reasoning_effort: str | None = None,
) -> T:
    """Validate structured model output, retry once with the validation error."""
    mock = settings.use_mock
    schema = json.dumps(model.model_json_schema(), ensure_ascii=False)
    messages = [
        {"role": "system", "content": f"{system}\n\nReturn one JSON object matching this schema:\n{schema}".strip()},
        {"role": "user", "content": prompt},
    ]
    _calls[pid] += 1
    last_error = ""
    for attempt in range(MAX_ATTEMPTS):
        start = time.monotonic()
        ok, response, usage, result = False, "", {}, None
        try:
            if mock:
                result = model.model_validate(mock_fn())
                response = result.model_dump_json()
            else:
                response, usage = _chat(messages)
                result = model.model_validate(_extract_json(response))
            if check:
                check(result)
            ok = True
            return result
        except (ValueError, ValidationError) as exc:
            last_error = str(exc)[:500]
            if mock:
                raise
            if isinstance(exc, SoftCheck) and attempt == MAX_ATTEMPTS - 1 and result is not None:
                return result
            messages.extend([
                {"role": "assistant", "content": response},
                {"role": "user", "content": f"Your last JSON was invalid: {last_error}. Return corrected JSON only."},
            ])
        finally:
            _log({
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "performance_id": pid, "stage": stage,
                "model": "mock" if mock else settings.llm_model,
                "prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest(),
                "prompt": prompt[:4000], "response": response, "tokens": usage,
                "latency_ms": round((time.monotonic() - start) * 1000),
                "validation_ok": ok, "retry": attempt,
            })
    raise LLMError(f"{stage}: invalid output after {MAX_ATTEMPTS} attempts: {last_error}")
