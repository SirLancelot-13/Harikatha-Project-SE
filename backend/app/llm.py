"""LLM wrapper: validates output against a pydantic model, retries once with the
validation error as feedback, and logs every attempt.

Providers: "sarvam" (hosted API), "local" (any OpenAI-compatible server you run, via
LOCAL_LLM_URL) and, for LLM_PROVIDER=auto, local first then Sarvam. Falling back to Sarvam
when the local model is unreachable OR keeps producing output that fails validation makes
the validators a quality gate for the local model.
MOCK_SARVAM=true uses the caller's mock_fn instead of the network.
"""
import hashlib
import json
import threading
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel, ValidationError

from . import local, sarvam
from .config import settings

MAX_ATTEMPTS = 2
_log_lock = threading.Lock()
_calls: Counter[str] = Counter()


class LLMError(RuntimeError):
    """The model answered, but never with output that passed validation."""


class SoftCheck(ValueError):
    """A check() failure worth one retry, but not worth failing the job: the last answer is kept."""


def calls_for(pid: str) -> int:
    return _calls[pid]


def _log(entry: dict) -> None:
    d = settings.data_dir / "runtime" / "llm_log"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{datetime.now(UTC).date()}.jsonl"
    with _log_lock, path.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in response")
    return json.loads(text[start : end + 1])


def providers() -> list[str]:
    """Providers to try, in order."""
    if settings.mock_sarvam:
        return ["mock"]
    if settings.llm_provider == "sarvam":
        return ["sarvam"]
    if settings.llm_provider == "local":
        return ["local"]
    return (["local"] if settings.local_llm_url else []) + ["sarvam"]  # auto


def _model_name(provider: str) -> str:
    return {"mock": "mock", "local": settings.local_llm_model}.get(provider, settings.llm_model)


def _chat(provider: str, messages: list[dict], reasoning_effort: str) -> tuple[str, dict]:
    if provider == "local":
        if not settings.local_llm_model:
            raise local.LocalError("LOCAL_LLM_MODEL is not set")
        data = local.chat({
            "model": settings.local_llm_model,
            "messages": messages,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
            "response_format": {"type": "json_object"},
        })
    else:
        data = sarvam.post("/v1/chat/completions", {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
            # The API accepts only low/medium/high (despite its docs); null disables reasoning.
            "reasoning_effort": None if reasoning_effort == "none" else reasoning_effort,
            "response_format": {"type": "json_object"},
        })
    try:
        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        raise LLMError(f"unexpected response shape from {provider}") from e
    if not content and choice.get("finish_reason") == "length":
        raise LLMError("response truncated by max_tokens (reasoning used the whole budget)")
    return content, data.get("usage") or {}


def _run[T: BaseModel](
    provider: str, *, pid: str, stage: str, prompt: str, model: type[T], mock_fn: Callable[[], T],
    system: str, check: Callable[[T], None] | None, reasoning_effort: str | None,
) -> T:
    mock = provider == "mock"
    schema = json.dumps(model.model_json_schema())
    messages = [
        {"role": "system", "content": f"{system}\n\nRespond with ONE JSON object and nothing else, "
                                      f"matching this JSON Schema:\n{schema}".strip()},
        {"role": "user", "content": prompt},
    ]
    last_error = ""
    for attempt in range(MAX_ATTEMPTS):
        start = time.monotonic()
        ok, response, usage, result = False, "", {}, None
        try:
            if mock:
                result = model.model_validate(mock_fn())
                response = result.model_dump(mode="json")
            else:
                response, usage = _chat(provider, messages, reasoning_effort or settings.reasoning_effort)
                result = model.model_validate(_extract_json(response))
            if check:
                check(result)
            ok = True
            return result
        except (ValueError, ValidationError) as e:  # JSONDecodeError and pydantic errors are ValueErrors
            last_error = str(e)[:500]
            if mock:
                raise
            if isinstance(e, SoftCheck) and attempt == MAX_ATTEMPTS - 1 and result is not None:
                return result
            messages += [
                {"role": "assistant", "content": response},
                {"role": "user", "content": f"That was invalid: {last_error}\nReturn corrected JSON only."},
            ]
        finally:
            _log({
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "performance_id": pid,
                "stage": stage,
                "provider": provider,
                "model": _model_name(provider),
                "temperature": settings.llm_temperature,
                "prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest(),
                "prompt": prompt[:4000],
                "response": response,
                "tokens": usage,
                "latency_ms": round((time.monotonic() - start) * 1000),
                "validation_ok": ok,
                "retry": attempt,
            })
    raise LLMError(f"{stage}: invalid output after {MAX_ATTEMPTS} attempts: {last_error}")


def complete_json[T: BaseModel](
    *, pid: str, stage: str, prompt: str, model: type[T], mock_fn: Callable[[], T],
    system: str = "", check: Callable[[T], None] | None = None, reasoning_effort: str | None = None,
) -> T:
    """check() may raise ValueError to reject bad output (triggers the retry). Raising SoftCheck
    instead means the final attempt is accepted anyway, for problems a later stage can fix.

    With several providers (LLM_PROVIDER=auto) a failure of one - unreachable, an HTTP error or
    output that never validates - moves on to the next; only the last provider's failure is raised."""
    _calls[pid] += 1
    order = providers()
    for i, provider in enumerate(order):
        try:
            return _run(
                provider, pid=pid, stage=stage, prompt=prompt, model=model, mock_fn=mock_fn,
                system=system, check=check, reasoning_effort=reasoning_effort,
            )
        except (LLMError, local.LocalError, sarvam.SarvamError) as e:
            if i == len(order) - 1:
                raise
            _log({
                "at": datetime.now(UTC).isoformat(timespec="seconds"), "performance_id": pid,
                "stage": stage, "provider": provider, "fallback_to": order[i + 1],
                "reason": f"{type(e).__name__}: {str(e)[:300]}",
            })
    raise AssertionError("unreachable")
