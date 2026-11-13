"""Local (open-source) providers with fallback to Sarvam. No network, no models: HTTP is mocked."""
import base64
import io
import json
import sys
import wave
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import calibration, llm, local, render, sarvam, tts
from app.config import settings
from app.main import app as backend_app
from app.schemas import Critique

RATE = 24000
LOCAL_HEALTH = {"ok": True, "engine": "stub", "sample_rate": RATE, "languages": ["hi-IN", "en-IN"]}


def wav_bytes(seconds=0.5, rate=RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x01" * int(rate * seconds))
    return buf.getvalue()


@pytest.fixture
def live(tmp_path, monkeypatch):
    """Live mode (not mock) with Sarvam + local transports that record every request."""
    monkeypatch.setattr(settings, "mock_sarvam", False)
    monkeypatch.setattr(settings, "sarvam_api_key", "sk_test_key")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "retry_delay_sec", 0)
    seen = {"sarvam": [], "local": []}

    def install(local_handler=None, sarvam_handler=None):
        def mk(name, handler):
            def wrapped(req):
                seen[name].append(req)
                return handler(req)
            return httpx.MockTransport(wrapped)

        def dead(req):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(local, "transport", mk("local", local_handler or dead))
        monkeypatch.setattr(sarvam, "transport", mk("sarvam", sarvam_handler or (lambda r: httpx.Response(500))))
        return seen

    return install


def healthy_local(synth=None):
    def handler(req):
        if req.url.path == "/health":
            return httpx.Response(200, json=LOCAL_HEALTH)
        return synth(req) if synth else httpx.Response(200, content=wav_bytes(), headers={"content-type": "audio/wav"})
    return handler


def sarvam_tts(req):
    return httpx.Response(200, json={"audios": [base64.b64encode(wav_bytes()).decode()]})


# --- choosing the backend -----------------------------------------------------

def test_default_provider_is_sarvam_and_never_probes_local(live):
    seen = live(local_handler=healthy_local())
    assert tts.pick_backend("hi-IN") == ("sarvam", None)
    assert seen["local"] == []


def test_mock_mode_always_wins(live, monkeypatch):
    seen = live(local_handler=healthy_local())
    monkeypatch.setattr(settings, "mock_sarvam", True)
    monkeypatch.setattr(settings, "tts_provider", "local")
    assert tts.pick_backend("hi-IN") == ("mock", None) and seen["local"] == []


def test_auto_uses_local_when_it_is_healthy_and_supports_the_language(live, monkeypatch):
    live(local_handler=healthy_local())
    monkeypatch.setattr(settings, "tts_provider", "auto")
    assert tts.pick_backend("hi-IN") == ("local", None)
    backend, why = tts.pick_backend("ta-IN")  # not in the server's language list
    assert backend == "sarvam" and "does not support ta-IN" in why


def test_auto_falls_back_when_local_is_down_and_local_mode_refuses(live, monkeypatch):
    live()  # local transport refuses connections
    monkeypatch.setattr(settings, "tts_provider", "auto")
    backend, why = tts.pick_backend("hi-IN")
    assert backend == "sarvam" and "not reachable" in why
    monkeypatch.setattr(settings, "tts_provider", "local")
    local.reset_health_cache()
    with pytest.raises(local.LocalError, match="not reachable"):
        tts.pick_backend("hi-IN")
    assert tts.planned_voice("hi-IN").startswith("bulbul")  # sizing a script never raises


def test_health_is_cached_so_a_dead_server_costs_one_probe(live, monkeypatch):
    seen = live()
    monkeypatch.setattr(settings, "tts_provider", "auto")
    for _ in range(5):
        tts.pick_backend("hi-IN")
    assert len(seen["local"]) == 1


def test_bad_health_payload_counts_as_unavailable(live):
    live(local_handler=lambda r: httpx.Response(200, json={"ok": True}))  # no languages/sample_rate
    assert local.health() is None


# --- local synthesis ------------------------------------------------------------

def test_local_synthesis_request_shape_cache_and_separate_voice_key(live, monkeypatch):
    seen = live(local_handler=healthy_local())
    monkeypatch.setattr(settings, "tts_provider", "local")
    samples, hit = tts.synthesize("नमस्ते।", "hi-IN", 0.9, "shloka", backend="local")
    assert len(samples) == RATE // 2 and not hit
    body = json.loads(next(r for r in seen["local"] if r.url.path == "/synthesize").content)
    assert body == {"text": "नमस्ते।", "language_code": "hi-IN", "pace": 0.9, "kind": "shloka"}
    assert tts.synthesize("नमस्ते।", "hi-IN", 0.9, "shloka", backend="local")[1] is True
    assert tts.cache_key("x", "hi-IN", 1.0, "local", "shloka") != tts.cache_key("x", "hi-IN", 1.0, "sarvam", "shloka")
    assert tts.cache_key("x", "hi-IN", 1.0, "local", "shloka") != tts.cache_key("x", "hi-IN", 1.0, "local", "song")
    assert tts.voice_id("local") == "local/stub" and tts.voice_id("sarvam") == "bulbul:v3/shubh"
    assert not any("sk_test_key" in str(r.headers) for r in seen["local"])  # the Sarvam key never goes local


def test_local_audio_is_validated(live):
    live(local_handler=healthy_local(lambda r: httpx.Response(200, content=wav_bytes(rate=44100))))
    with pytest.raises(local.LocalError, match="44100"):
        tts.synthesize("Hello.", "en-IN", backend="local")
    live(local_handler=healthy_local(lambda r: httpx.Response(200, content=b"not a wav")))
    with pytest.raises(local.LocalError, match="unexpected WAV"):
        tts.synthesize("Hello again.", "en-IN", backend="local")
    live(local_handler=healthy_local(lambda r: httpx.Response(422, json={"detail": "nope"})))
    with pytest.raises(local.LocalError, match="422: nope"):
        tts.synthesize("Third.", "en-IN", backend="local")


# --- one voice per performance ---------------------------------------------------

def script_of(n=3):
    return {"language": "hi-IN", "segments": [
        {"id": f"s.{i}", "type": "narration", "text": f"वाक्य संख्या {i}।", "delivery": {"pace": 1.0}} for i in range(n)
    ]}


def test_a_midway_local_failure_redoes_the_whole_script_with_sarvam(live, monkeypatch):
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        if calls["n"] == 2:
            return httpx.Response(500, json={"detail": "CUDA out of memory"})
        return httpx.Response(200, content=wav_bytes())

    seen = live(local_handler=healthy_local(flaky), sarvam_handler=sarvam_tts)
    monkeypatch.setattr(settings, "tts_provider", "auto")
    script = script_of(3)
    backend, why = tts.pick_backend("hi-IN")
    clips, _, backend, why = render._synth_with_fallback(script, backend, why)
    assert backend == "sarvam" and "CUDA out of memory" in why
    assert len(seen["sarvam"]) == 3 and set(clips) == {"s.0", "s.1", "s.2"}  # every segment, one voice


def test_local_only_mode_never_falls_back(live, monkeypatch):
    live(local_handler=healthy_local(lambda r: httpx.Response(500, json={"detail": "boom"})), sarvam_handler=sarvam_tts)
    monkeypatch.setattr(settings, "tts_provider", "local")
    with pytest.raises(local.LocalError, match="boom"):
        render._synth_with_fallback(script_of(2), "local", None)


def test_render_job_records_which_backend_made_the_voice(tmp_path, monkeypatch, live):
    client = TestClient(backend_app)
    monkeypatch.setattr(settings, "mock_sarvam", True)  # make the script offline, in mock mode
    pid = client.post("/api/v1/performances", json={"title": "G", "story": "A. B.", "duration_minutes": 2}).json()["id"]

    def synth(req):
        text = json.loads(req.content)["text"]
        return httpx.Response(200, content=wav_bytes(seconds=len(text) / 14.0))

    live(local_handler=healthy_local(synth))
    monkeypatch.setattr(settings, "mock_sarvam", False)  # then render live, with the local voice
    monkeypatch.setattr(settings, "tts_provider", "auto")
    client.post(f"/api/v1/performances/{pid}/render")
    status = client.get(f"/api/v1/performances/{pid}/status").json()
    assert status["stage"] == "done", status
    meta = json.loads((settings.performances_dir / pid / "meta.json").read_text())
    assert status["tts_backend"] == meta["tts_backend"] == "local" and meta["tts_fallback"] is None
    assert any(k.startswith(("hi-IN|local/stub", "en-IN|local/stub"))
               for k in json.loads((tmp_path / "runtime" / "calibration.json").read_text()))


def test_calibration_is_kept_per_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    calibration.update("hi-IN", 1000, 100, "local/indic-parler-tts")  # 10 cps
    calibration.update("hi-IN", 3000, 100)  # 30 cps on the default (Sarvam) voice
    assert calibration.get("hi-IN", "local/indic-parler-tts") == 10.0
    assert calibration.get("hi-IN") == 30.0 and calibration.get("hi-IN", "local/other") == calibration.FALLBACK_CPS


# --- the LLM: local first, Sarvam as the fallback ----------------------------------

def chat_ok(content='{"ok": true, "issues": []}'):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})


def ask():
    return llm.complete_json(pid="p", stage="critique", prompt="hi", model=Critique, mock_fn=lambda: None)


@pytest.fixture
def llm_env(monkeypatch):
    monkeypatch.setattr(settings, "local_llm_url", "http://127.0.0.1:11434/v1")
    monkeypatch.setattr(settings, "local_llm_model", "qwen-test")


def test_local_llm_is_used_first_and_gets_no_sarvam_key_or_reasoning_flag(live, llm_env, monkeypatch):
    seen = live(local_handler=lambda r: chat_ok(), sarvam_handler=lambda r: chat_ok('{"ok": false}'))
    monkeypatch.setattr(settings, "llm_provider", "auto")
    assert ask().ok is True and seen["sarvam"] == []
    req = seen["local"][0]
    assert req.url.path == "/v1/chat/completions"
    body = json.loads(req.content)
    assert body["model"] == "qwen-test" and "reasoning_effort" not in body
    assert body["response_format"] == {"type": "json_object"}
    assert "api-subscription-key" not in req.headers and "sk_test_key" not in str(req.headers)
    log = next((settings.data_dir / "runtime" / "llm_log").glob("*.jsonl")).read_text()
    assert '"provider": "local"' in log and "sk_test_key" not in log


def test_llm_falls_back_to_sarvam_when_local_is_down(live, llm_env, monkeypatch):
    seen = live(sarvam_handler=lambda r: chat_ok('{"ok": false, "issues": []}'))
    monkeypatch.setattr(settings, "llm_provider", "auto")
    assert ask().ok is False and len(seen["sarvam"]) == 1
    assert seen["sarvam"][0].headers["api-subscription-key"] == "sk_test_key"
    log = (next((settings.data_dir / "runtime" / "llm_log").glob("*.jsonl"))).read_text()
    assert '"fallback_to": "sarvam"' in log


def test_llm_falls_back_when_the_local_model_never_produces_valid_json(live, llm_env, monkeypatch):
    seen = live(local_handler=lambda r: chat_ok("I cannot do that"), sarvam_handler=lambda r: chat_ok())
    monkeypatch.setattr(settings, "llm_provider", "auto")
    assert ask().ok is True
    assert len(seen["local"]) == 2 and len(seen["sarvam"]) == 1  # one retry locally, then Sarvam


def test_llm_local_only_does_not_fall_back_and_explains_misconfiguration(live, llm_env, monkeypatch):
    seen = live(sarvam_handler=lambda r: chat_ok())
    monkeypatch.setattr(settings, "llm_provider", "local")
    with pytest.raises(local.LocalError, match="network error"):
        ask()
    assert seen["sarvam"] == []
    monkeypatch.setattr(settings, "local_llm_url", "")
    with pytest.raises(local.LocalError, match="LOCAL_LLM_URL"):
        ask()
    monkeypatch.setattr(settings, "local_llm_url", "http://x/v1")
    monkeypatch.setattr(settings, "local_llm_model", "")
    with pytest.raises(local.LocalError, match="LOCAL_LLM_MODEL"):
        ask()


def test_auto_without_a_local_llm_url_just_uses_sarvam(live, monkeypatch):
    seen = live(sarvam_handler=lambda r: chat_ok())
    monkeypatch.setattr(settings, "llm_provider", "auto")
    assert ask().ok and seen["local"] == []


# --- the real server app, wired to the real client ------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "local_tts"))
import tts_server


@pytest.fixture
def server():
    return TestClient(tts_server.create_app(tts_server.StubEngine()))


def test_server_contract(server):
    h = server.get("/health").json()
    assert h["ok"] and h["sample_rate"] == RATE and "hi-IN" in h["languages"]
    short = server.post("/synthesize", json={"text": "नमस्ते।", "language_code": "hi-IN"})
    long = server.post("/synthesize", json={"text": "नमस्ते। " * 20, "language_code": "hi-IN", "kind": "shloka"})
    for r in (short, long):
        assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
        with wave.open(io.BytesIO(r.content)) as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, RATE)
    assert len(long.content) > len(short.content)
    bad = [{"text": "x", "language_code": "xx-IN"}, {"text": "x" * 1001, "language_code": "hi-IN"},
           {"text": "x", "language_code": "hi-IN", "pace": 5}, {"text": "", "language_code": "hi-IN"},
           {"text": "x", "language_code": "hi-IN", "kind": "shout"}]
    assert [server.post("/synthesize", json=b).status_code for b in bad] == [422] * 5


def test_server_reports_engine_failures_as_500(monkeypatch):
    engine = tts_server.StubEngine()
    monkeypatch.setattr(engine, "synthesize", lambda *a: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")))
    r = TestClient(tts_server.create_app(engine)).post("/synthesize", json={"text": "x", "language_code": "hi-IN"})
    assert r.status_code == 500 and "CUDA out of memory" in r.json()["detail"]


def test_voice_description_and_speaker_coverage():
    d = tts_server.describe("Rohit", "shloka", 0.8)
    assert d.startswith("Rohit speaks slowly") and "devotional" in d and "very close up" in d
    assert "fast" in tts_server.describe("Rohit", "narration", 1.5)
    assert "moderate speed" in tts_server.describe("Rohit", "narration", 1.0)
    assert set(tts_server.PARLER_LANGUAGES) == set(tts_server.DEFAULT_SPEAKERS)
    assert "pa-IN" not in tts_server.PARLER_LANGUAGES  # only "unofficial" on the model card


def test_backend_client_and_server_agree_end_to_end(server, live, monkeypatch):
    """The real backend client talking to the real server app (stub engine): the contract holds."""
    seen = live(sarvam_handler=sarvam_tts)
    monkeypatch.setattr(local, "transport", server._transport)
    monkeypatch.setattr(settings, "tts_provider", "auto")
    backend, why = tts.pick_backend("hi-IN")
    assert (backend, why) == ("local", None) and tts.voice_id("local") == "local/stub"
    samples, _ = tts.synthesize("नमस्ते। यह एक परीक्षण है।", "hi-IN", 1.0, "shloka", backend="local")
    assert len(samples) > RATE // 4 and seen["sarvam"] == []
    assert tts.pick_backend("pa-IN")[0] == "sarvam"  # the stub does not serve Punjabi: fall back
    assert np.abs(np.asarray(samples, dtype=np.float32)).max() > 0
