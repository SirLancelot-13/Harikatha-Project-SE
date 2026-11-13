import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import mixer, pitch
from app.config import settings
from app.main import app

RATE = 24000
client = TestClient(app)


def voiced(seconds=3.0, f0=140.0):
    """A harmonic, voice-like signal with a gentle amplitude wobble."""
    t = np.arange(int(seconds * RATE)) / RATE
    x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 12)) * (0.2 + 0.05 * np.sin(2 * np.pi * 3 * t))
    return x.astype(np.float32)


def f0_of(x, a, b):
    return mixer.estimate_f0(x[int(a * RATE): int(b * RATE)], RATE)


def test_lift_raises_pitch_by_the_requested_amount_in_the_middle():
    x = voiced()
    y, rep = pitch.lift(x, RATE, 2.0)
    assert rep["applied"] is True and rep["peak_st"] == 2.0
    before, after = f0_of(x, 1.0, 2.0), f0_of(y, 1.0, 2.0)
    assert after / before == pytest.approx(2 ** (2 / 12), rel=0.03)  # +2 semitones within 3%


def test_contour_eases_in_and_out_and_never_exceeds_the_peak():
    times = np.arange(0, 3.0, pitch.FRAME_MS / 1000)
    c = pitch.contour(times, 2.0)
    assert c[0] == pytest.approx(0.0, abs=1e-6) and c[-1] < 0.01  # no jump at either edge
    assert c.max() == pytest.approx(2.0, abs=1e-6) and np.argmax(c) > pitch.RAMP_SEC / (pitch.FRAME_MS / 1000) - 2
    assert np.all(np.diff(c[: int(pitch.RAMP_SEC / (pitch.FRAME_MS / 1000))]) >= 0)  # monotone rise


def test_lift_audibly_eases_in_on_real_audio():
    x = voiced()
    y, _ = pitch.lift(x, RATE, 3.0)
    edge = f0_of(y, 0.0, 0.3) / f0_of(x, 0.0, 0.3)
    middle = f0_of(y, 1.2, 1.8) / f0_of(x, 1.2, 1.8)
    assert edge < 2 ** (2.4 / 12) < 2 ** (2.6 / 12) < middle  # still rising at the start, full lift later


def test_lift_keeps_length_and_loudness_and_does_not_clip():
    x = voiced()
    y, _ = pitch.lift(x, RATE, 1.5)
    assert len(y) == len(x) and y.dtype == np.float32
    assert abs(20 * np.log10(mixer.rms(y) / mixer.rms(x))) < 1.0
    assert float(np.abs(y).max()) <= 1.0


def test_lift_is_capped_and_zero_lift_is_a_no_op():
    x = voiced()
    _, rep = pitch.lift(x, RATE, 10.0)
    assert rep["peak_st"] == pitch.MAX_LIFT_ST == 3.0
    same, rep0 = pitch.lift(x, RATE, 0.0)
    assert same is x and rep0["applied"] is False


def test_unvoiced_or_tiny_clips_are_left_alone():
    noise = (np.random.default_rng(1).standard_normal(RATE * 2) * 0.1).astype(np.float32)
    out, rep = pitch.lift(noise, RATE, 2.0)
    assert out is noise and rep["applied"] is False
    _, rep = pitch.lift(voiced(0.2), RATE, 2.0)
    assert rep["reason"] == "clip too short"


def test_a_vocoder_failure_never_fails_the_render(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("world crashed")

    monkeypatch.setattr(pitch.pw, "harvest", boom)
    x = voiced()
    out, rep = pitch.lift(x, RATE, 2.0)
    assert out is x and "world crashed" in rep["reason"]


def _render(tmp_path, monkeypatch, minutes=2):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    pid = client.post("/api/v1/performances", json={
        "title": "G", "story": "An elephant is caught. Vishnu saves him.", "duration_minutes": minutes,
    }).json()["id"]
    return pid, client


def test_shlokas_get_a_default_lift_and_other_segments_do_not(tmp_path, monkeypatch):
    pid, c = _render(tmp_path, monkeypatch)
    segs = c.get(f"/api/v1/performances/{pid}").json()["script"]["segments"]
    lifts = {s["id"]: s["delivery"]["pitch_lift_st"] for s in segs}
    assert lifts["invocation.1"] == lifts["mangalam.1"] == settings.shloka_lift_st
    assert all(v == 0 for k, v in lifts.items() if k not in {"invocation.1", "mangalam.1"})


def test_render_applies_the_lift_only_to_shlokas_and_keeps_the_timeline(tmp_path, monkeypatch):
    pid, c = _render(tmp_path, monkeypatch)
    c.post(f"/api/v1/performances/{pid}/render")
    assert c.get(f"/api/v1/performances/{pid}/status").json()["stage"] == "done"
    d = settings.performances_dir / pid
    rows = {r["id"]: r for r in json.loads((d / "stages" / "08_modulation.json").read_text())["segments"]}
    assert rows["invocation.1"]["pitch"]["applied"] is True
    assert rows["invocation.1"]["pitch"]["f0_target_hz"] > rows["invocation.1"]["pitch"]["f0_before_hz"]
    assert "pitch" not in rows["prastavana.1"]
    mix = json.loads((d / "stages" / "09_mix.json").read_text())
    assert mix["drone"]["shift_semitones"] is not None  # drone chosen from the unmodulated voice


def test_pitch_modulation_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "pitch_modulation", False)
    pid, c = _render(tmp_path, monkeypatch)
    c.post(f"/api/v1/performances/{pid}/render")
    rows = json.loads((settings.performances_dir / pid / "stages" / "08_modulation.json").read_text())["segments"]
    assert all("pitch" not in r for r in rows)


def test_editing_the_lift_is_validated_and_versioned(tmp_path, monkeypatch):
    pid, c = _render(tmp_path, monkeypatch)
    url = f"/api/v1/performances/{pid}/segments/invocation.1"
    ok = c.patch(url, json={"delivery": {"pitch_lift_st": 2.5}})
    assert ok.status_code == 200 and ok.json()["segment"]["delivery"]["pitch_lift_st"] == 2.5
    assert ok.json()["script_version"] == 2
    assert c.patch(url, json={"delivery": {"pitch_lift_st": 9}}).status_code == 422  # above the 3 st cap
    assert c.patch(url, json={"delivery": {"pitch_lift_st": -1}}).status_code == 422
