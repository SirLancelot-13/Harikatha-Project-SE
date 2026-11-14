import json

from fastapi.testclient import TestClient

from app import mixer
from app.config import settings
from app.main import app

client = TestClient(app)


def test_create_and_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    r = client.post("/api/v1/performances", json={"title": "Gajendra", "story": "x"})
    assert r.status_code == 201
    pid = r.json()["id"]
    assert client.get(f"/api/v1/performances/{pid}/status").json()["stage"] == "scripted"
    assert client.get("/api/v1/performances/nope").status_code == 404


def test_script_versions_are_immutable(tmp_path, monkeypatch):
    from app import store
    from app.schemas import GenerateRequest

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    pid = store.create(GenerateRequest(title="t", story="s"))
    d = settings.performances_dir / pid

    meta = store.read(pid, "meta.json")
    assert meta["settings"]["tts_model"] == "bulbul:v3"
    assert "sarvam_api_key" not in meta["settings"]

    assert store.save_script(pid, {"segments": ["a"]}, "initial") == 1
    assert store.save_script(pid, {"segments": ["b"]}, "fit round 1") == 2

    assert store.read(pid, "script.v1.json") == {"segments": ["a"]}
    assert store.read(pid, "script.json") == {"segments": ["b"]}
    assert [v["v"] for v in store.read(pid, "meta.json")["script_versions"]] == [1, 2]
    assert store.write_status(pid, "scripted")["script_version"] == 2
    assert (d / "script.v2.json").exists()


def test_script_job_writes_stage_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    r = client.post("/api/v1/performances", json={
        "title": "Gajendra", "story": "An elephant is caught. Vishnu saves him.",
        "duration_minutes": 6,
    })
    pid = r.json()["id"]
    d = settings.performances_dir / pid
    for name in ["01_analysis", "02_budget", "03_retrieval", "05_critique"]:
        assert (d / "stages" / f"{name}.json").exists()
    assert (d / "stages" / "04_drafts" / "invocation.json").exists()
    assert (d / "script.v1.json").exists()

    got = client.get(f"/api/v1/performances/{pid}").json()
    assert got["status"]["script_version"] == 1
    # invocation, opening song, intro, 2 story parts (the story has 2 events), shloka + meaning, closing verses, mangalam
    assert [s["section"] for s in got["script"]["segments"]] == [
        "invocation", "opening_song", "prastavana", "nirupana_1", "nirupana_2",
        "shloka_meaning", "shloka_meaning", "closing_verses", "mangalam",
    ]
    assert list((tmp_path / "runtime" / "llm_log").glob("*.jsonl"))


def _scripted(tmp_path, monkeypatch, minutes=2):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    r = client.post("/api/v1/performances", json={
        "title": "G", "story": "An elephant is caught. Vishnu saves him.", "duration_minutes": minutes,
    })
    return r.json()["id"]


def test_render_produces_timeline_and_audio(tmp_path, monkeypatch):
    pid = _scripted(tmp_path, monkeypatch)
    assert client.post(f"/api/v1/performances/{pid}/render").status_code == 202
    status = client.get(f"/api/v1/performances/{pid}/status").json()
    assert status["stage"] == "done", status
    tl = client.get(f"/api/v1/performances/{pid}").json()["timeline"]
    assert tl["script_version"] == status["script_version"]
    assert tl["segments"][0]["start"] == mixer.PRE_ROLL  # drone and bell play first
    assert all(a["end"] < b["start"] for a, b in zip(tl["segments"], tl["segments"][1:]))
    assert client.get(f"/api/v1/performances/{pid}/audio?fmt=wav").status_code == 200
    assert client.get(f"/api/v1/performances/{pid}/clips/invocation.1").status_code == 200
    assert client.get(f"/api/v1/performances/{pid}/clips/..%2Fstatus").status_code == 404


def test_fit_round_shortens_overrunning_segments(tmp_path, monkeypatch):
    from app import tts

    pid = _scripted(tmp_path, monkeypatch)
    monkeypatch.setattr(tts, "MOCK_CHARS_PER_SEC", 1.0)  # far slower than budget -> overrun
    client.post(f"/api/v1/performances/{pid}/render")
    d = settings.performances_dir / pid
    assert (d / "stages" / "07_fit_round1.json").exists(), client.get(f"/api/v1/performances/{pid}/status").json()
    versions = [v["reason"] for v in json.loads((d / "meta.json").read_text())["script_versions"]]
    assert any(r.startswith("fit round 1") for r in versions)


def test_edit_segment_and_guards(tmp_path, monkeypatch):
    pid = _scripted(tmp_path, monkeypatch)
    url = f"/api/v1/performances/{pid}/segments/prastavana.1"
    r = client.patch(url, json={"text": "new text", "delivery": {"pace": 0.9}})
    assert r.status_code == 200 and r.json()["script_version"] == 2
    segs = {s["id"]: s for s in client.get(f"/api/v1/performances/{pid}").json()["script"]["segments"]}
    assert segs["prastavana.1"]["text"] == "new text"
    assert client.patch(url.replace("prastavana.1", "nope"), json={"text": "x"}).status_code == 404
    rating = {"rater_id": "r1", "clarity": 4, "musicality": 3, "harikatha_feel": 4, "fidelity": 5}
    assert client.post(f"/api/v1/performances/{pid}/ratings", json=rating).status_code == 201
    assert client.post(f"/api/v1/performances/{pid}/ratings", json={**rating, "clarity": 9}).status_code == 422
    assert (tmp_path / "runtime" / "evals" / "ratings.jsonl").exists()


def test_render_requires_script(tmp_path, monkeypatch):
    from app import store
    from app.schemas import GenerateRequest

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    pid = store.create(GenerateRequest(title="t", story="s"))  # no job run -> still 'analysing'
    assert client.post(f"/api/v1/performances/{pid}/render").status_code == 409
