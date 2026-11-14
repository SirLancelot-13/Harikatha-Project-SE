"""The shape of a performance: opening shloka and song, intro, story parts, a shloka with its meaning,
closing verses and a final shloka. Budgets, retrieval, event splitting and the 2-minute minimum."""
import itertools
import json

import pytest
from fastapi.testclient import TestClient

from app import mixer, planner
from app.config import settings
from app.main import app
from app.schemas import GenerateRequest, StoryAnalysis

client = TestClient(app)
EVENTS = [f"Event {i}." for i in range(1, 11)]


def make(minutes, events=EVENTS, cps=13.2):
    req = GenerateRequest(title="t", story="s", duration_minutes=minutes)
    analysis = StoryAnalysis(characters=[], events=list(events), deity="Vishnu", bhava="bhakti", moral="m")
    return planner.budget(req, analysis, cps)


def sections(budgets):
    return [b.section for b in budgets]


@pytest.mark.parametrize("minutes", [2, 3, 4, 8, 20, 60])
def test_structure_order_and_total_time(minutes):
    budgets = make(minutes)
    names = sections(budgets)
    story = [n for n in names if n.startswith("nirupana")]
    assert names == ["invocation", "opening_song", "prastavana", *story, "shloka_meaning", "closing_verses", "mangalam"]
    types = {b.section: [s.type for s in b.slots] for b in budgets}
    assert types["invocation"] == ["shloka"] and types["opening_song"] == ["song"]
    assert types["shloka_meaning"] == ["shloka", "narration"]  # the verse, then its meaning
    assert types["closing_verses"] == ["song"] and types["mangalam"] == ["shloka"]
    assert all(types[n] == ["narration"] for n in story)
    total = sum(b.target_sec for b in budgets)
    assert total == pytest.approx(minutes * 60, abs=1.0)  # the story parts absorb whatever the other blocks leave


@pytest.mark.parametrize("minutes", [2, 4, 8])
def test_the_story_keeps_most_of_the_time_and_minimums_hold(minutes):
    budgets = make(minutes)
    story = sum(b.target_sec for b in budgets if b.section.startswith("nirupana"))
    assert story >= 0.5 * minutes * 60
    mins = {"invocation.shloka": 5, "opening_song.song": 12, "prastavana.narration": 10,
            "shloka_meaning.shloka": 6, "shloka_meaning.narration": 12, "closing_verses.song": 10, "mangalam.shloka": 5}
    for b in budgets:
        for s in b.slots:
            assert s.target_sec >= mins.get(f"{b.section}.{s.type}", 0) - 1e-9 and s.target_chars > 0


def test_two_minutes_is_one_story_part_of_about_a_minute():
    budgets = make(2)
    parts = [b for b in budgets if b.kind == "nirupana"]
    assert len(parts) == 1 and parts[0].target_sec == pytest.approx(60, abs=1)
    assert (parts[0].part, parts[0].parts) == (1, 1)


def test_story_parts_scale_with_duration_and_events_stay_in_story_order():
    for minutes, expected_parts in [(2, 1), (4, 2), (8, 4)]:
        parts = [b for b in make(minutes) if b.kind == "nirupana"]
        assert len(parts) == expected_parts and all(p.target_sec <= planner._STORY_PART_SEC for p in parts)
        assert [p.part for p in parts] == list(range(1, expected_parts + 1))
        assert all(p.parts == expected_parts for p in parts)
        assert [e for p in parts for e in p.events] == EVENTS  # contiguous chunks: the order is not scrambled
        sizes = [len(p.events) for p in parts]
        assert max(sizes) - min(sizes) <= 1


def test_parts_are_capped_by_the_number_of_events():
    parts = [b for b in make(8, events=["Only one event."]) if b.kind == "nirupana"]
    assert len(parts) == 1 and parts[0].events == ["Only one event."]
    assert len([b for b in make(8, events=["a", "b"]) if b.kind == "nirupana"]) == 2


def test_chars_follow_the_measured_speaking_speed():
    slow, fast = make(4, cps=10.0), make(4, cps=20.0)
    for a, b in zip(slow, fast, strict=True):
        for x, y in zip(a.slots, b.slots, strict=True):
            assert y.target_chars == pytest.approx(2 * x.target_chars, abs=1)


def test_a_one_minute_request_is_rejected():
    assert client.post("/api/v1/performances", json={"title": "t", "story": "s", "duration_minutes": 1}).status_code == 422
    assert client.post("/api/v1/performances", json={"title": "t", "story": "s", "duration_minutes": 2}).status_code == 201


def test_retrieval_offers_shlokas_and_songs_to_the_right_sections(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    rec = lambda i, kind: {"id": i, "kind": kind, "language": "hi-IN", "status": "verified", "deity": ["Vishnu"], "text": "x"}
    (tmp_path / "songs.json").write_text(json.dumps([rec("sh1", "shloka"), rec("so1", "song")]))
    req = GenerateRequest(title="t", story="s", language="hi-IN", duration_minutes=4)
    analysis = StoryAnalysis(characters=[], events=EVENTS, deity="Vishnu", bhava="b", moral="m")
    got = planner.retrieve(req, analysis, planner.budget(req, analysis, 13.2))
    offered = {k: v["offered_ids"] for k, v in got.items()}
    assert offered["invocation"] == offered["shloka_meaning"] == offered["mangalam"] == ["sh1"]
    assert offered["opening_song"] == offered["closing_verses"] == ["so1"]
    assert offered["nirupana_1"] == [] and offered["prastavana"] == []  # the story is always original narration


# --- how the mixer treats the new structure -------------------------------------------

RATE = 24000


def clip(seconds=2.0):
    import numpy as np

    t = np.arange(int(seconds * RATE)) / RATE
    return (sum(np.sin(2 * np.pi * 140 * h * t) / h for h in range(1, 6)) * 0.15).astype(np.float32)


def script_with(layout):
    segs = [{"id": f"{sec}.{i}", "section": sec, "type": typ, "delivery": {"pace": 1.0}} for i, (sec, typ) in enumerate(layout, 1)]
    return {"segments": segs}, {s["id"]: clip() for s in segs}


def test_pauses_for_story_parts_and_for_a_shloka_followed_by_its_meaning():
    layout = [("invocation", "shloka"), ("opening_song", "song"), ("prastavana", "narration"),
              ("nirupana_1", "narration"), ("nirupana_2", "narration"),
              ("shloka_meaning", "shloka"), ("shloka_meaning", "narration"),
              ("closing_verses", "song"), ("mangalam", "shloka")]
    script, clips = script_with(layout)
    tl = mixer.build(script, clips, RATE).timeline
    gaps = {f"{a['id']}>{b['id']}": b["start"] - a["end"] for a, b in itertools.pairwise(tl)}
    P = mixer.PAUSE
    assert gaps["nirupana_1.4>nirupana_2.5"] == pytest.approx(P["narration"], abs=0.005)  # one continuous telling
    assert gaps["shloka_meaning.6>shloka_meaning.7"] == pytest.approx(P["after_shloka"], abs=0.005)  # verse lands first
    assert gaps["invocation.1>opening_song.2"] == pytest.approx(P["section"], abs=0.005)
    assert gaps["prastavana.3>nirupana_1.4"] == pytest.approx(P["section"], abs=0.005)
    assert gaps["nirupana_2.5>shloka_meaning.6"] == pytest.approx(P["section"], abs=0.005)
    assert gaps["shloka_meaning.7>closing_verses.8"] == pytest.approx(P["section"], abs=0.005)


def test_one_chime_as_the_story_begins_not_one_per_part():
    layout = [("invocation", "shloka"), ("nirupana_1", "narration"), ("nirupana_2", "narration"),
              ("nirupana_3", "narration"), ("mangalam", "shloka")]
    script, clips = script_with(layout)
    recipe = mixer.build(script, clips, RATE).recipe
    assert len(recipe["bells_at_sec"]) == 3  # opening bell, one story chime, closing bell
