"""Live-mode code paths, with Sarvam's HTTP API replaced by httpx.MockTransport."""
import base64
import io
import json
import wave

import httpx
import pytest

from app import calibration, llm, planner, sarvam, tts
from app.config import settings
from app.schemas import Critique, SectionBudget, Slot, StoryAnalysis


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "mock_sarvam", False)
    monkeypatch.setattr(settings, "sarvam_api_key", "sk_test_key")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "retry_delay_sec", 0)
    requests: list[httpx.Request] = []

    def install(handler):
        def wrapped(req: httpx.Request) -> httpx.Response:
            requests.append(req)
            return handler(req)

        monkeypatch.setattr(sarvam, "transport", httpx.MockTransport(wrapped))
        return requests

    return install


def _chat_reply(content: str, finish="stop") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}],
                                     "usage": {"total_tokens": 5}})


def _wav_b64(rate=24000, seconds=0.5) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x01" * int(rate * seconds))
    return base64.b64encode(buf.getvalue()).decode()


def test_chat_request_shape_and_auth(live):
    reqs = live(lambda r: _chat_reply('{"ok": true, "issues": []}'))
    out = llm.complete_json(pid="p", stage="critique", prompt="hi", model=Critique,
                            mock_fn=lambda: Critique(ok=False))
    assert out.ok is True
    req = reqs[0]
    assert req.url.path == "/v1/chat/completions"
    assert req.headers["api-subscription-key"] == "sk_test_key"
    body = json.loads(req.content)
    assert body["model"] == "sarvam-105b" and body["max_tokens"] == 6000
    assert body["reasoning_effort"] == "low" and body["response_format"] == {"type": "json_object"}
    log = next((settings.data_dir / "runtime" / "llm_log").glob("*.jsonl")).read_text()
    assert "sk_test_key" not in log


def test_invalid_json_is_retried_with_feedback(live):
    replies = iter(["not json at all", '```json\n{"ok": true}\n```'])
    reqs = live(lambda r: _chat_reply(next(replies)))
    out = llm.complete_json(pid="p", stage="critique", prompt="hi", model=Critique, mock_fn=lambda: None)
    assert out.ok
    assert "invalid" in json.loads(reqs[1].content)["messages"][-1]["content"]


def test_gives_up_after_two_bad_answers(live):
    live(lambda r: _chat_reply("nope"))
    with pytest.raises(llm.LLMError):
        llm.complete_json(pid="p", stage="critique", prompt="hi", model=Critique, mock_fn=lambda: None)


def test_truncated_reasoning_is_reported(live):
    live(lambda r: _chat_reply("", finish="length"))
    with pytest.raises(llm.LLMError, match="max_tokens"):
        llm.complete_json(pid="p", stage="s", prompt="hi", model=Critique, mock_fn=lambda: None)


def test_retries_429_then_succeeds_and_hides_key_on_failure(live):
    statuses = iter([429, 503, 200])

    def handler(r):
        s = next(statuses)
        return _chat_reply('{"ok": true}') if s == 200 else httpx.Response(s, json={"error": {"message": "slow down"}})

    reqs = live(handler)
    assert llm.complete_json(pid="p", stage="s", prompt="x", model=Critique, mock_fn=lambda: None).ok
    assert len(reqs) == 3

    live(lambda r: httpx.Response(403, json={"error": {"message": "bad key", "code": "invalid_api_key_error"}}))
    with pytest.raises(sarvam.SarvamError, match="bad key") as e:
        sarvam.post("/v1/chat/completions", {})
    assert "sk_test_key" not in str(e.value)


def test_missing_key_fails_fast(live, monkeypatch):
    monkeypatch.setattr(settings, "sarvam_api_key", "")
    with pytest.raises(sarvam.SarvamError, match="SARVAM_API_KEY"):
        sarvam.post("/x", {})


def test_tts_request_decode_and_cache(live):
    reqs = live(lambda r: httpx.Response(200, json={"request_id": "r", "audios": [_wav_b64()]}))
    samples, hit = tts.synthesize("Hello there.", "en-IN", pace=3.0)
    assert not hit and len(samples) == 12000
    body = json.loads(reqs[0].content)
    assert reqs[0].url.path == "/text-to-speech"
    assert body["model"] == "bulbul:v3" and body["speaker"] == "shubh" and body["language_code"] == "en-IN"
    assert body["pace"] == 2.0 and body["speech_sample_rate"] == 24000  # pace clamped to API range
    _, hit2 = tts.synthesize("Hello there.", "en-IN", pace=3.0)
    assert hit2 and len(reqs) == 1


def test_tts_rejects_wrong_sample_rate(live):
    live(lambda r: httpx.Response(200, json={"audios": [_wav_b64(rate=22050)]}))
    with pytest.raises(sarvam.SarvamError, match="22050"):
        tts.synthesize("Hi.", "en-IN")


def test_split_text_packs_sentences_and_cuts_long_ones():
    assert tts.split_text("One. Two. Three.", limit=100) == ["One. Two. Three."]
    assert tts.split_text("One. Two. Three.", limit=9) == ["One. Two.", "Three."]
    chunks = tts.split_text("word " * 200, limit=100)
    assert all(len(c) <= 100 for c in chunks) and " ".join(chunks).split() == ["word"] * 200
    assert tts.split_text("नमस्ते। राम आए।", limit=9) == ["नमस्ते।", "राम आए।"]


def _budget():
    return SectionBudget(section="nirupana_1", kind="nirupana", target_sec=60, events=["e"], slots=[
        Slot(type="narration", target_sec=20, target_chars=200),
        Slot(type="song", target_sec=20, target_chars=200),
    ])


def test_write_section_uses_corpus_verbatim_and_rejects_unoffered_ids(live, tmp_path):
    corpus = {"s1": {"id": "s1", "text": "VERBATIM TEXT", "title": "t"}}
    replies = iter([
        json.dumps({"slots": [{"text": "x"}, {"source_id": "invented"}]}),  # unoffered id -> retry
        json.dumps({"slots": [{"text": "A story."}, {"source_id": "s1"}]}),
    ])
    reqs = live(lambda r: _chat_reply(next(replies)))
    from app.schemas import GenerateRequest

    req = GenerateRequest(title="t", story="s")
    analysis = StoryAnalysis(characters=[], events=["e"], deity="Vishnu", bhava="bhakti", moral="m")
    out = planner.write_section("p", req, analysis, _budget(), ["s1"], corpus)
    segs = planner.assemble(_budget(), out, corpus)
    assert [s.origin for s in segs] == ["original", "traditional"]
    assert segs[1].text == "VERBATIM TEXT" and segs[1].source_ids == ["s1"]
    assert json.loads(reqs[0].content)["reasoning_effort"] is None  # writing skips reasoning


def test_write_section_overlong_text_is_kept_after_one_retry_but_wrong_slot_count_fails(live):
    from app.schemas import GenerateRequest

    analysis = StoryAnalysis(characters=[], events=["e"], deity="d", bhava="b", moral="m")
    req = GenerateRequest(title="t", story="s")
    long_reply = json.dumps({"slots": [{"text": "x" * 900}, {"text": "ok"}]})
    reqs = live(lambda r: _chat_reply(long_reply))
    out = planner.write_section("p", req, analysis, _budget(), [], {})
    assert len(reqs) == 2 and len(out.slots[0].text) == 900  # one retry, then accepted: the fit loop shortens it

    live(lambda r: _chat_reply(json.dumps({"slots": [{"text": "only one"}]})))
    with pytest.raises(llm.LLMError, match="expected 2 slots"):
        planner.write_section("p", req, analysis, _budget(), [], {})


def test_critique_fixes_touch_only_original_segments(live):
    from app.schemas import Critique, CritiqueIssue, Script, Segment

    segs = [
        Segment(id="a.1", section="a", type="narration", text="Bad invented claim. " * 5, target_sec=5),
        Segment(id="a.2", section="a", type="song", text="TRAD", origin="traditional",
                source_ids=["s1"], target_sec=5),
    ]
    script = Script(title="t", language="en-IN", target_sec=10, segments=segs)
    verdict = Critique(ok=False, issues=[
        CritiqueIssue(segment_id="a.1", problem="invented", fix_instruction="remove claim"),
        CritiqueIssue(segment_id="a.2", problem="odd", fix_instruction="change"),
    ])
    live(lambda r: _chat_reply('{"text": "Fixed."}'))
    fixes = planner.apply_fixes("p", script, verdict, cps=10)
    assert [f["applied"] for f in fixes] == [True, False]
    assert script.segments[0].text == "Fixed." and script.segments[1].text == "TRAD"
    assert fixes[0]["before"].startswith("Bad invented")


def test_rewrite_falls_back_to_trim_when_model_ignores_limit(live):
    live(lambda r: _chat_reply(json.dumps({"text": "y" * 500})))
    seg = {"id": "a.1", "text": "First sentence. Second sentence. Third sentence."}
    assert planner.rewrite_segment("p", seg, "shorten", 20, "en-IN") == "First sentence."


def test_calibration_smooths_and_keys_by_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    assert calibration.get("en-IN") == calibration.FALLBACK_CPS
    calibration.update("en-IN", 1000, 100)  # 10 cps, first measurement taken as is
    assert calibration.get("en-IN") == 10.0
    calibration.update("en-IN", 2000, 100)  # 20 cps -> 0.7*10 + 0.3*20
    assert calibration.get("en-IN") == 13.0
    calibration.update("en-IN", 10, 1)  # too little text to trust
    assert calibration.get("en-IN") == 13.0
    monkeypatch.setattr(settings, "mock_sarvam", False)
    assert calibration.get("en-IN") == calibration.FALLBACK_CPS




# --- language handling ---------------------------------------------------------

def test_language_problem_detects_english_and_marathi_drift():
    from app import languages

    hindi = "आइए प्यारे श्रोताओ, आज हम हरि की कथा गाएंगे।"
    assert languages.problem(hindi, "hi-IN") is None
    assert languages.problem("Let us hear the story of Hari today", "hi-IN") is not None
    assert languages.problem("एकदा गजेंद्र नावाचा हत्ती तलावात खेळत होता", "hi-IN") == (
        "this looks like Marathi, not Hindi; use Hindi vocabulary and grammar")
    assert languages.problem("Let us hear the story of Hari today", "en-IN") is None
    assert languages.problem("Om", "hi-IN") is None  # too short to judge
    assert languages.problem("சொல்லுவோம் ஹரியின் கதையை இன்று", "ta-IN") is None
    assert languages.name("hi-IN") == "Hindi"


def _hindi_setup():
    from app.schemas import GenerateRequest

    analysis = StoryAnalysis(characters=[], events=["An elephant is caught."], deity="Vishnu", bhava="b", moral="m")
    return GenerateRequest(title="t", story="s", language="hi-IN"), analysis


def test_english_output_for_hindi_is_retried_with_feedback_then_accepted(live):
    english = json.dumps({"slots": [{"text": "Let us hear the story of the great elephant today"},
                                    {"text": "He cried out to the Lord for help and was saved"}]})
    hindi = json.dumps({"slots": [{"text": "आइए आज हम महान हाथी की कथा सुनें और भक्ति करें"},
                                  {"text": "उसने प्रभु को पुकारा और प्रभु ने उसे बचा लिया हे"}]})
    replies = iter([english, hindi])
    reqs = live(lambda r: _chat_reply(next(replies)))
    req, analysis = _hindi_setup()
    out = planner.write_section("p", req, analysis, _budget(), [], {})
    assert out.slots[0].text.startswith("आइए")
    first = json.loads(reqs[0].content)["messages"]
    assert "Hindi" in first[0]["content"] and "retell them in Hindi" in first[1]["content"]
    assert "mostly Latin letters" in json.loads(reqs[1].content)["messages"][-1]["content"]


def test_marathi_drift_gets_one_retry_and_is_not_fatal(live):
    marathi = json.dumps({"slots": [{"text": "एकदा गजेंद्र नावाचा हत्ती तलावात खेळत होता तेव्हा"},
                                    {"text": "प्रभूने त्याला वाचवले आहे आणि तो खूप आनंदी झाला"}]})
    reqs = live(lambda r: _chat_reply(marathi))
    req, analysis = _hindi_setup()
    out = planner.write_section("p", req, analysis, _budget(), [], {})
    assert len(reqs) == 2 and out.slots[0].text.startswith("एकदा")  # kept: a human can still fix it in the editor


def test_critique_failure_does_not_lose_the_script(tmp_path, monkeypatch):
    from app import planner as pl
    from app import store
    from app.schemas import GenerateRequest

    monkeypatch.setattr(settings, "data_dir", tmp_path)

    def boom(*a, **k):
        raise llm.LLMError("response truncated by max_tokens")

    monkeypatch.setattr(pl, "critique", boom)
    pid = store.create(GenerateRequest(title="t", story="An elephant is caught. Vishnu saves him."))
    pl.run_script_job(pid)
    assert store.read(pid, "status.json")["stage"] == "scripted"
    stage = store.read(pid, "stages/05_critique.json")
    assert stage["ok"] is True and "truncated" in stage["skipped"]


def test_writer_is_told_each_sections_purpose_and_sees_the_whole_verse_it_must_explain(live):
    from app.schemas import GenerateRequest

    verse = "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत। अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम्॥ " * 3  # > 160 chars
    corpus = {"v1": {"id": "v1", "title": "a verse", "text": verse}}
    req = GenerateRequest(title="t", story="s", language="hi-IN", duration_minutes=4)
    analysis = StoryAnalysis(characters=[], events=[f"e{i}" for i in range(6)], deity="Vishnu", bhava="bhakti", moral="surrender")
    budgets = {b.section: b for b in planner.budget(req, analysis, 13.2)}
    reply = json.dumps({"slots": [{"source_id": "v1"}, {"text": "इस श्लोक का अर्थ यह है कि प्रभु सदा रक्षा करते हैं और भक्त को शरण देते हैं।"}]})
    reqs = live(lambda r: _chat_reply(reply))
    out = planner.write_section("p", req, analysis, budgets["shloka_meaning"], ["v1"], corpus)
    assert out.slots[0].source_id == "v1"
    prompt = json.loads(reqs[0].content)["messages"][1]["content"]
    assert verse[:300] in prompt  # the full verse, not the 160-character preview used elsewhere
    assert "Purpose: first a shloka" in prompt and "unpacks that shloka's meaning" in prompt
    system = json.loads(reqs[0].content)["messages"][0]["content"]
    assert "MUST explain the shloka in the slot before it" in system

    reply2 = json.dumps({"slots": [{"text": "कथा का पहला भाग यहाँ सुनाया जाता है।"}]})
    reqs = live(lambda r: _chat_reply(reply2))
    planner.write_section("p", req, analysis, budgets["nirupana_1"], [], {})
    prompt = json.loads(reqs[-1].content)["messages"][1]["content"]  # the fixture's list keeps earlier requests
    assert "part 1 of 2 of the story" in prompt and "do not wrap up the whole story" in prompt
    assert "e0; e1; e2" in prompt and "e3" not in prompt  # contiguous events, nothing from the other part


def test_writer_asks_for_songs_as_even_rhyming_lines_and_retries_a_one_line_song(live):
    from app.schemas import GenerateRequest

    req = GenerateRequest(title="t", story="s", language="hi-IN", duration_minutes=4)
    analysis = StoryAnalysis(characters=[], events=["e1", "e2"], deity="Vishnu", bhava="bhakti", moral="m")
    song_budget = {b.section: b for b in planner.budget(req, analysis, 13.2)}["opening_song"]

    one_line = json.dumps({"slots": [{"text": "हे प्रभु हे दयानिधान करुणा की मूरत हमारी रक्षा करो आज"}]})
    reqs = live(lambda r: _chat_reply(one_line))
    out = planner.write_section("p", req, analysis, song_budget, [], {})
    assert len(reqs) == 2 and "\n" not in out.slots[0].text  # one retry asking for lines, then accepted (a later stage can fix it)
    system = json.loads(reqs[0].content)["messages"][0]["content"]
    assert "ONE LINE PER LINE" in system and "rhyming" in system and "8 to 12 syllables" in system
    assert "short rhyming lines, one per line" in json.loads(reqs[1].content)["messages"][-1]["content"]

    lines = json.dumps({"slots": [{"text": "हे प्रभु हे दयानिधान,\nकरुणा की तुम हो खान।\nसुन लो भक्त पुकार,\nकरो हमारा उद्धार॥"}]})
    reqs = live(lambda r: _chat_reply(lines))
    out = planner.write_section("p", req, analysis, song_budget, [], {})
    assert len(reqs) == 3 and out.slots[0].text.count("\n") == 3  # the fixture list is shared: exactly one new request
    segs = planner.assemble(song_budget, out, {})
    assert segs[0].text.count("\n") == 3  # line breaks survive assembly


def test_rewriting_a_song_keeps_it_as_lines(live):
    reqs = live(lambda r: _chat_reply(json.dumps({"text": "पहली,\nदूसरी॥"})))
    seg = {"id": "opening_song.1", "type": "song", "text": "कोई लंबा गीत\nजिसकी दो पंक्तियाँ हैं"}
    out = planner.rewrite_segment("p", seg, "shorten", 40, "hi-IN")
    assert out == "पहली,\nदूसरी॥"
    assert "keep it as short separate lines" in json.loads(reqs[-1].content)["messages"][1]["content"]
    narration = {"id": "x.1", "type": "narration", "text": "कोई कथा"}
    planner.rewrite_segment("p", narration, "shorten", 40, "hi-IN")
    assert "keep it as short separate lines" not in json.loads(reqs[-1].content)["messages"][1]["content"]
