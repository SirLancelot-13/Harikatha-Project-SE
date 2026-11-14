"""Script generation, stages 1-5 of WORKFLOW_AND_STORAGE.md s2.2."""
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor

from . import calibration, languages, llm, sarvam, store, tts
from .config import settings
from .schemas import (
    Critique,
    Delivery,
    GenerateRequest,
    RewriteOutput,
    Script,
    SectionBudget,
    SectionOutput,
    Segment,
    Slot,
    SlotText,
    StoryAnalysis,
)

# The shape of a performance. Each slot is (type, share of the total time, minimum seconds); the minimums
# keep a 2-minute performance workable. The story parts ("nirupana") sit between the opening and the
# closing blocks and are ELASTIC: they take whatever time the other blocks leave (about 55-60%).
_OPENING = [
    ("invocation", [("shloka", 0.04, 5)]),
    ("opening_song", [("song", 0.08, 12)]),
    ("prastavana", [("narration", 0.08, 10)]),
]
_CLOSING = [
    ("shloka_meaning", [("shloka", 0.04, 6), ("narration", 0.10, 12)]),  # a shloka, then its meaning unpacked
    ("closing_verses", [("song", 0.06, 10)]),
    ("mangalam", [("shloka", 0.04, 5)]),
]
_STORY_PART_SEC = 75  # aim for story parts of about this length
_MAX_PARTS = 6

# What each section is for; passed to the writer.
_PURPOSE = {
    "invocation": "the opening invocation: a short prayer to the deity before the story begins",
    "opening_song": "a short devotional song that opens the performance, praising the deity and hinting at the theme",
    "prastavana": "a warm introduction to the audience that sets up the story and why it matters",
    "nirupana": "part {part} of {parts} of the story, told in order: cover only the events listed, flow naturally "
                "from the previous part, and do not wrap up the whole story unless this is the last part",
    "shloka_meaning": "first a shloka that captures the story's central teaching, then a narration that unpacks "
                      "that shloka's meaning in simple words and ties it back to the story",
    "closing_verses": "a short devotional verse offering the story's teaching and surrender to the Lord, "
                      "leading into the final blessing",
    "mangalam": "the closing benediction shloka that ends the performance",
}

_PERSONA = (
    "You are a Harikatha storyteller: you retell Hindu devotional stories as a spoken-and-sung "
    "performance. Write with warmth and bhakti, in plain language an audience can follow by ear."
)


def _load_corpus() -> dict[str, dict]:
    path = settings.data_dir / "songs.json"
    return {s["id"]: s for s in json.loads(path.read_text())} if path.exists() else {}


# --- stage 1 -------------------------------------------------------------

def analyse(pid: str, req: GenerateRequest) -> StoryAnalysis:
    prompt = f"Title: {req.title}\nStory:\n{req.story}"

    def mock() -> StoryAnalysis:
        events = [s.strip() for s in req.story.replace("\n", " ").split(".") if s.strip()]
        return StoryAnalysis(
            characters=[], events=events or [req.title], deity="unknown",
            bhava="bhakti", moral="[mock]",
        )

    return llm.complete_json(
        pid=pid, stage="analyse", prompt=prompt, model=StoryAnalysis, mock_fn=mock,
        system="Analyse the story for a Harikatha performance. List the characters, the key events "
               "in story order (each a short sentence, 4-12 events), the main deity (a single name, "
               "e.g. 'Vishnu'), the dominant bhava (e.g. bhakti, karuna, vira) and the moral.",
    )


# --- stage 2 -------------------------------------------------------------

def budget(req: GenerateRequest, analysis: StoryAnalysis, cps: float) -> list[SectionBudget]:
    total = req.duration_minutes * 60

    def block(section: str, specs: list[tuple[str, float, float]]) -> SectionBudget:
        slots = []
        for typ, share, minimum in specs:
            sec = max(float(minimum), total * share)
            slots.append(Slot(type=typ, target_sec=sec, target_chars=round(sec * cps)))
        return SectionBudget(
            section=section, kind=section, target_sec=sum(s.target_sec for s in slots), slots=slots
        )

    opening = [block(name, specs) for name, specs in _OPENING]
    closing = [block(name, specs) for name, specs in _CLOSING]
    story_total = max(25.0, total - sum(b.target_sec for b in opening + closing))
    n = min(_MAX_PARTS, max(1, math.ceil(story_total / _STORY_PART_SEC)), max(1, len(analysis.events)))
    each = story_total / n
    base, extra = divmod(len(analysis.events), n)  # contiguous chunks, in story order
    story, start = [], 0
    for i in range(n):
        size = base + (1 if i < extra else 0)
        story.append(SectionBudget(
            section=f"nirupana_{i + 1}", kind="nirupana", target_sec=each, part=i + 1, parts=n,
            events=analysis.events[start: start + size],
            slots=[Slot(type="narration", target_sec=each, target_chars=round(each * cps))],
        ))
        start += size
    return opening + story + closing


# --- stage 3 -------------------------------------------------------------

def retrieve(req: GenerateRequest, analysis: StoryAnalysis, budgets: list[SectionBudget]) -> dict:
    """Offer verified corpus records per section. Only IDs are stored."""
    corpus = _load_corpus().values()
    kinds = {"invocation": "shloka", "shloka_meaning": "shloka", "mangalam": "shloka",
             "opening_song": "song", "closing_verses": "song"}
    result = {}
    for b in budgets:
        kind = kinds.get(b.kind)
        offered = [
            s["id"] for s in corpus
            if s.get("status") == "verified" and s.get("language") == req.language
            and s.get("kind") == kind
            and (analysis.deity in s.get("deity", []) or not s.get("deity"))
        ]
        result[b.section] = {
            "query": {"deity": analysis.deity, "language": req.language, "kind": kind},
            "offered_ids": offered,
            "used_ids": [],
        }
    return result


# --- stage 4 -------------------------------------------------------------

def _mock_text(slot_type: str, b: SectionBudget) -> str:
    """Placeholder text for mock mode. Songs follow the same rule as real ones: several short lines."""
    if slot_type == "song":
        return f"[mock song for {b.section}]\n" + (" ".join(b.events) or "[mock second line]")
    return f"[mock {slot_type} for {b.section}] " + " ".join(b.events)


def write_section(
    pid: str, req: GenerateRequest, analysis: StoryAnalysis, b: SectionBudget, offered: list[str],
    corpus: dict[str, dict],
) -> SectionOutput:
    slot_lines = "\n".join(
        f"{i}. type={s.type}, about {s.target_chars} characters" for i, s in enumerate(b.slots, 1)
    )
    # For "a shloka and its meaning" the writer must see the whole verse it is about to explain.
    shown = 600 if b.kind == "shloka_meaning" else 160
    offered_lines = "\n".join(
        f"- id={i}: {corpus[i].get('title', '')} :: {corpus[i]['text'][:shown]}" for i in offered
    ) or "(none)"
    purpose = _PURPOSE[b.kind].format(part=b.part, parts=b.parts)
    lang = languages.name(req.language)
    prompt = (
        f"Write the spoken text in {lang}. The story and events below may be in another language: "
        f"retell them in {lang}, do not copy their wording.\n"
        f"Story: {req.title}\nDeity: {analysis.deity}; bhava: {analysis.bhava}; moral: {analysis.moral}\n"
        f"Characters: {', '.join(analysis.characters) or 'n/a'}\n"
        f"Section: {b.section} ({b.kind}). Purpose: {purpose}.\n"
        f"Events to cover in this section: {'; '.join(b.events) or 'n/a'}\n\nWrite exactly {len(b.slots)} slot(s), in order:\n{slot_lines}\n\n"
        f"Verified traditional texts you may use verbatim:\n{offered_lines}"
    )

    def check(out: SectionOutput) -> None:
        if len(out.slots) != len(b.slots):
            raise ValueError(f"expected {len(b.slots)} slots, got {len(out.slots)}")
        for i, (slot, want) in enumerate(zip(out.slots, b.slots, strict=True), 1):
            if slot.source_id and slot.source_id not in offered:
                raise ValueError(f"slot {i}: source_id {slot.source_id!r} was not offered")
            if not slot.source_id and not slot.text.strip():
                raise ValueError(f"slot {i}: empty text")
            if not slot.source_id and (why := languages.problem(slot.text, req.language)):
                err = llm.SoftCheck if "Marathi" in why else ValueError  # wrong script is fatal, drift is not
                raise err(f"slot {i}: {why}")
            if want.type == "song" and not slot.source_id and len([ln for ln in slot.text.splitlines() if ln.strip()]) < 2:
                raise llm.SoftCheck(f"slot {i}: write the song as 4 to 8 short rhyming lines, one per line")
            if not slot.source_id and len(slot.text) > want.target_chars * 1.25 + 20:
                raise llm.SoftCheck(f"slot {i}: {len(slot.text)} chars, limit is about {want.target_chars}")

    def mock() -> SectionOutput:
        return SectionOutput(slots=[
            SlotText(text=_mock_text(s.type, b)[: max(s.target_chars, 40)]) for s in b.slots
        ])

    return llm.complete_json(
        pid=pid, stage=f"write:{b.section}", prompt=prompt, model=SectionOutput, mock_fn=mock,
        check=check, reasoning_effort=settings.writing_reasoning_effort,
        system=_PERSONA + f" ALWAYS write in {lang}, in its own script, whatever language the story "
        f"is told in; keep names of people and places but transliterate them. Fill each slot in order. 'shloka' slots: a short devotional invocation or "
        "benediction. When a section is 'a shloka then its meaning', the narration slot MUST explain the "
        "shloka in the slot before it (if that slot uses an offered text, explain that text) in simple words. 'narration' slots: spoken storytelling that follows the events. 'song' slots: a "
        "short devotional song written as 4 to 8 short lines, ONE LINE PER LINE of the text (separated by "
        "newlines), each line about the same length (say 8 to 12 syllables), with the line endings rhyming "
        "(couplets AABB or alternating ABAB, or a repeated refrain line); it must sing well when read aloud. NEVER quote scripture or classical songs from memory and never claim "
        "a line comes from a scripture or composer. To use a traditional text, set that slot's "
        "source_id to one of the offered ids (text is then ignored); otherwise write original text "
        "and leave source_id null. Stay at or under the character count for each slot.",
    )


def assemble(b: SectionBudget, out: SectionOutput, corpus: dict[str, dict]) -> list[Segment]:
    """Turn raw LLM output into Segments: corpus text is inserted verbatim, never LLM-typed."""
    segments = []
    for i, (slot, text_out) in enumerate(zip(b.slots, out.slots, strict=True), 1):
        record = corpus.get(text_out.source_id) if text_out.source_id else None
        segments.append(Segment(
            id=f"{b.section}.{i}", section=b.section, type=slot.type,
            text=record["text"] if record else text_out.text.strip(),
            origin="traditional" if record else "original",
            source_ids=[text_out.source_id] if record else [],
            target_sec=slot.target_sec,
            delivery=Delivery(pitch_lift_st=min(3.0, settings.shloka_lift_st) if slot.type == "shloka" else 0.0),
        ))
    return segments


# --- stage 5 -------------------------------------------------------------

def critique(pid: str, script: Script, analysis: StoryAnalysis) -> Critique:
    lines = "\n".join(f"[{s.id}] ({s.type}, {s.origin}) {s.text}" for s in script.segments)
    lang = languages.name(script.language)
    prompt = (
        f"Expected language: {lang}\nDeity: {analysis.deity}; events: {'; '.join(analysis.events)}"
        f"\n\nScript:\n{lines}"
    )
    return llm.complete_json(
        pid=pid, stage="critique", prompt=prompt, model=Critique, mock_fn=lambda: Critique(ok=True),
        reasoning_effort=settings.critique_reasoning_effort,
        system="You review a Harikatha script. Flag only real problems: text that claims to quote a "
               "scripture or composer (likely invented), facts that contradict the listed events or "
               "deity, text not in the expected language (for Hindi: also flag Marathi or other "
               "languages that share the script), or incoherent lines. Segments with origin "
               "'traditional' are verified: never flag them. For each problem give the segment_id, "
               "the problem and a fix_instruction. Set ok=true and issues=[] if nothing is wrong.",
    )


def rewrite_segment(pid: str, seg: dict, instruction: str, max_chars: int, language: str) -> str:
    """Rewrite one original segment. Falls back to a hard trim if the model won't comply."""
    lang = languages.name(language)
    keep = "\nThis is a song: keep it as short separate lines (one per line) with the same rhyme." if seg.get("type") == "song" else ""
    prompt = f"Write in {lang}.{keep}\nInstruction: {instruction}\nMaximum {max_chars} characters.\nText:\n{seg['text']}"

    def check(out: RewriteOutput) -> None:
        if not out.text.strip():
            raise ValueError("empty text")
        if len(out.text) > max_chars * 1.1:
            raise ValueError(f"{len(out.text)} chars, maximum is {max_chars}")
        if why := languages.problem(out.text, language):
            raise ValueError(why)

    def mock() -> RewriteOutput:
        return RewriteOutput(text=trim(seg["text"], max_chars))

    try:
        return llm.complete_json(
            pid=pid, stage=f"rewrite:{seg['id']}", prompt=prompt, model=RewriteOutput, mock_fn=mock,
            check=check, reasoning_effort=settings.writing_reasoning_effort,
            system=_PERSONA + f" Rewrite the text as instructed, keeping its meaning, in {lang} "
            "and its devotional tone. Do not add scripture quotations.",
        ).text.strip()
    except llm.LLMError:
        return trim(seg["text"], max_chars)


def trim(text: str, max_chars: int) -> str:
    """Cut at the last sentence end (or space) within max_chars."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for mark in ".!?।॥":
        i = cut.rfind(mark)
        if i > max_chars * 0.4:
            return cut[: i + 1]
    return cut[: cut.rfind(" ")].rstrip() if " " in cut else cut


def apply_fixes(pid: str, script: Script, verdict: Critique, cps: float) -> list[dict]:
    """One rewrite pass over flagged *original* segments; traditional text is never touched."""
    by_id = {s.id: s for s in script.segments}
    fixes = []
    for issue in verdict.issues:
        seg = by_id.get(issue.segment_id)
        if seg is None or seg.origin != "original":
            fixes.append({"segment_id": issue.segment_id, "applied": False, "reason": "not an original segment"})
            continue
        before = seg.text
        seg.text = rewrite_segment(
            pid, seg.model_dump(), issue.fix_instruction, round(seg.target_sec * cps), script.language
        )
        fixes.append({"segment_id": seg.id, "applied": True, "problem": issue.problem,
                      "before": before, "after": seg.text})
    return fixes


# --- job -----------------------------------------------------------------

def run_script_job(pid: str) -> None:
    """Background job. Every stage writes its own artifact; failures land in status.json."""
    timings: dict[str, float] = {}
    t0 = time.monotonic()

    def lap(name: str) -> None:
        nonlocal t0
        timings[name] = round(time.monotonic() - t0, 2)
        t0 = time.monotonic()

    try:
        req = GenerateRequest.model_validate(store.read(pid, "request.json"))
        corpus = _load_corpus()

        store.write_status(pid, "analysing", progress=0.1)
        analysis = analyse(pid, req)
        store.write_stage(pid, "01_analysis.json", analysis.model_dump())
        lap("analyse")

        store.write_status(pid, "planning", progress=0.3)
        cps = calibration.get(req.language, tts.planned_voice(req.language))
        budgets = budget(req, analysis, cps)
        store.write_stage(pid, "02_budget.json", {
            "chars_per_sec": cps, "sections": [b.model_dump() for b in budgets],
        })
        retrieval = retrieve(req, analysis, budgets)
        store.write_stage(pid, "03_retrieval.json", retrieval)
        lap("plan")

        store.write_status(pid, "writing", progress=0.5)
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(
                lambda b: write_section(pid, req, analysis, b, retrieval[b.section]["offered_ids"], corpus),
                budgets,
            ))
        for b, out in zip(budgets, outputs, strict=True):
            store.write_stage(pid, f"04_drafts/{b.section}.json", out.model_dump())
        lap("write")

        script = Script(
            title=req.title, language=req.language, target_sec=req.duration_minutes * 60,
            segments=[s for b, out in zip(budgets, outputs, strict=True) for s in assemble(b, out, corpus)],
        )
        for sec, r in retrieval.items():
            r["used_ids"] = sorted({i for s in script.segments if s.section == sec for i in s.source_ids})
        store.write_stage(pid, "03_retrieval.json", retrieval)

        store.write_status(pid, "writing", "critique", progress=0.85)
        try:
            verdict = critique(pid, script, analysis)
        except (llm.LLMError, sarvam.SarvamError) as e:  # a quality pass must not lose the script
            verdict, skipped = Critique(ok=True), f"{type(e).__name__}: {e}"
        else:
            skipped = None
        fixes = apply_fixes(pid, script, verdict, cps)
        store.write_stage(pid, "05_critique.json", {
            **verdict.model_dump(), "fixes": fixes, **({"skipped": skipped} if skipped else {}),
        })
        lap("critique")

        reason = "initial + critique" if not any(f["applied"] for f in fixes) else "initial + critique fixes"
        store.save_script(pid, script.model_dump(mode="json"), reason)
        store.update_meta(
            pid, timings_sec=timings, llm_calls=llm.calls_for(pid),
            settings={**store.read(pid, "meta.json")["settings"], "chars_per_sec_used": cps},
        )
        store.write_status(pid, "scripted", progress=1.0)
    except Exception as e:  # noqa: BLE001 - surfaced to the UI via status.json
        store.write_status(pid, "error", f"{type(e).__name__}: {e}")
