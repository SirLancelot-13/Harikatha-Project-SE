"""Script generation, stages 1-5 of WORKFLOW_AND_STORAGE.md s2.2."""
import json
import time
from concurrent.futures import ThreadPoolExecutor

from . import calibration, languages, llm, store
from .config import settings
from .schemas import (
    Critique,
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

# Fraction of total time per section kind; nirupana shares whatever is left.
_FIXED = {"invocation": 0.05, "prastavana": 0.15, "mangalam": 0.05}

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
    n_nir = max(1, min(4, req.duration_minutes // 2))
    nir_sec = total * (1 - sum(_FIXED.values())) / n_nir

    def slots(types: list[str], sec: float) -> list[Slot]:
        each = sec / len(types)
        return [Slot(type=t, target_sec=each, target_chars=round(each * cps)) for t in types]

    def fixed(kind: str, types: list[str]) -> SectionBudget:
        sec = total * _FIXED[kind]
        return SectionBudget(section=kind, kind=kind, target_sec=sec, slots=slots(types, sec))

    chunks = [analysis.events[i::n_nir] for i in range(n_nir)]
    out = [fixed("invocation", ["shloka"]), fixed("prastavana", ["narration"])]
    out += [
        SectionBudget(section=f"nirupana_{i + 1}", kind="nirupana", target_sec=nir_sec,
                      events=chunks[i], slots=slots(["narration", "song", "narration"], nir_sec))
        for i in range(n_nir)
    ]
    out.append(fixed("mangalam", ["shloka"]))
    return out


# --- stage 3 -------------------------------------------------------------

def retrieve(req: GenerateRequest, analysis: StoryAnalysis, budgets: list[SectionBudget]) -> dict:
    """Offer verified corpus records per section. Only IDs are stored."""
    corpus = _load_corpus().values()
    kinds = {"invocation": "shloka", "mangalam": "shloka", "nirupana": "song"}
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

def write_section(
    pid: str, req: GenerateRequest, analysis: StoryAnalysis, b: SectionBudget, offered: list[str],
    corpus: dict[str, dict],
) -> SectionOutput:
    slot_lines = "\n".join(
        f"{i}. type={s.type}, about {s.target_chars} characters" for i, s in enumerate(b.slots, 1)
    )
    offered_lines = "\n".join(
        f"- id={i}: {corpus[i].get('title', '')} :: {corpus[i]['text'][:160]}" for i in offered
    ) or "(none)"
    lang = languages.name(req.language)
    prompt = (
        f"Write the spoken text in {lang}. The story and events below may be in another language: "
        f"retell them in {lang}, do not copy their wording.\n"
        f"Story: {req.title}\nDeity: {analysis.deity}; bhava: {analysis.bhava}; moral: {analysis.moral}\n"
        f"Characters: {', '.join(analysis.characters) or 'n/a'}\n"
        f"Section: {b.section} ({b.kind})\nEvents to cover in this section: "
        f"{'; '.join(b.events) or 'n/a'}\n\nWrite exactly {len(b.slots)} slot(s), in order:\n{slot_lines}\n\n"
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
            if not slot.source_id and len(slot.text) > want.target_chars * 1.25 + 20:
                raise llm.SoftCheck(f"slot {i}: {len(slot.text)} chars, limit is about {want.target_chars}")

    def mock() -> SectionOutput:
        return SectionOutput(slots=[
            SlotText(text=(f"[mock {s.type} for {b.section}] " + " ".join(b.events))[: max(s.target_chars, 20)])
            for s in b.slots
        ])

    return llm.complete_json(
        pid=pid, stage=f"write:{b.section}", prompt=prompt, model=SectionOutput, mock_fn=mock,
        check=check, reasoning_effort=settings.writing_reasoning_effort,
        system=_PERSONA + f" ALWAYS write in {lang}, in its own script, whatever language the story "
        f"is told in; keep names of people and places but transliterate them. Fill each slot in order. 'shloka' slots: a short devotional invocation or "
        "benediction. 'narration' slots: spoken storytelling that follows the events. 'song' slots: a "
        "short devotional verse. NEVER quote scripture or classical songs from memory and never claim "
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
        ))
    # Attach a stable scene key and emotional direction to each narration unit.
    for segment in segments:
        segment.scene = f"{b.section}-{segment.id.split('.')[-1]}"
        segment.delivery.emotion = _emotion_for(segment, b.kind)
        segment.delivery.emphasis_words = _emphasis_for(segment.text)
    return segments


def _emotion_for(segment: Segment, section_kind: str) -> str:
    if section_kind == "invocation":
        return "reverent, soft and prayerful"
    if section_kind == "mangalam":
        return "serene, grateful and peaceful"
    if segment.type == "song":
        return "devotional, melodic and warmly expressive"
    text = segment.text.lower()
    cues = {
        "wonder": ("miracle", "प्रकट", "दर्शन", "चमत्कार", "अद्भुत", "विस्मय"),
        "sorrow": ("वियोग", "विलाप", "आँसू", "दुःख", "करुणा", "रोने"),
        "courage": ("युद्ध", "रणभूमि", "वीर", "पराक्रम", "धर्मयुद्ध", "battle"),
        "joy": ("आनंद", "हर्ष", "उत्सव", "प्रसन्न", "विजय", "जय"),
        "wisdom": ("उपदेश", "ज्ञान", "धर्म", "सत्य", "तत्त्व", "नीति"),
    }
    for emotion, words in cues.items():
        if any(word in text for word in words):
            return {
                "wonder": "awe-filled, spacious and gently rising",
                "sorrow": "tender, compassionate and unhurried",
                "courage": "bold, energetic and resonant",
                "joy": "bright, joyful and uplifting",
                "wisdom": "calm, clear and contemplative",
            }[emotion]
    return "warm, conversational and devotional"


def _emphasis_for(text: str) -> list[str]:
    import re
    words = re.findall(r"[\\w\\u0900-\\u097f]+", text)
    stop = {"और", "तथा", "लेकिन", "क्योंकि", "फिर", "उसके", "उनके", "एक", "यह", "वह", "के", "की", "का"}
    candidates = [w for w in words if len(w) >= 3 and w.lower() not in stop]
    return list(dict.fromkeys(candidates[:5]))


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
    prompt = f"Write in {lang}.\nInstruction: {instruction}\nMaximum {max_chars} characters.\nText:\n{seg['text']}"

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
        cps = calibration.get(req.language)
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
        except llm.LLMError as e:  # a quality pass must not lose the script
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
