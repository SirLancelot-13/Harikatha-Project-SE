from typing import Literal

from pydantic import BaseModel, Field

LanguageCode = Literal[
    "bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN"
]


class GenerateRequest(BaseModel):
    title: str
    story: str
    language: LanguageCode = "en-IN"
    duration_minutes: int = Field(default=8, ge=1, le=60)
    audience: str = "general"


class StoryAnalysis(BaseModel):
    characters: list[str]
    events: list[str]
    deity: str
    bhava: str
    moral: str


class Slot(BaseModel):
    type: Literal["shloka", "narration", "song"]
    target_sec: float
    target_chars: int
    pace: float = 1.0


class SectionBudget(BaseModel):
    section: str
    kind: Literal["invocation", "prastavana", "nirupana", "mangalam"]
    target_sec: float
    events: list[str] = []
    slots: list[Slot]


class Delivery(BaseModel):
    pace: float = 1.0
    gain_db: float = 0.0
    emotion: str = "devotional"
    emphasis_words: list[str] = []
    melodic: bool = False
    raga: str | None = None
    tala: str | None = None
    tempo_bpm: int | None = None
    emphasis: list[tuple[int, int]] = []


class Segment(BaseModel):
    id: str
    section: str
    type: Literal["shloka", "narration", "song"]
    text: str
    origin: Literal["original", "traditional"] = "original"
    source_ids: list[str] = []
    target_sec: float
    delivery: Delivery = Delivery()
    scene: str | None = None
    sfx: str | None = None


class Script(BaseModel):
    title: str
    language: str
    target_sec: float
    segments: list[Segment]
    content_notes: str = ""


class SlotText(BaseModel):
    text: str = ""
    source_id: str | None = None


class SectionOutput(BaseModel):
    slots: list[SlotText]


class RewriteOutput(BaseModel):
    text: str


class CritiqueIssue(BaseModel):
    segment_id: str
    problem: str
    fix_instruction: str


class Critique(BaseModel):
    ok: bool
    issues: list[CritiqueIssue] = []


class Status(BaseModel):
    stage: Literal[
        "analysing", "planning", "writing", "scripted", "queued",
        "synthesizing", "rewriting", "modulating", "mixing", "done", "error",
    ]
    detail: str = ""
    progress: float | None = None
    script_version: int | None = None
    updated_at: str


class SegmentEdit(BaseModel):
    text: str | None = None
    delivery: dict | None = None


class Rating(BaseModel):
    rater_id: str
    clarity: int = Field(ge=1, le=5)
    musicality: int = Field(ge=1, le=5)
    harikatha_feel: int = Field(ge=1, le=5)
    fidelity: int = Field(ge=1, le=5)
    comment: str = ""
