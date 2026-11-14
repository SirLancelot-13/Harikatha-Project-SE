"""Local TTS server for the Harikatha backend.

A tiny HTTP service with an engine-agnostic contract, so the model behind it can be swapped:

    GET  /health      -> {"ok": true, "engine": str, "sample_rate": 24000, "languages": [...]}
    POST /synthesize  {text, language_code, pace, kind[, speaker]} -> audio/wav (PCM16 mono, 24 kHz)

Engines: "parler" (ai4bharat/indic-parler-tts, default) and "stub" (a tone, for testing the wiring
without a GPU or model download). Pick with LOCAL_TTS_ENGINE. Binds to 127.0.0.1 only.
"""
import io
import logging
import os
import re
import statistics
import threading
import wave
from contextlib import asynccontextmanager
from typing import Literal, Protocol

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

SAMPLE_RATE = 24000  # the backend expects exactly this; engines resample to it
MAX_TEXT_CHARS = 1000
log = logging.getLogger("local_tts")

# Speakers recommended on the Indic Parler-TTS model card. The card gives no genders, so listen
# and override with LOCAL_TTS_SPEAKER (applies to every language) if you want another voice.
DEFAULT_SPEAKERS = {
    "hi-IN": "Rohit", "ta-IN": "Jaya", "te-IN": "Prakash", "en-IN": "Thoma", "mr-IN": "Sanjay",
    "bn-IN": "Arjun", "gu-IN": "Yash", "kn-IN": "Suresh", "ml-IN": "Anjali", "od-IN": "Manas",
}
# Language codes the card lists as officially supported (Punjabi is only "unofficial", so it is left
# out: the backend then falls back to Sarvam for it).
PARLER_LANGUAGES = tuple(DEFAULT_SPEAKERS)

# What the delivery should sound like, per segment type. Wording follows the Indic Parler-TTS model
# card's examples ("expressive", "emotive", "stressing...", "energetic"). Measured on Hindi: the words
# move pitch CENTRE and a little speed, but expressivity wording barely changes pitch spread (already
# wider than Bulbul's) and emotion words are not validated for Hindi on the card. See
# tune_description.py for how this was measured.
_STYLE = {
    "narration": (
        "an expressive and emotive voice, stressing the important words for emphasis, with a warm, engaging, "
        "devotional feeling and natural rises and falls in pitch"
    ),
    "shloka": (
        "a reverent, resonant and devotional voice, giving gentle emphasis to each phrase, with deep feeling"
    ),
    "song": (
        "an expressive, joyful and heartfelt voice with flowing, melodic phrasing and emphasis on the key words"
    ),
}


# --- songs: tell the voice it is a song, and how it is built --------------------------------------
_PUNCT = ".,;:!?।॥\"'()-–—…"
_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def count_syllables(line: str) -> int:
    """Syllables (aksharas) in a line. Indic scripts share one layout (independent vowels at offsets
    0x05-0x14 and consonants at 0x15-0x39 of their Unicode block, virama at 0x4D), so one rule covers
    them all; anything else falls back to counting vowel groups."""
    chars = list(line)
    n = 0
    for i, c in enumerate(chars):
        cp = ord(c)
        if 0x0900 <= cp <= 0x0DFF:
            off = cp & 0x7F
            if 0x05 <= off <= 0x14:
                n += 1
            elif 0x15 <= off <= 0x39:
                nxt = ord(chars[i + 1]) if i + 1 < len(chars) else 0
                if not (0x0900 <= nxt <= 0x0DFF and (nxt & 0x7F) == 0x4D):  # a virama joins it to the next
                    n += 1
    return n or len(re.findall(r"[aeiouy]+", line.lower()))


def split_song_lines(text: str) -> list[str]:
    """The song's lines: its own line breaks, or (one paragraph) its sentence-like marks."""
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", text) if ln.strip()]
    if len(lines) == 1:
        lines = [p.strip() for p in re.split(r"(?<=[.!?।॥;])\s+", lines[0]) if p.strip()]
    return lines


def _ending(line: str) -> str:
    words = [w for w in re.split(r"[\s" + re.escape(_PUNCT) + "]+", line) if w]
    return words[-1].lower()[-2:] if words else ""  # the last two characters of the last word


def rhyme_scheme(lines: list[str]) -> str:
    """Letters for the line endings: matching endings share a letter, e.g. AABB or ABAB."""
    seen: dict[str, str] = {}
    out = []
    for ln in lines:
        key = _ending(ln)
        seen.setdefault(key, chr(ord("A") + min(len(seen), 25)))
        out.append(seen[key])
    return "".join(out)


def analyse_song(text: str) -> dict:
    lines = split_song_lines(text)
    counts = [count_syllables(ln) for ln in lines]
    scheme = rhyme_scheme(lines)
    typical = int(statistics.median(counts)) if counts else 0
    return {
        "lines": lines, "syllables": counts, "typical": typical, "spread": (max(counts) - min(counts)) if counts else 0,
        "scheme": scheme, "rhymed": len(set(scheme)) < len(scheme),
        "refrain": len({ln.strip(_PUNCT + " ").lower() for ln in lines}) < len(lines),
    }


def song_clause(text: str) -> str:
    """Sentences for the voice description: line count, line length, rhyme, refrain, and how to deliver it."""
    a = analyse_song(text)
    n = len(a["lines"])
    if n < 2:
        return "This is a short devotional song, delivered in a flowing, rhythmic, sing-song way."
    parts = [f"This is a devotional song of {_NUMBER_WORDS.get(n, n)} short lines"]
    if a["spread"] <= max(2, round(a["typical"] * 0.25)):
        parts.append(f"each of about {a['typical']} syllables, so the lines feel even")
    else:
        parts.append(f"with lines of {min(a['syllables'])} to {max(a['syllables'])} syllables")
    if a["rhymed"]:
        parts.append(f"rhyming in an {a['scheme']} pattern" if n <= 8 else "with rhyming line endings")
    if a["refrain"]:
        parts.append("with a repeated refrain")
    return (", ".join(parts) + ". The delivery is sing-song and rhythmic, with a steady beat, "
            "a gentle lift at the end of each line and a brief pause between the lines.")


def song_text(text: str) -> str:
    """The song as the voice should read it: every line ends in punctuation (the card says punctuation
    sets the prosody), so each line end becomes a small break and the last one a full stop."""
    lines = split_song_lines(text)
    if not lines:
        return text
    lines = [ln if ln[-1] in ".,;:!?।॥" else ln + "," for ln in lines]
    if lines[-1][-1] == ",":
        lines[-1] = lines[-1][:-1] + ("।" if any(0x0900 <= ord(c) <= 0x0DFF for c in lines[-1]) else ".")
    return " ".join(lines)


def describe(speaker: str, kind: str, pace: float, text: str | None = None) -> str:
    """Build the natural-language voice description Parler-TTS conditions on (modelled on the
    examples on its model card). Pitch is left balanced: the backend applies its own gentle lift.
    For a song, `text` is analysed so the description can say how many lines it has, how long they are
    and how they rhyme."""
    if pace < 0.85:
        speed = "slowly"
    elif pace < 1.0:
        speed = "at a calm, measured pace"
    elif pace <= 1.1:
        speed = "at a fairly fast, lively pace" if kind != "shloka" else "at a calm, measured pace"
    else:
        speed = "fast"
    style = _STYLE.get(kind, _STYLE["narration"])
    song = f" {song_clause(text)}" if kind == "song" and text else ""
    return (
        f"{speaker} speaks {speed} in {style}, with a balanced pitch.{song} "
        "The recording is of very high quality, with the speaker's voice sounding clear and very close up."
    )


class Engine(Protocol):
    name: str
    languages: tuple[str, ...]

    def load(self) -> None: ...
    def synthesize(
        self, text: str, language: str, pace: float, kind: str, speaker: str | None,
        description: str | None = None, seed: int | None = None, temperature: float | None = None,
    ) -> np.ndarray:
        """Return float32 mono audio in [-1, 1] at SAMPLE_RATE. `description` and `seed` are tuning hooks."""


class StubEngine:
    """A quiet tone whose length follows the text: exercises the wiring without a model."""

    name = "stub"
    languages = ("hi-IN", "en-IN", "ta-IN", "te-IN")

    def load(self) -> None:
        pass

    def synthesize(self, text, language, pace, kind, speaker, description=None, seed=None, temperature=None):
        n = max(SAMPLE_RATE // 3, round(len(text) / (14.0 * pace) * SAMPLE_RATE))
        t = np.arange(n) / SAMPLE_RATE
        fade = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.02)
        return (0.1 * np.sin(2 * np.pi * 180 * t) * fade).astype(np.float32)


class ParlerEngine:
    name = "indic-parler-tts"
    languages = PARLER_LANGUAGES

    def __init__(self) -> None:
        self.repo = os.environ.get("LOCAL_TTS_MODEL", "ai4bharat/indic-parler-tts")
        self.override_speaker = os.environ.get("LOCAL_TTS_SPEAKER") or None

    def load(self) -> None:
        import torch  # heavy imports live here so the stub and the tests need no torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        self.torch = torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        log.info("loading %s on %s (%s)", self.repo, self.device, dtype)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(self.repo, torch_dtype=dtype).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo)
        self.desc_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.native_rate = int(self.model.config.sampling_rate)
        log.info("ready: native sample rate %s Hz", self.native_rate)

    def synthesize(self, text, language, pace, kind, speaker, description=None, seed=None, temperature=None):
        torch = self.torch
        speaker = speaker or self.override_speaker or DEFAULT_SPEAKERS[language]
        if seed is not None:
            torch.manual_seed(seed)  # reproducible sampling, for A/B comparisons
        description = (description or describe(speaker, kind, pace, text)).replace("{speaker}", speaker)
        if kind == "song":
            text = song_text(text)  # after the description has read the line structure
        desc = self.desc_tokenizer(description, return_tensors="pt").to(self.device)
        prompt = self.tokenizer(text, return_tensors="pt").to(self.device)
        extra = {"temperature": temperature, "do_sample": True} if temperature is not None else {}
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=desc.input_ids, attention_mask=desc.attention_mask,
                prompt_input_ids=prompt.input_ids, prompt_attention_mask=prompt.attention_mask, **extra,
            )
        audio = out.float().cpu().numpy().squeeze()
        if self.native_rate != SAMPLE_RATE:
            import torchaudio

            audio = torchaudio.functional.resample(
                torch.from_numpy(audio).float(), self.native_rate, SAMPLE_RATE
            ).numpy()
        return audio.astype(np.float32)


class SynthRequest(BaseModel):
    text: str = Field(min_length=1)
    language_code: str
    pace: float = Field(default=1.0, ge=0.5, le=2.0)
    kind: Literal["narration", "shloka", "song"] = "narration"
    speaker: str | None = None
    description: str | None = None  # tuning hook: replaces the built-in description ({speaker} is filled in)
    seed: int | None = None  # tuning hook: fixes the sampling seed
    temperature: float | None = Field(default=None, gt=0, le=2)  # tuning hook: sampling temperature


PEAK_LIMIT = 0.95  # engines can come out at or beyond full scale; clipping would be audible


def to_wav(audio: np.ndarray) -> bytes:
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > PEAK_LIMIT:
        audio = audio * (PEAK_LIMIT / peak)  # scale down, never clip
    pcm = np.round(np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def create_app(engine: Engine, load: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if load:
            engine.load()  # at startup, so the first request does not pay for model loading
        yield

    app = FastAPI(title="Harikatha local TTS", lifespan=lifespan)
    lock = threading.Lock()  # one generation at a time: the GPU is the shared resource

    @app.get("/health")
    def health():
        return {"ok": True, "engine": engine.name, "sample_rate": SAMPLE_RATE, "languages": list(engine.languages)}

    @app.post("/synthesize")
    def synthesize(req: SynthRequest):
        if req.language_code not in engine.languages:
            raise HTTPException(422, f"{engine.name} does not support {req.language_code}")
        if len(req.text) > MAX_TEXT_CHARS:
            raise HTTPException(422, f"text is {len(req.text)} chars, the limit is {MAX_TEXT_CHARS}")
        try:
            with lock:
                audio = engine.synthesize(
                    req.text, req.language_code, req.pace, req.kind, req.speaker, req.description, req.seed,
                    req.temperature,
                )
        except Exception as e:
            log.exception("synthesis failed")
            raise HTTPException(500, f"{type(e).__name__}: {e}") from e
        if audio.size == 0 or not np.isfinite(audio).all():
            raise HTTPException(500, "engine produced empty or invalid audio")
        return Response(content=to_wav(audio), media_type="audio/wav")

    return app


def make_engine() -> Engine:
    name = os.environ.get("LOCAL_TTS_ENGINE", "parler")
    if name == "stub":
        return StubEngine()
    if name == "parler":
        return ParlerEngine()
    raise SystemExit(f"unknown LOCAL_TTS_ENGINE={name!r} (use 'parler' or 'stub')")


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("LOCAL_TTS_PORT", "8100"))
    uvicorn.run(create_app(make_engine()), host="127.0.0.1", port=port)  # loopback only


if __name__ == "__main__":
    main()
