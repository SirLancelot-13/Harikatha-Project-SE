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

_STYLE = {
    "narration": "slightly expressive and animated",
    "shloka": "calm, devotional and slightly expressive",
    "song": "expressive and warm",
}


def describe(speaker: str, kind: str, pace: float) -> str:
    """Build the natural-language voice description Parler-TTS conditions on (modelled on the
    examples on its model card). Pitch is left balanced: the backend applies its own gentle lift."""
    if pace < 0.85:
        speed = "slowly"
    elif pace < 1.0:
        speed = "a little slowly"
    elif pace <= 1.1:
        speed = "at a moderate speed"
    else:
        speed = "fast"
    style = _STYLE.get(kind, _STYLE["narration"])
    return (
        f"{speaker} speaks {speed} with a {style} delivery and a balanced pitch. "
        "The recording is of very high quality, with the speaker's voice sounding clear and very close up."
    )


class Engine(Protocol):
    name: str
    languages: tuple[str, ...]

    def load(self) -> None: ...
    def synthesize(self, text: str, language: str, pace: float, kind: str, speaker: str | None) -> np.ndarray:
        """Return float32 mono audio in [-1, 1] at SAMPLE_RATE."""


class StubEngine:
    """A quiet tone whose length follows the text: exercises the wiring without a model."""

    name = "stub"
    languages = ("hi-IN", "en-IN", "ta-IN", "te-IN")

    def load(self) -> None:
        pass

    def synthesize(self, text, language, pace, kind, speaker):
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

    def synthesize(self, text, language, pace, kind, speaker):
        torch = self.torch
        speaker = speaker or self.override_speaker or DEFAULT_SPEAKERS[language]
        desc = self.desc_tokenizer(describe(speaker, kind, pace), return_tensors="pt").to(self.device)
        prompt = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=desc.input_ids, attention_mask=desc.attention_mask,
                prompt_input_ids=prompt.input_ids, prompt_attention_mask=prompt.attention_mask,
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
                audio = engine.synthesize(req.text, req.language_code, req.pace, req.kind, req.speaker)
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
