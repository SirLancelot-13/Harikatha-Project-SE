"""TTS with a global content-addressed cache (one entry per sentence chunk).

Backends: "sarvam" (Bulbul), "local" (self-hosted open-source model, see local_tts/), and
"mock" (a quiet tone, no network). One performance always uses ONE backend so the voice
never changes mid-story; see pick_backend() and render.py for how fallback is decided.
"""
import base64
import hashlib
import re
import wave
from array import array

from . import audio, local, sarvam
from .config import settings

MOCK_CHARS_PER_SEC = 12.0
MAX_CHUNK_CHARS = 400  # API limit is 2500; short chunks keep prosody natural and cache well
_MOCK_HZ = 200  # divides every supported rate into whole periods
_SENTENCE_END = re.compile(r"(?<=[.!?।॥])\s+")


def split_text(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Greedy-pack whole sentences into chunks of at most `limit` chars."""
    chunks: list[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        while len(sentence) > limit:  # a single run-on sentence: cut at the last space
            cut = sentence.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if chunks and sentence and len(chunks[-1]) + 1 + len(sentence) <= limit:
            chunks[-1] += " " + sentence
        elif sentence:
            chunks.append(sentence)
    return chunks or [text]


def split_song(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """A song keeps its line breaks, so the local voice can read its line length and rhyme.
    Chunks are whole groups of lines (never part of a line), joined with newlines."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return split_text(text)
    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        if current and len("\n".join([*current, line])) > limit:
            chunks.append("\n".join(current))
            current = []
        current.append(line)
    chunks.append("\n".join(current))
    return chunks


def flatten_lines(text: str) -> str:
    """For voices that read plain text (Bulbul): one line of text, each line end kept as a small pause."""
    if "\n" not in text:
        return text
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return " ".join(ln if ln[-1] in ".,;:!?।॥" else ln + "," for ln in lines)


# --- choosing a backend --------------------------------------------------------

def voice_id(backend: str) -> str:
    """Names the voice for cache keys and calibration, so different voices never share entries."""
    if backend == "local":
        return f"local/{(local.health() or {}).get('engine', 'unknown')}"
    return f"{settings.tts_model}/{settings.tts_speaker}"


def pick_backend(language: str) -> tuple[str, str | None]:
    """(backend, why_not_local). Decided once per render so the whole story has one voice."""
    if settings.mock_sarvam:
        return "mock", None
    if settings.tts_provider == "sarvam":
        return "sarvam", None
    info = local.health()
    if info is not None and language in info["languages"]:
        return "local", None
    reason = "local TTS server is not reachable" if info is None else f"local TTS does not support {language}"
    if settings.tts_provider == "local":
        raise local.LocalError(reason)
    return "sarvam", reason


def planned_voice(language: str) -> str:
    """Voice used to size the script before rendering. Never raises."""
    try:
        backend, _ = pick_backend(language)
    except local.LocalError:
        backend = "sarvam"
    return voice_id("local" if backend == "local" else "sarvam")


# --- synthesis -------------------------------------------------------------------

def cache_key(text: str, language: str, pace: float, backend: str = "sarvam", kind: str = "") -> str:
    parts = [text, language, f"{pace:.3f}", backend, str(settings.tts_sample_rate)]
    if backend == "local":
        parts += [voice_id("local"), kind]
    elif backend == "sarvam":
        parts += [settings.tts_model, settings.tts_speaker]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def _mock_tone(text: str, pace: float) -> array:
    rate = settings.tts_sample_rate
    n = max(rate // 3, round(len(text) / (MOCK_CHARS_PER_SEC * pace) * rate))
    period = rate // _MOCK_HZ
    cycle = array("h", (round(6000 * ((i / period) % 1 * 2 - 1)) for i in range(period)))
    samples = (cycle * (n // period + 1))[:n]
    fade = min(n // 2, rate // 50)
    for i in range(fade):  # 20 ms fades avoid clicks at segment joins
        samples[i] = samples[i] * i // fade
        samples[n - 1 - i] = samples[n - 1 - i] * i // fade
    return samples


def _bulbul(text: str, language: str, pace: float) -> bytes:
    data = sarvam.post("/text-to-speech", {
        "text": text,
        "language_code": language,
        "speaker": settings.tts_speaker,
        "model": settings.tts_model,
        "pace": min(2.0, max(0.5, pace)),
        "speech_sample_rate": settings.tts_sample_rate,
        "output_audio_codec": "wav",
        "temperature": 0.6,
    })
    # Per the docs, audios may be split across elements: join, then decode.
    return base64.b64decode("".join(data["audios"]))


def _store_wav(path, wav_bytes: bytes, error: type[Exception]) -> array:
    """Validate downloaded audio, then publish it atomically into the cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{id(wav_bytes)}.tmp")  # atomic: concurrent renders share the cache
    tmp.write_bytes(wav_bytes)
    try:
        samples, rate = audio.read_wav(tmp)
    except (wave.Error, EOFError) as e:
        tmp.unlink(missing_ok=True)
        raise error(f"unexpected WAV: {e}") from e
    if rate != settings.tts_sample_rate:
        tmp.unlink(missing_ok=True)
        raise error(f"returned {rate} Hz, expected {settings.tts_sample_rate}")
    if len(samples) == 0:
        tmp.unlink(missing_ok=True)
        raise error("returned empty audio")
    tmp.replace(path)
    return samples


def _synth_chunk(text: str, language: str, pace: float, kind: str, backend: str) -> tuple[array, bool]:
    key = cache_key(text, language, pace, backend, kind)
    path = settings.data_dir / "runtime" / "tts_cache" / f"{key}.wav"
    if path.exists():
        return audio.read_wav(path)[0], True
    if backend == "mock":
        samples = _mock_tone(text, pace)
        audio.write_wav(path, samples, settings.tts_sample_rate)
        return samples, False
    if backend == "local":
        return _store_wav(path, local.synthesize(text, language, pace, kind), local.LocalError), False
    return _store_wav(path, _bulbul(text, language, pace), sarvam.SarvamError), False


def synthesize(
    text: str, language: str, pace: float = 1.0, kind: str = "narration", backend: str | None = None
) -> tuple[array, bool]:
    """Returns (samples, all_chunks_cached). Mock mode always wins: it never uses the network."""
    backend = "mock" if settings.mock_sarvam else (backend or "sarvam")
    if backend == "mock":
        chunks = [text]
    elif backend == "local" and kind == "song":
        chunks = split_song(text)  # the local voice is told how the song is built, so keep its lines
    else:
        chunks = split_text(flatten_lines(text))
    out, all_hit = array("h"), True
    for chunk in chunks:
        samples, hit = _synth_chunk(chunk, language, pace, kind, backend)
        out.extend(samples)
        all_hit &= hit
    return out, all_hit
