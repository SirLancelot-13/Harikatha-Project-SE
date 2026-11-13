"""Local Indic Parler-TTS synthesis with a content-addressed WAV cache.

Model weights are downloaded from Hugging Face on first live use. No hosted TTS API
or API key is used. Sentence chunks retain the script's semantic boundaries.
"""
import hashlib
import re
import threading
import wave
from array import array

import numpy as np

from . import audio
from .config import settings

MOCK_CHARS_PER_SEC = 12.0
MAX_CHUNK_CHARS = 350
_MOCK_HZ = 200
_SENTENCE_END = re.compile(r"(?<=[.!?।॥])\s+")
_model_lock = threading.Lock()
_model = None
_tokenizer = None


def split_text(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    chunks: list[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        while len(sentence) > limit:
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


def cache_key(text: str, language: str, pace: float, emotion: str = "devotional", emphasis: tuple[str, ...] = ()) -> str:
    parts = [text, language, f"{pace:.3f}", settings.parler_model,
             settings.tts_speaker, emotion, ",".join(emphasis), str(settings.tts_sample_rate), "mock" if settings.use_mock else "live"]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def _mock_tone(text: str, pace: float) -> array:
    rate = settings.tts_sample_rate
    n = max(rate // 3, round(len(text) / (MOCK_CHARS_PER_SEC * pace) * rate))
    period = rate // _MOCK_HZ
    cycle = array("h", (round(6000 * ((i / period) % 1 * 2 - 1)) for i in range(period)))
    samples = (cycle * (n // period + 1))[:n]
    fade = min(n // 2, rate // 50)
    for i in range(fade):
        samples[i] = samples[i] * i // fade
        samples[n - 1 - i] = samples[n - 1 - i] * i // fade
    return samples


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    with _model_lock:
        if _model is not None:
            return _model, _tokenizer
        try:
            import torch
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Indic Parler-TTS dependencies are missing. Install the optional tts dependencies "
                "from pyproject.toml, then restart the backend."
            ) from exc
        device = settings.tts_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        _model = ParlerTTSForConditionalGeneration.from_pretrained(
            settings.parler_model, torch_dtype=dtype
        ).to(device)
        _tokenizer = AutoTokenizer.from_pretrained(settings.parler_model)
        _model.eval()
        return _model, _tokenizer


def _speaker_description(language: str, pace: float, emotion: str = "devotional", emphasis: tuple[str, ...] = ()) -> str:
    pace_words = "slow and deliberate" if pace < 0.88 else "brisk but clear" if pace > 1.12 else "measured"
    language_name = {
        "hi-IN": "Hindi", "en-IN": "Indian English", "te-IN": "Telugu",
        "ta-IN": "Tamil", "kn-IN": "Kannada", "bn-IN": "Bengali",
        "mr-IN": "Marathi", "ml-IN": "Malayalam", "gu-IN": "Gujarati",
        "od-IN": "Odia", "pa-IN": "Punjabi",
    }.get(language, "Hindi")
    return (
        f"{settings.tts_speaker} The speaker narrates in {language_name}, with {pace_words} pacing, "
        "clear diction, " + emotion + " delivery, expressive rises and falls, natural breaths. "\n        + "Emphasize these words when present: " + (", ".join(emphasis) or "key names and devotional phrases") + "."
    )


def _parler(text: str, language: str, pace: float, emotion: str, emphasis: tuple[str, ...]) -> array:
    import torch
    model, tokenizer = _load_model()
    device = next(model.parameters()).device
    description = _speaker_description(language, pace, emotion, emphasis)
    description_inputs = tokenizer(description, return_tensors="pt").to(device)
    prompt_inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        generation = model.generate(
            input_ids=description_inputs.input_ids,
            attention_mask=description_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
        )
    samples = generation.detach().cpu().numpy().squeeze().astype(np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 0:
        samples = samples / max(1.0, peak) * 0.92
    pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    return array("h", pcm.tolist())


def _synth_chunk(text: str, language: str, pace: float, emotion: str, emphasis: tuple[str, ...]) -> tuple[array, bool]:
    path = settings.data_dir / "runtime" / "tts_cache" / f"{cache_key(text, language, pace, emotion, emphasis)}.wav"
    if path.exists():
        return audio.read_wav(path)[0], True
    samples = _mock_tone(text, pace) if settings.use_mock else _parler(text, language, pace, emotion, emphasis)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio.write_wav(path, samples, settings.tts_sample_rate)
    return samples, False


def synthesize(text: str, language: str, pace: float = 1.0, emotion: str = "devotional", emphasis: list[str] | None = None) -> tuple[array, bool]:
    """Returns (PCM samples, all_chunks_cached)."""
    chunks = [text] if settings.use_mock else split_text(text)
    emphasis_tuple = tuple((emphasis or [])[:5])
    out, all_hit = array("h"), True
    for chunk in chunks:
        samples, hit = _synth_chunk(chunk, language, pace, emotion, emphasis_tuple)
        out.extend(samples)
        all_hit &= hit
    return out, all_hit
