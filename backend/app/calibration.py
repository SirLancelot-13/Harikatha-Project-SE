"""Measured TTS speed (chars/sec), smoothed over runs. Used to turn seconds into characters."""
import json

from .config import settings

FALLBACK_CPS = 13.2
_SMOOTHING = 0.3  # weight of the newest measurement


def _path():
    return settings.data_dir / "runtime" / "calibration.json"


def _key(language: str, voice: str | None) -> str:
    voice = voice or f"{settings.tts_model}/{settings.tts_speaker}"  # default: the Sarvam voice
    return f"{language}|{voice}|{'mock' if settings.mock_sarvam else 'live'}"


def _load() -> dict:
    return json.loads(_path().read_text()) if _path().exists() else {}


def get(language: str, voice: str | None = None) -> float:
    return _load().get(_key(language, voice), FALLBACK_CPS)


def update(language: str, chars: int, seconds: float, voice: str | None = None) -> None:
    if chars < 50 or seconds <= 0:
        return
    data = _load()
    measured = chars / seconds
    old = data.get(_key(language, voice))
    data[_key(language, voice)] = round(
        measured if old is None else old * (1 - _SMOOTHING) + measured * _SMOOTHING, 3
    )
    _path().parent.mkdir(parents=True, exist_ok=True)
    _path().write_text(json.dumps(data, indent=2))
