from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mock_sarvam: bool = True
    sarvam_api_key: str = ""
    sarvam_base_url: str = "https://api.sarvam.ai"
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    sounds_dir: Path = Path(__file__).resolve().parent.parent / "sounds"

    llm_model: str = "sarvam-105b"
    reasoning_effort: str = "low"  # analyse + critique
    critique_reasoning_effort: str = "none"  # "low" ran out of tokens on long Devanagari scripts
    writing_reasoning_effort: str = "none"  # write + rewrite: "low" burned ~4.7k tokens per slot
    llm_temperature: float = 0.5
    llm_max_tokens: int = 6000  # reasoning tokens share this budget
    tts_model: str = "bulbul:v3"
    tts_speaker: str = "shubh"
    tts_sample_rate: int = 24000
    overrun_threshold: float = 0.2
    pitch_modulation: bool = True  # PITCH_MODULATION=false turns the shloka lift off
    shloka_lift_st: float = 1.5  # default gentle lift for shlokas (hard cap is 3.0)
    mix_disable: str = ""  # comma list of layers to leave out: drone,scene,pulse,bells

    # Open-source models run locally; Sarvam stays the default and the fallback.
    #   sarvam = Sarvam only | local = local only (fail if it is down) | auto = local first, else Sarvam
    tts_provider: Literal["sarvam", "local", "auto"] = "sarvam"
    llm_provider: Literal["sarvam", "local", "auto"] = "sarvam"
    local_tts_url: str = "http://127.0.0.1:8100"
    local_tts_timeout_sec: float = 180.0
    local_llm_url: str = ""  # OpenAI-compatible base URL incl. /v1, e.g. http://127.0.0.1:11434/v1
    local_llm_model: str = ""
    local_llm_api_key: str = ""  # only if your local server wants one; never the Sarvam key
    local_llm_timeout_sec: float = 300.0
    local_connect_timeout_sec: float = 3.0  # a dead local server must fail fast, then we fall back

    request_timeout_sec: float = 120.0
    max_retries: int = 3
    retry_delay_sec: float = 1.0

    @property
    def performances_dir(self) -> Path:
        return self.data_dir / "runtime" / "performances"

    def snapshot(self) -> dict:
        """Settings recorded in meta.json. Never includes the API key."""
        return {
            "llm_model": self.llm_model,
            "reasoning_effort": self.reasoning_effort,
            "writing_reasoning_effort": self.writing_reasoning_effort,
            "critique_reasoning_effort": self.critique_reasoning_effort,
            "tts_model": self.tts_model,
            "tts_speaker": self.tts_speaker,
            "tts_sample_rate": self.tts_sample_rate,
            "overrun_threshold": self.overrun_threshold,
            "mock": self.mock_sarvam,
            "tts_provider": self.tts_provider,
            "llm_provider": self.llm_provider,
        }


settings = Settings()
