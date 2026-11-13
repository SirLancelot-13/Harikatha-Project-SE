from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Provider-neutral local inference defaults. The old Sarvam settings remain
    # readable for backwards compatibility with existing local .env files.
    mock_mode: bool | None = None
    mock_sarvam: bool = True
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen2.5:3b"
    parler_model: str = "ai4bharat/indic-parler-tts"
    tts_speaker: str = "A warm, expressive Indian storyteller with a clear, resonant voice, natural Hindi pronunciation, gentle devotional warmth and measured pacing."
    tts_device: str = "auto"
    tts_sample_rate: int = 24000
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    sounds_dir: Path = Path(__file__).resolve().parent.parent / "sounds"

    reasoning_effort: str = "low"
    critique_reasoning_effort: str = "none"
    writing_reasoning_effort: str = "none"
    llm_temperature: float = 0.5
    llm_max_tokens: int = 6000
    overrun_threshold: float = 0.2
    mix_disable: str = ""

    request_timeout_sec: float = 240.0
    max_retries: int = 3
    retry_delay_sec: float = 1.0

    @property
    def use_mock(self) -> bool:
        # MOCK_MODE takes precedence; MOCK_SARVAM is retained as a legacy alias.
        return self.mock_mode if self.mock_mode is not None else self.mock_sarvam

    @property
    def tts_model(self) -> str:
        return self.parler_model

    @property
    def performances_dir(self) -> Path:
        return self.data_dir / "runtime" / "performances"

    def snapshot(self) -> dict:
        """Settings recorded in meta.json. Never includes secrets."""
        return {
            "llm_provider": "ollama",
            "llm_model": self.llm_model,
            "tts_provider": "indic-parler-tts",
            "tts_model": self.parler_model,
            "tts_speaker": self.tts_speaker,
            "tts_sample_rate": self.tts_sample_rate,
            "overrun_threshold": self.overrun_threshold,
            "mock": self.use_mock,
        }


settings = Settings()
