import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def _force_mock(monkeypatch):
    """Tests must never depend on .env or hit the real Sarvam API."""
    monkeypatch.setattr(settings, "mock_sarvam", True)
    monkeypatch.setattr(settings, "sarvam_api_key", "")


@pytest.fixture(autouse=True)
def _no_recorded_sounds(tmp_path_factory, monkeypatch):
    """Default to an empty sounds dir so tests use the synthesised beds, like CI does."""
    monkeypatch.setattr(settings, "sounds_dir", tmp_path_factory.mktemp("sounds"))


@pytest.fixture(autouse=True)
def _isolate_providers(monkeypatch):
    """Tests never depend on TTS_PROVIDER / LLM_PROVIDER / LOCAL_* from the developer's .env."""
    from app import local

    monkeypatch.setattr(settings, "tts_provider", "sarvam")
    monkeypatch.setattr(settings, "llm_provider", "sarvam")
    monkeypatch.setattr(settings, "local_llm_url", "")
    monkeypatch.setattr(settings, "local_llm_model", "")
    monkeypatch.setattr(local, "transport", None)
    local.reset_health_cache()
    yield
    local.reset_health_cache()
