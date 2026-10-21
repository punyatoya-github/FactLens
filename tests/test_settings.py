from __future__ import annotations

import pytest

from factlens.core.errors import ConfigurationError
from factlens.core.settings import Settings


def _fresh(**kwargs: object) -> Settings:
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


def test_defaults_are_sensible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = _fresh()
    assert settings.chat_model == "gpt-6-luna"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.chat_temperature is None
    assert settings.share is False
    assert settings.hybrid_weights == [0.4, 0.6]
    assert settings.cache_dir == settings.data_dir / "cache"
    assert settings.vector_store_dir == settings.data_dir / "vectors"


def test_missing_api_key_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FACTLENS_OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        _fresh().require_openai_api_key()


def test_api_key_read_from_standard_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-standard")
    assert _fresh().require_openai_api_key() == "sk-standard"


def test_api_key_accepts_prefixed_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("FACTLENS_OPENAI_API_KEY", "sk-prefixed")
    assert _fresh().require_openai_api_key() == "sk-prefixed"


def test_blank_api_key_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        _fresh(openai_api_key="   ").require_openai_api_key()


def test_factlens_prefix_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACTLENS_VECTOR_SEARCH_K", "25")
    monkeypatch.setenv("FACTLENS_SHARE", "true")
    settings = _fresh()
    assert settings.vector_search_k == 25
    assert settings.share is True


def test_per_agent_model_overrides_fall_back_to_chat_model() -> None:
    settings = _fresh(chat_model="base", synthesis_model="big")
    assert settings.resolve_model("scope") == "base"
    assert settings.resolve_model("synthesis") == "big"
    assert settings.resolve_model("verification") == "base"


@pytest.mark.parametrize("weights", [[1.0], [0.5, 0.5, 0.5], [-0.1, 1.0], [0.0, 0.0]])
def test_invalid_hybrid_weights_are_rejected(weights: list[float]) -> None:
    with pytest.raises(ValueError, match="hybrid_weights"):
        _fresh(hybrid_weights=weights)
