"""Application settings, loaded from environment variables and an optional ``.env`` file.

All FactLens options use the ``FACTLENS_`` prefix. The OpenAI key is read from the standard
``OPENAI_API_KEY`` variable (``FACTLENS_OPENAI_API_KEY`` is also accepted).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from factlens.core import constants
from factlens.core.errors import ConfigurationError


class Settings(BaseSettings):
    """Typed configuration for FactLens."""

    model_config = SettingsConfigDict(
        env_prefix="FACTLENS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- OpenAI -----------------------------------------------------------------------
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "FACTLENS_OPENAI_API_KEY"),
    )
    chat_model: str = "gpt-6-luna"
    embedding_model: str = "text-embedding-3-small"
    scope_model: str | None = None
    synthesis_model: str | None = None
    verification_model: str | None = None
    # ``None`` leaves sampling at the model default (some models reject custom values).
    chat_temperature: float | None = None
    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 2

    # --- Uploads ----------------------------------------------------------------------
    max_file_size_bytes: int = constants.DEFAULT_MAX_FILE_SIZE_BYTES
    max_total_size_bytes: int = constants.DEFAULT_MAX_TOTAL_SIZE_BYTES

    # --- Parsing ------------------------------------------------------------------------
    # "auto": parse PDFs without OCR and retry with OCR only if almost no text comes out
    # (scanned documents). "never" is fastest; "always" OCRs every image region (slowest).
    ocr_mode: Literal["auto", "always", "never"] = "auto"
    # In "auto" mode, a PDF yielding fewer text characters than this is retried with OCR.
    ocr_fallback_min_chars: int = 200

    # --- Chunking ---------------------------------------------------------------------
    chunk_max_chars: int = 2000
    chunk_overlap_chars: int = 200

    # --- Retrieval --------------------------------------------------------------------
    vector_search_k: int = 10
    bm25_search_k: int = 4
    # Weights for [lexical, vector] results in reciprocal-rank fusion.
    hybrid_weights: list[float] = Field(default_factory=lambda: [0.4, 0.6])
    scope_check_top_k: int = 20

    # --- Pipeline ---------------------------------------------------------------------
    # How many times the answer may be revised after a failed verification.
    max_revisions: int = 2

    # --- Storage ----------------------------------------------------------------------
    data_dir: Path = Path(".factlens")
    cache_expire_days: int = 7

    # --- Logging ----------------------------------------------------------------------
    log_level: str = "INFO"
    log_file: Path | None = None

    # --- Server -----------------------------------------------------------------------
    server_host: str = "127.0.0.1"
    server_port: int = 7860
    # Creates a public Gradio link when true. Off by default: documents may be sensitive.
    share: bool = False
    examples_dir: Path = Path("examples")

    @field_validator("hybrid_weights")
    @classmethod
    def _validate_hybrid_weights(cls, value: list[float]) -> list[float]:
        if len(value) != 2 or any(weight < 0 for weight in value) or sum(value) <= 0:
            raise ValueError("hybrid_weights must be two non-negative numbers: [lexical, vector]")
        return value

    @property
    def cache_dir(self) -> Path:
        """Directory holding parsed-document caches."""
        return self.data_dir / "cache"

    @property
    def vector_store_dir(self) -> Path:
        """Directory holding the persistent vector store."""
        return self.data_dir / "vectors"

    def resolve_model(self, role: str) -> str:
        """Return the chat model for an agent role: scope, synthesis or verification."""
        override = {
            "scope": self.scope_model,
            "synthesis": self.synthesis_model,
            "verification": self.verification_model,
        }.get(role)
        return override or self.chat_model

    def require_openai_api_key(self) -> str:
        """Return the OpenAI API key or raise a clear error if it is not configured."""
        if self.openai_api_key is None or not self.openai_api_key.get_secret_value().strip():
            raise ConfigurationError(
                "OPENAI_API_KEY is not set. Add it to your environment or to a .env file "
                "(see .env.example)."
            )
        return self.openai_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
