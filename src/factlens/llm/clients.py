"""Construction of OpenAI chat and embedding clients, plus a safe text-generation helper."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from factlens.core.errors import LLMError
from factlens.core.settings import Settings

AgentRole = Literal["scope", "synthesis", "verification"]


def build_chat_model(settings: Settings, role: AgentRole = "synthesis") -> BaseChatModel:
    """Create the OpenAI chat model configured for an agent ``role``."""
    settings.require_openai_api_key()
    options: dict[str, Any] = {}
    if settings.chat_temperature is not None:
        options["temperature"] = settings.chat_temperature
    return ChatOpenAI(
        model=settings.resolve_model(role),
        api_key=settings.openai_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        **options,
    )


def build_embeddings(settings: Settings) -> Embeddings:
    """Create the OpenAI embedding model."""
    settings.require_openai_api_key()
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


def generate_text(model: BaseChatModel, prompt: str, *, purpose: str) -> str:
    """Send ``prompt`` to ``model`` and return the stripped text reply.

    Any failure is re-raised as ``LLMError`` naming the step that failed, so the UI can show
    a real error instead of silently treating the failure as "no answer".
    """
    try:
        return str(model.invoke(prompt).text).strip()
    except Exception as error:
        raise LLMError(f"{purpose} failed: {error}") from error
