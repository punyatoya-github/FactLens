"""Shared fixtures and fakes. Nothing here touches the network."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from pydantic import Field

from factlens.core.settings import Settings


class RecordingChatModel(FakeListChatModel):
    """A scripted chat model that remembers every prompt it received."""

    prompts: list[str] = Field(default_factory=list)

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        self.prompts.append(str(messages[-1].content))
        return super()._call(messages, stop, run_manager, **kwargs)


class FailingChatModel(FakeListChatModel):
    """A chat model whose every call raises, to exercise error handling."""

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        raise RuntimeError("boom")


class CountingEmbeddings(DeterministicFakeEmbedding):
    """Deterministic embeddings that count how many texts were embedded."""

    embedded_texts: int = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts += len(texts)
        return super().embed_documents(texts)


class StaticRetriever:
    """A retriever that always returns the same documents."""

    def __init__(self, documents: Sequence[Document]) -> None:
        self._documents = list(documents)
        self.queries: list[str] = []

    def invoke(self, input: str, /) -> list[Document]:
        self.queries.append(input)
        return list(self._documents)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        openai_api_key="test-key",
        data_dir=tmp_path / ".factlens",
        examples_dir=tmp_path / "examples",
    )


@pytest.fixture
def make_chat_model() -> type[RecordingChatModel]:
    return RecordingChatModel


@pytest.fixture
def embeddings() -> CountingEmbeddings:
    return CountingEmbeddings(size=32)


def make_documents(*texts: str) -> list[Document]:
    return [Document(page_content=text, metadata={"source": "test.md"}) for text in texts]
