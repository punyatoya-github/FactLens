"""The minimal interface every retriever in FactLens satisfies."""

from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document


class Retriever(Protocol):
    """Anything that can turn a query string into a ranked list of chunks."""

    def invoke(self, input: str, /) -> list[Document]:
        """Return chunks ranked from most to least relevant."""
        ...
