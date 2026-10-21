"""BM25 keyword retrieval."""

from __future__ import annotations

import re
from collections.abc import Sequence

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lower-case ``text`` and split it into word tokens, ignoring punctuation."""
    return _TOKEN_PATTERN.findall(text.lower())


class LexicalRetriever:
    """Ranks chunks with BM25 and returns the top ``k`` that match at least one query term."""

    def __init__(self, documents: Sequence[Document], k: int) -> None:
        self._documents = list(documents)
        self._k = k
        corpus = [tokenize(document.page_content) for document in self._documents]
        self._bm25: BM25Okapi | None = BM25Okapi(corpus) if any(corpus) else None

    def invoke(self, input: str, /) -> list[Document]:
        """Return up to ``k`` matching chunks, best first."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(input))
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        return [self._documents[index] for index in ranked[: self._k] if scores[index] > 0]
