"""Hybrid retrieval: BM25 and vector search fused with weighted reciprocal-rank fusion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from loguru import logger

from factlens.core.settings import Settings
from factlens.ingestion.hashing import combine_hashes, hash_text
from factlens.retrieval.base import Retriever
from factlens.retrieval.lexical import LexicalRetriever

DEFAULT_RANK_CONSTANT = 60


def weighted_reciprocal_rank_fusion(
    rank_lists: Sequence[Sequence[Document]],
    weights: Sequence[float],
    rank_constant: int = DEFAULT_RANK_CONSTANT,
) -> list[Document]:
    """Merge ranked lists into one, scoring each chunk by ``sum(weight / (rank + c))``.

    Chunks are identified by their text, so the same passage returned by several retrievers
    is merged and scored cumulatively. Ties keep their first-seen order.
    """
    if len(rank_lists) != len(weights):
        raise ValueError("Number of rank lists must equal the number of weights.")
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, Document] = {}
    for rank_list, weight in zip(rank_lists, weights, strict=True):
        for rank, document in enumerate(rank_list, start=1):
            scores[document.page_content] += weight / (rank + rank_constant)
            first_seen.setdefault(document.page_content, document)
    ordered = sorted(first_seen, key=lambda content: scores[content], reverse=True)
    return [first_seen[content] for content in ordered]


class HybridRetriever:
    """Runs several retrievers and fuses their rankings."""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        weights: Sequence[float],
        rank_constant: int = DEFAULT_RANK_CONSTANT,
    ) -> None:
        if len(retrievers) != len(weights):
            raise ValueError("Number of retrievers must equal the number of weights.")
        self._retrievers = list(retrievers)
        self._weights = list(weights)
        self._rank_constant = rank_constant

    def invoke(self, input: str, /) -> list[Document]:
        """Return fused results for ``input``, best first."""
        rank_lists = [retriever.invoke(input) for retriever in self._retrievers]
        return weighted_reciprocal_rank_fusion(rank_lists, self._weights, self._rank_constant)


def _collection_name(chunks: Sequence[Document], embedding_model: str) -> str:
    """Name a vector collection after its content and embedding model.

    Identical chunks embedded with the same model always map to the same collection (so it can
    be reused), while any change in content or model maps to a fresh one. This also keeps one
    upload's chunks from ever leaking into another's results.
    """
    digest = combine_hashes((hash_text(chunk.page_content) for chunk in chunks), embedding_model)
    return f"factlens-{digest[:24]}"


def build_hybrid_retriever(
    chunks: Sequence[Document],
    *,
    settings: Settings,
    embeddings: Embeddings,
) -> HybridRetriever:
    """Build a BM25 + vector retriever over ``chunks``, reusing stored embeddings if possible."""
    usable = [chunk for chunk in chunks if chunk.page_content.strip()]
    if not usable:
        raise ValueError("Cannot build a retriever without any non-empty chunks.")

    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(settings.vector_store_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    name = _collection_name(usable, settings.embedding_model)

    if client.get_or_create_collection(name).count() == len(usable):
        logger.info("Reusing stored embeddings ({})", name)
        vector_store = Chroma(client=client, collection_name=name, embedding_function=embeddings)
    else:
        client.delete_collection(name)
        logger.info("Embedding {} chunks ({})", len(usable), name)
        vector_store = Chroma.from_documents(
            documents=usable,
            embedding=embeddings,
            client=client,
            collection_name=name,
            ids=[f"chunk-{index}" for index in range(len(usable))],
        )

    lexical = LexicalRetriever(usable, k=settings.bm25_search_k)
    vector = vector_store.as_retriever(search_kwargs={"k": settings.vector_search_k})
    return HybridRetriever([lexical, vector], settings.hybrid_weights)
