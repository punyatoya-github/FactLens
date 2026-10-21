"""Hybrid (lexical + vector) retrieval."""

from factlens.retrieval.base import Retriever
from factlens.retrieval.hybrid import (
    HybridRetriever,
    build_hybrid_retriever,
    weighted_reciprocal_rank_fusion,
)
from factlens.retrieval.lexical import LexicalRetriever

__all__ = [
    "HybridRetriever",
    "LexicalRetriever",
    "Retriever",
    "build_hybrid_retriever",
    "weighted_reciprocal_rank_fusion",
]
