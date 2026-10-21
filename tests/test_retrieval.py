from __future__ import annotations

import pytest
from langchain_core.documents import Document

from factlens.core.settings import Settings
from factlens.retrieval import (
    HybridRetriever,
    LexicalRetriever,
    build_hybrid_retriever,
    weighted_reciprocal_rank_fusion,
)
from factlens.retrieval.lexical import tokenize
from tests.conftest import CountingEmbeddings, StaticRetriever, make_documents


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("PUE: 1.10, in Singapore!") == ["pue", "1", "10", "in", "singapore"]


def test_lexical_retriever_ranks_matching_chunks_first() -> None:
    # BM25 gives a term that appears in half the corpus an IDF of zero, so the corpus needs
    # enough unrelated chunks to look like a real document.
    filler = [f"unrelated gardening note number {index}" for index in range(6)]
    docs = make_documents(
        "solar panels reduce emissions",
        "data center cooling efficiency",
        "cooling towers and water use",
        *filler,
    )
    results = LexicalRetriever(docs, k=2).invoke("data center cooling")
    assert results[0].page_content == "data center cooling efficiency"
    assert len(results) == 2


def test_lexical_retriever_returns_nothing_without_term_overlap() -> None:
    docs = make_documents("alpha beta", "gamma delta")
    assert LexicalRetriever(docs, k=3).invoke("zebra") == []


def test_lexical_retriever_handles_empty_corpus() -> None:
    assert LexicalRetriever([], k=3).invoke("anything") == []


def test_fusion_scores_follow_weighted_rrf_formula() -> None:
    a, b, c = (Document(page_content=text) for text in "abc")
    # a: ranked 1st in list one (w=0.4) and 2nd in list two (w=0.6)
    # b: ranked 2nd in list one, c: ranked 1st in list two
    fused = weighted_reciprocal_rank_fusion([[a, b], [c, a]], [0.4, 0.6], rank_constant=60)
    score_a = 0.4 / 61 + 0.6 / 62
    score_c = 0.6 / 61
    score_b = 0.4 / 62
    assert score_a > score_c > score_b
    assert [d.page_content for d in fused] == ["a", "c", "b"]


def test_fusion_merges_duplicates_across_lists() -> None:
    a, a_again = Document(page_content="a"), Document(page_content="a")
    fused = weighted_reciprocal_rank_fusion([[a], [a_again]], [0.5, 0.5])
    assert len(fused) == 1


def test_fusion_rejects_mismatched_weights() -> None:
    with pytest.raises(ValueError, match="weights"):
        weighted_reciprocal_rank_fusion([[Document(page_content="a")]], [0.5, 0.5])


def test_hybrid_retriever_queries_every_retriever() -> None:
    first = StaticRetriever(make_documents("one"))
    second = StaticRetriever(make_documents("two"))
    results = HybridRetriever([first, second], [0.5, 0.5]).invoke("q")
    assert {d.page_content for d in results} == {"one", "two"}
    assert first.queries == second.queries == ["q"]


def test_build_hybrid_retriever_returns_relevant_chunks(
    settings: Settings, embeddings: CountingEmbeddings
) -> None:
    docs = make_documents(
        "PUE efficiency in Singapore data center 2022",
        "regional carbon free energy in Asia Pacific",
        "employee volunteering programs",
    )
    retriever = build_hybrid_retriever(docs, settings=settings, embeddings=embeddings)
    results = retriever.invoke("Singapore data center PUE")
    assert results
    assert results[0].page_content.startswith("PUE efficiency")


def test_stored_embeddings_are_reused_for_identical_chunks(
    settings: Settings, embeddings: CountingEmbeddings
) -> None:
    docs = make_documents("first passage here", "second passage there")
    build_hybrid_retriever(docs, settings=settings, embeddings=embeddings)
    embedded_after_first = embeddings.embedded_texts
    assert embedded_after_first == 2
    build_hybrid_retriever(docs, settings=settings, embeddings=embeddings)
    assert embeddings.embedded_texts == embedded_after_first


def test_new_upload_never_leaks_chunks_from_a_previous_one(
    settings: Settings, embeddings: CountingEmbeddings
) -> None:
    first = make_documents("secret budget figures for project falcon")
    second = make_documents("recipe for lentil soup with carrots")
    build_hybrid_retriever(first, settings=settings, embeddings=embeddings)
    retriever = build_hybrid_retriever(second, settings=settings, embeddings=embeddings)
    for query in ("falcon budget", "lentil soup"):
        contents = {d.page_content for d in retriever.invoke(query)}
        assert contents <= {"recipe for lentil soup with carrots"}


def test_changing_the_embedding_model_forces_a_rebuild(
    settings: Settings, embeddings: CountingEmbeddings
) -> None:
    docs = make_documents("some passage")
    build_hybrid_retriever(docs, settings=settings, embeddings=embeddings)
    other = settings.model_copy(update={"embedding_model": "text-embedding-3-large"})
    build_hybrid_retriever(docs, settings=other, embeddings=embeddings)
    assert embeddings.embedded_texts == 2


def test_building_without_usable_chunks_fails(
    settings: Settings, embeddings: CountingEmbeddings
) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_hybrid_retriever(make_documents("   "), settings=settings, embeddings=embeddings)
