"""Entry point for answering a question over a retriever."""

from __future__ import annotations

from typing import Any

from loguru import logger

from factlens.agents import AnswerSynthesizer, ClaimVerifier, ScopeClassifier
from factlens.core.settings import Settings
from factlens.llm.clients import build_chat_model
from factlens.pipeline.graph import PipelineState, build_answer_graph
from factlens.retrieval.base import Retriever
from factlens.schemas.verdicts import PipelineResult, ScopeVerdict


class AnswerPipeline:
    """Retrieves context for a question and runs it through the multi-agent graph."""

    def __init__(
        self,
        scope_classifier: ScopeClassifier,
        synthesizer: AnswerSynthesizer,
        verifier: ClaimVerifier,
        *,
        max_revisions: int,
    ) -> None:
        self._graph: Any = build_answer_graph(
            scope_classifier=scope_classifier,
            synthesizer=synthesizer,
            verifier=verifier,
            max_revisions=max_revisions,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> AnswerPipeline:
        """Build a pipeline whose agents use the OpenAI models configured in ``settings``."""
        return cls(
            ScopeClassifier(build_chat_model(settings, "scope"), top_k=settings.scope_check_top_k),
            AnswerSynthesizer(build_chat_model(settings, "synthesis")),
            ClaimVerifier(build_chat_model(settings, "verification")),
            max_revisions=settings.max_revisions,
        )

    def run(self, question: str, retriever: Retriever) -> PipelineResult:
        """Answer ``question`` using passages found by ``retriever``."""
        documents = retriever.invoke(question)
        logger.info("Retrieved {} chunks for the question", len(documents))
        initial_state: PipelineState = {
            "question": question,
            "documents": documents,
            "scope_verdict": None,
            "draft_answer": "",
            "verification_report": None,
            "revision_count": 0,
        }
        final_state: PipelineState = self._graph.invoke(initial_state)
        return PipelineResult(
            answer=final_state["draft_answer"],
            scope_verdict=final_state["scope_verdict"] or ScopeVerdict.OUT_OF_SCOPE,
            verification_report=final_state["verification_report"],
            revisions_used=final_state["revision_count"],
        )
