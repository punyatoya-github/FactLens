"""Decides whether the retrieved passages can answer a question at all."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger

from factlens.llm.clients import generate_text
from factlens.schemas.verdicts import ScopeVerdict

_PROMPT_TEMPLATE = """\
You are an AI relevance checker between a user's question and provided document content.

**Instructions:**
- Classify how well the document content addresses the user's question.
- Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.
- Do not include any additional text or explanation.

**Labels:**
1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all the \
details needed for a complete answer.
3) "NO_MATCH": The passages do not discuss or mention the question's topic at all.

**Important:** If the passages mention or reference the topic or timeframe of the question in \
any way, even if incomplete, respond with "PARTIAL" instead of "NO_MATCH".

**Question:** {question}
**Passages:** {passages}

**Respond ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH**
"""


class ScopeClassifier:
    """Classifies a question as answerable, partially answerable, or out of scope."""

    def __init__(self, chat_model: BaseChatModel, *, top_k: int) -> None:
        self._model = chat_model
        self._top_k = top_k

    def classify(self, question: str, documents: Sequence[Document]) -> ScopeVerdict:
        """Return how well the top retrieved ``documents`` cover ``question``."""
        if not documents:
            logger.debug("No documents retrieved; classifying as out of scope")
            return ScopeVerdict.OUT_OF_SCOPE
        passages = "\n\n".join(document.page_content for document in documents[: self._top_k])
        prompt = _PROMPT_TEMPLATE.format(question=question, passages=passages)
        reply = generate_text(self._model, prompt, purpose="Scope check")
        return self._parse_verdict(reply)

    @staticmethod
    def _parse_verdict(reply: str) -> ScopeVerdict:
        label = reply.strip().strip("\"'`*. ").upper()
        try:
            verdict = ScopeVerdict(label)
        except ValueError:
            logger.warning("Scope classifier returned an unrecognised label: {!r}", reply)
            return ScopeVerdict.OUT_OF_SCOPE
        logger.debug("Scope verdict: {}", verdict)
        return verdict
