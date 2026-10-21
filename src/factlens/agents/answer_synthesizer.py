"""Drafts an answer from retrieved passages."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from factlens.core.constants import NO_ANSWER_MESSAGE
from factlens.llm.clients import generate_text

_PROMPT_TEMPLATE = """\
You are an AI assistant designed to provide precise and factual answers based on the given \
context.

**Instructions:**
- Answer the following question using only the provided context.
- Be clear, concise, and factual.
- Return as much information as you can get from the context.

**Question:** {question}
**Context:**
{context}
{feedback_section}
**Provide your answer below:**
"""

_FEEDBACK_TEMPLATE = """
**Reviewer feedback on your previous draft:**
{feedback}
Fix these problems. Remove or correct any unsupported claims and contradictions, and rely \
strictly on the context above.
"""


class AnswerSynthesizer:
    """Writes a draft answer grounded in the retrieved context."""

    def __init__(self, chat_model: BaseChatModel) -> None:
        self._model = chat_model

    def synthesize(
        self,
        question: str,
        documents: Sequence[Document],
        *,
        feedback: str | None = None,
    ) -> str:
        """Return a draft answer, optionally revising a previous draft given reviewer feedback."""
        context = "\n\n".join(document.page_content for document in documents)
        feedback_section = _FEEDBACK_TEMPLATE.format(feedback=feedback) if feedback else ""
        prompt = _PROMPT_TEMPLATE.format(
            question=question, context=context, feedback_section=feedback_section
        )
        answer = generate_text(self._model, prompt, purpose="Answer drafting")
        return answer or NO_ANSWER_MESSAGE
