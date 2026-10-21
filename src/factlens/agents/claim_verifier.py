"""Checks a draft answer against the passages it was drawn from."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger

from factlens.llm.clients import generate_text
from factlens.schemas.verdicts import VerificationReport

_PROMPT_TEMPLATE = """\
You are an AI assistant designed to verify the accuracy and relevance of answers based on \
provided context.

**Instructions:**
- Verify the following answer against the provided context.
- Check for:
1. Direct/indirect factual support (YES/NO)
2. Unsupported claims (list any if present)
3. Contradictions (list any if present)
4. Relevance to the question (YES/NO)
- Provide additional details or explanations where relevant.
- Respond in the exact format specified below without adding any unrelated information.

**Format:**
Supported: YES/NO
Unsupported Claims: [item1, item2, ...]
Contradictions: [item1, item2, ...]
Relevant: YES/NO
Additional Details: [Any extra information or explanations]

**Question:** {question}
**Answer:** {answer}
**Context:**
{context}

**Respond ONLY with the above format.**
"""

_FIELDS = (
    "supported",
    "unsupported claims",
    "contradictions",
    "relevant",
    "additional details",
)
_EMPTY_LIST_MARKERS = {"", "none", "n/a", "na", "no"}


class ClaimVerifier:
    """Verifies that a draft answer is supported by, and relevant to, the retrieved passages."""

    def __init__(self, chat_model: BaseChatModel) -> None:
        self._model = chat_model

    def verify(
        self, question: str, answer: str, documents: Sequence[Document]
    ) -> VerificationReport:
        """Return a verification report for ``answer`` given the source ``documents``."""
        context = "\n\n".join(document.page_content for document in documents)
        prompt = _PROMPT_TEMPLATE.format(question=question, answer=answer, context=context)
        reply = generate_text(self._model, prompt, purpose="Answer verification")
        return self._parse_report(reply)

    @staticmethod
    def _parse_report(reply: str) -> VerificationReport:
        """Parse the model's ``Key: value`` reply, tolerating markdown bullets and bold."""
        values: dict[str, str] = {}
        current: str | None = None
        for raw_line in reply.splitlines():
            line = raw_line.replace("*", "").strip().lstrip("-•#> ").strip()
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() in _FIELDS:
                current = key.strip().lower()
                values[current] = value.strip()
            elif current == "additional details" and line:
                values[current] = f"{values[current]} {line}".strip()

        if not values:
            logger.warning("Verifier reply could not be parsed: {!r}", reply[:200])
            return VerificationReport(additional_details="Failed to parse the model's response.")

        return VerificationReport(
            supported=_parse_yes_no(values.get("supported", "")),
            unsupported_claims=_parse_list(values.get("unsupported claims", "")),
            contradictions=_parse_list(values.get("contradictions", "")),
            relevant=_parse_yes_no(values.get("relevant", "")),
            additional_details=values.get("additional details", "").strip("[] "),
        )


def _parse_yes_no(value: str) -> bool:
    return value.strip().upper().startswith("YES")


def _parse_list(value: str) -> list[str]:
    inner = value.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    if inner.strip().lower() in _EMPTY_LIST_MARKERS:
        return []
    return [item.strip().strip("\"'") for item in inner.split(",") if item.strip()]
