"""Verdicts and results produced by the agents and the pipeline."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ScopeVerdict(StrEnum):
    """How well the retrieved passages cover a question.

    The enum values are the labels the scope classifier prompt asks the model to emit.
    """

    ANSWERABLE = "CAN_ANSWER"
    PARTIAL = "PARTIAL"
    OUT_OF_SCOPE = "NO_MATCH"

    @property
    def is_in_scope(self) -> bool:
        """Whether the pipeline should go on to draft an answer."""
        return self is not ScopeVerdict.OUT_OF_SCOPE


class VerificationReport(BaseModel):
    """Outcome of checking a draft answer against the retrieved passages."""

    supported: bool = False
    unsupported_claims: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    relevant: bool = False
    additional_details: str = ""

    @property
    def needs_revision(self) -> bool:
        """True when the draft is unsupported, off-topic, or contradicts the sources."""
        return not self.supported or not self.relevant or bool(self.contradictions)


class PipelineResult(BaseModel):
    """Final output of one question run through the answer pipeline."""

    answer: str
    scope_verdict: ScopeVerdict
    verification_report: VerificationReport | None = None
    revisions_used: int = 0
