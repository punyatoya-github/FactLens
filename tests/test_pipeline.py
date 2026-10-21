from __future__ import annotations

from langchain_core.documents import Document

from factlens.core.constants import OUT_OF_SCOPE_MESSAGE
from factlens.pipeline import AnswerPipeline, format_revision_feedback
from factlens.schemas import ScopeVerdict, VerificationReport
from tests.conftest import StaticRetriever, make_documents

PASSING = VerificationReport(supported=True, relevant=True)
FAILING = VerificationReport(
    supported=False,
    relevant=True,
    unsupported_claims=["revenue doubled"],
    additional_details="Not in the context",
)


class StubClassifier:
    def __init__(self, verdict: ScopeVerdict) -> None:
        self.verdict = verdict
        self.calls = 0

    def classify(self, question: str, documents: list[Document]) -> ScopeVerdict:
        self.calls += 1
        return self.verdict


class StubSynthesizer:
    def __init__(self) -> None:
        self.feedbacks: list[str | None] = []

    def synthesize(
        self, question: str, documents: list[Document], *, feedback: str | None = None
    ) -> str:
        self.feedbacks.append(feedback)
        return f"draft {len(self.feedbacks)}"


class StubVerifier:
    def __init__(self, reports: list[VerificationReport]) -> None:
        self._reports = reports
        self.calls: list[tuple[str, str]] = []

    def verify(self, question: str, answer: str, documents: list[Document]) -> VerificationReport:
        self.calls.append((question, answer))
        return self._reports[min(len(self.calls) - 1, len(self._reports) - 1)]


def _pipeline(
    verdict: ScopeVerdict, reports: list[VerificationReport], max_revisions: int = 2
) -> tuple[AnswerPipeline, StubClassifier, StubSynthesizer, StubVerifier]:
    classifier, synthesizer, verifier = (
        StubClassifier(verdict),
        StubSynthesizer(),
        StubVerifier(reports),
    )
    pipeline = AnswerPipeline(
        classifier,  # type: ignore[arg-type]
        synthesizer,  # type: ignore[arg-type]
        verifier,  # type: ignore[arg-type]
        max_revisions=max_revisions,
    )
    return pipeline, classifier, synthesizer, verifier


def test_out_of_scope_question_skips_drafting_and_verification() -> None:
    pipeline, _, synthesizer, verifier = _pipeline(ScopeVerdict.OUT_OF_SCOPE, [PASSING])
    result = pipeline.run("weather?", StaticRetriever(make_documents("ctx")))
    assert result.answer == OUT_OF_SCOPE_MESSAGE
    assert result.scope_verdict is ScopeVerdict.OUT_OF_SCOPE
    assert result.verification_report is None
    assert synthesizer.feedbacks == [] and verifier.calls == []


def test_empty_retrieval_is_handled_by_the_classifier_path() -> None:
    pipeline, classifier, _, _ = _pipeline(ScopeVerdict.OUT_OF_SCOPE, [PASSING])
    pipeline.run("q?", StaticRetriever([]))
    assert classifier.calls == 1


def test_supported_answer_passes_on_the_first_draft() -> None:
    pipeline, _, synthesizer, verifier = _pipeline(ScopeVerdict.ANSWERABLE, [PASSING])
    result = pipeline.run("q?", StaticRetriever(make_documents("ctx")))
    assert result.answer == "draft 1"
    assert result.verification_report == PASSING
    assert result.revisions_used == 0
    assert synthesizer.feedbacks == [None]
    assert verifier.calls == [("q?", "draft 1")]


def test_partial_scope_still_gets_an_answer() -> None:
    pipeline, *_ = _pipeline(ScopeVerdict.PARTIAL, [PASSING])
    result = pipeline.run("q?", StaticRetriever(make_documents("ctx")))
    assert result.answer == "draft 1"
    assert result.scope_verdict is ScopeVerdict.PARTIAL


def test_failed_verification_triggers_a_revision_with_feedback() -> None:
    pipeline, _, synthesizer, verifier = _pipeline(ScopeVerdict.ANSWERABLE, [FAILING, PASSING])
    result = pipeline.run("q?", StaticRetriever(make_documents("ctx")))
    assert result.answer == "draft 2"
    assert result.revisions_used == 1
    assert synthesizer.feedbacks[0] is None
    assert synthesizer.feedbacks[1] is not None and "revenue doubled" in synthesizer.feedbacks[1]
    assert verifier.calls == [("q?", "draft 1"), ("q?", "draft 2")]
    assert result.verification_report == PASSING


def test_revisions_are_capped_and_the_last_report_is_returned() -> None:
    pipeline, _, synthesizer, verifier = _pipeline(
        ScopeVerdict.ANSWERABLE, [FAILING], max_revisions=2
    )
    result = pipeline.run("q?", StaticRetriever(make_documents("ctx")))
    assert len(synthesizer.feedbacks) == 3  # first draft + 2 revisions
    assert len(verifier.calls) == 3
    assert result.revisions_used == 2
    assert result.verification_report is not None and result.verification_report.needs_revision


def test_zero_max_revisions_never_revises() -> None:
    pipeline, _, synthesizer, _ = _pipeline(ScopeVerdict.ANSWERABLE, [FAILING], max_revisions=0)
    result = pipeline.run("q?", StaticRetriever(make_documents("ctx")))
    assert len(synthesizer.feedbacks) == 1
    assert result.revisions_used == 0


def test_feedback_lists_every_problem() -> None:
    report = VerificationReport(
        supported=False,
        relevant=False,
        unsupported_claims=["a", "b"],
        contradictions=["c"],
        additional_details="notes",
    )
    feedback = format_revision_feedback(report)
    for fragment in ("a; b", "c", "did not address the question", "notes"):
        assert fragment in feedback


def test_feedback_has_a_default_message() -> None:
    report = VerificationReport(supported=False, relevant=True)
    assert "not fully supported" in format_revision_feedback(report)
