from __future__ import annotations

import pytest

from factlens.agents import AnswerSynthesizer, ClaimVerifier, ScopeClassifier
from factlens.core.constants import NO_ANSWER_MESSAGE
from factlens.core.errors import LLMError
from factlens.schemas import ScopeVerdict
from tests.conftest import FailingChatModel, RecordingChatModel, make_documents


# --- ScopeClassifier ------------------------------------------------------------------
@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("CAN_ANSWER", ScopeVerdict.ANSWERABLE),
        ("partial", ScopeVerdict.PARTIAL),
        ('"NO_MATCH".', ScopeVerdict.OUT_OF_SCOPE),
        ("**CAN_ANSWER**", ScopeVerdict.ANSWERABLE),
        ("I think it is probably fine", ScopeVerdict.OUT_OF_SCOPE),
        ("", ScopeVerdict.OUT_OF_SCOPE),
    ],
)
def test_scope_classifier_parses_labels(reply: str, expected: ScopeVerdict) -> None:
    model = RecordingChatModel(responses=[reply])
    verdict = ScopeClassifier(model, top_k=3).classify("q?", make_documents("passage"))
    assert verdict is expected


def test_scope_classifier_skips_the_model_when_nothing_was_retrieved() -> None:
    model = RecordingChatModel(responses=["CAN_ANSWER"])
    assert ScopeClassifier(model, top_k=3).classify("q?", []) is ScopeVerdict.OUT_OF_SCOPE
    assert model.prompts == []


def test_scope_classifier_only_shows_the_top_k_passages() -> None:
    model = RecordingChatModel(responses=["PARTIAL"])
    docs = make_documents("first-passage", "second-passage", "third-passage")
    ScopeClassifier(model, top_k=2).classify("what is it?", docs)
    prompt = model.prompts[0]
    assert "what is it?" in prompt
    assert "first-passage" in prompt
    assert "second-passage" in prompt
    assert "third-passage" not in prompt


def test_model_failures_surface_as_llm_errors_not_silent_no_match() -> None:
    classifier = ScopeClassifier(FailingChatModel(responses=["x"]), top_k=3)
    with pytest.raises(LLMError, match="Scope check failed"):
        classifier.classify("q?", make_documents("passage"))


# --- AnswerSynthesizer ----------------------------------------------------------------
def test_synthesizer_returns_model_answer_and_uses_context() -> None:
    model = RecordingChatModel(responses=["  The PUE was 1.10.  "])
    answer = AnswerSynthesizer(model).synthesize("What was PUE?", make_documents("PUE = 1.10"))
    assert answer == "The PUE was 1.10."
    assert "PUE = 1.10" in model.prompts[0]
    assert "Reviewer feedback" not in model.prompts[0]


def test_synthesizer_includes_feedback_on_revision() -> None:
    model = RecordingChatModel(responses=["revised"])
    AnswerSynthesizer(model).synthesize(
        "q?", make_documents("ctx"), feedback="Unsupported claims: revenue doubled"
    )
    assert "Reviewer feedback" in model.prompts[0]
    assert "revenue doubled" in model.prompts[0]


def test_synthesizer_falls_back_when_model_returns_nothing() -> None:
    model = RecordingChatModel(responses=["   "])
    assert AnswerSynthesizer(model).synthesize("q?", make_documents("ctx")) == NO_ANSWER_MESSAGE


# --- ClaimVerifier --------------------------------------------------------------------
def _verify(reply: str):  # type: ignore[no-untyped-def]
    model = RecordingChatModel(responses=[reply])
    return ClaimVerifier(model).verify("q?", "answer", make_documents("ctx")), model


def test_verifier_parses_the_requested_format() -> None:
    report, _ = _verify(
        "Supported: YES\n"
        "Unsupported Claims: []\n"
        "Contradictions: []\n"
        "Relevant: YES\n"
        "Additional Details: Everything checks out."
    )
    assert report.supported and report.relevant
    assert report.unsupported_claims == [] and report.contradictions == []
    assert report.additional_details == "Everything checks out."
    assert not report.needs_revision


def test_verifier_parses_lists_and_flags_revision() -> None:
    report, _ = _verify(
        "Supported: NO\n"
        "Unsupported Claims: [\"revenue doubled\", 'margin rose']\n"
        "Contradictions: [PUE was 1.2 not 1.1]\n"
        "Relevant: YES\n"
        "Additional Details: [Two claims lack support]"
    )
    assert not report.supported
    assert report.unsupported_claims == ["revenue doubled", "margin rose"]
    assert report.contradictions == ["PUE was 1.2 not 1.1"]
    assert report.additional_details == "Two claims lack support"
    assert report.needs_revision


def test_verifier_tolerates_markdown_bold_and_bullets() -> None:
    report, _ = _verify(
        "- **Supported:** YES\n"
        "- **Unsupported Claims:** None\n"
        "- **Contradictions:** None\n"
        "- **Relevant:** Yes, directly\n"
        "- **Additional Details:** Looks fine"
    )
    assert report.supported and report.relevant
    assert report.unsupported_claims == [] and report.contradictions == []


def test_verifier_joins_multi_line_details() -> None:
    report, _ = _verify(
        "Supported: YES\nRelevant: YES\nAdditional Details: first line\nsecond line\n"
    )
    assert report.additional_details == "first line second line"


def test_contradictions_force_revision_even_if_marked_supported() -> None:
    report, _ = _verify("Supported: YES\nContradictions: [x]\nRelevant: YES")
    assert report.needs_revision


def test_unparseable_reply_yields_a_failing_report() -> None:
    report, _ = _verify("I could not do that.")
    assert not report.supported and not report.relevant
    assert "Failed to parse" in report.additional_details
    assert report.needs_revision


def test_verifier_prompt_contains_question_answer_and_context() -> None:
    _, model = _verify("Supported: YES\nRelevant: YES")
    prompt = model.prompts[0]
    assert "q?" in prompt and "answer" in prompt and "ctx" in prompt
