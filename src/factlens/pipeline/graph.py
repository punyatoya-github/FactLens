"""LangGraph wiring: scope check -> synthesis -> verification, with bounded revision."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from factlens.agents import AnswerSynthesizer, ClaimVerifier, ScopeClassifier
from factlens.core.constants import OUT_OF_SCOPE_MESSAGE
from factlens.schemas.verdicts import ScopeVerdict, VerificationReport


class PipelineState(TypedDict):
    """Data passed between pipeline nodes."""

    question: str
    documents: list[Document]
    scope_verdict: ScopeVerdict | None
    draft_answer: str
    verification_report: VerificationReport | None
    revision_count: int


def format_revision_feedback(report: VerificationReport) -> str:
    """Turn a failed verification report into instructions for the next draft."""
    lines: list[str] = []
    if report.unsupported_claims:
        lines.append("Unsupported claims: " + "; ".join(report.unsupported_claims))
    if report.contradictions:
        lines.append("Contradictions with the context: " + "; ".join(report.contradictions))
    if not report.relevant:
        lines.append("The answer did not address the question.")
    if report.additional_details:
        lines.append("Reviewer notes: " + report.additional_details)
    return "\n".join(lines) or "The answer was not fully supported by the context."


def route_after_scope_check(state: PipelineState) -> Literal["synthesize", "end"]:
    """Continue to drafting only when the question is in scope."""
    verdict = state["scope_verdict"]
    return "synthesize" if verdict is not None and verdict.is_in_scope else "end"


def build_answer_graph(
    *,
    scope_classifier: ScopeClassifier,
    synthesizer: AnswerSynthesizer,
    verifier: ClaimVerifier,
    max_revisions: int,
) -> Any:
    """Compile the answer pipeline graph.

    After a failed verification the draft is revised with the verifier's feedback, at most
    ``max_revisions`` times; the last draft and its report are returned either way.
    """

    def scope_check_node(state: PipelineState) -> dict[str, Any]:
        verdict = scope_classifier.classify(state["question"], state["documents"])
        update: dict[str, Any] = {"scope_verdict": verdict}
        if not verdict.is_in_scope:
            update["draft_answer"] = OUT_OF_SCOPE_MESSAGE
        return update

    def synthesis_node(state: PipelineState) -> dict[str, Any]:
        previous_report = state["verification_report"]
        feedback = format_revision_feedback(previous_report) if previous_report else None
        answer = synthesizer.synthesize(state["question"], state["documents"], feedback=feedback)
        revisions = state["revision_count"] + (1 if previous_report else 0)
        return {"draft_answer": answer, "revision_count": revisions}

    def verification_node(state: PipelineState) -> dict[str, Any]:
        report = verifier.verify(state["question"], state["draft_answer"], state["documents"])
        return {"verification_report": report}

    def route_after_verification(state: PipelineState) -> Literal["revise", "end"]:
        report = state["verification_report"]
        needs_revision = report is not None and report.needs_revision
        return "revise" if needs_revision and state["revision_count"] < max_revisions else "end"

    graph = StateGraph(PipelineState)
    graph.add_node("scope_check", scope_check_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("verification", verification_node)
    graph.add_edge(START, "scope_check")
    graph.add_conditional_edges(
        "scope_check", route_after_scope_check, {"synthesize": "synthesis", "end": END}
    )
    graph.add_edge("synthesis", "verification")
    graph.add_conditional_edges(
        "verification", route_after_verification, {"revise": "synthesis", "end": END}
    )
    return graph.compile()
