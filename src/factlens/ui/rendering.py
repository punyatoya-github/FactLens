"""Turning pipeline results into markdown for display."""

from __future__ import annotations

from factlens.schemas.verdicts import ScopeVerdict, VerificationReport


def _yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def render_verification_report(report: VerificationReport | None) -> str:
    """Format a verification report as markdown."""
    if report is None:
        return "_No verification was run for this question._"
    unsupported = ", ".join(report.unsupported_claims) or "None"
    contradictions = ", ".join(report.contradictions) or "None"
    details = report.additional_details or "None"
    return "\n".join(
        [
            f"**Supported:** {_yes_no(report.supported)}",
            f"**Unsupported Claims:** {unsupported}",
            f"**Contradictions:** {contradictions}",
            f"**Relevant:** {_yes_no(report.relevant)}",
            f"**Additional Details:** {details}",
        ]
    )


def render_scope_note(verdict: ScopeVerdict, revisions_used: int) -> str:
    """One-line summary of how the pipeline handled the question."""
    parts = [f"Scope: {verdict.value}"]
    if revisions_used:
        plural = "revision" if revisions_used == 1 else "revisions"
        parts.append(f"{revisions_used} {plural} after verification")
    return " · ".join(parts)
