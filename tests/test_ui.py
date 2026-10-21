from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from factlens.core.settings import Settings
from factlens.ingestion import DocumentIngestor
from factlens.schemas import PipelineResult, ScopeVerdict, VerificationReport
from factlens.ui import app as app_module
from factlens.ui.app import (
    AppServices,
    build_interface,
    handle_question_submission,
    load_example_into_inputs,
)
from factlens.ui.examples import EXAMPLE_CASES
from factlens.ui.rendering import render_scope_note, render_verification_report
from tests.conftest import CountingEmbeddings


class StubPipeline:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def run(self, question: str, retriever: Any) -> PipelineResult:
        self.questions.append(question)
        retriever.invoke(question)
        return PipelineResult(
            answer="The answer.",
            scope_verdict=ScopeVerdict.ANSWERABLE,
            verification_report=VerificationReport(supported=True, relevant=True),
            revisions_used=1,
        )


class SpyIngestor(DocumentIngestor):
    ingest_calls = 0

    def ingest_files(self, paths: Any) -> Any:
        self.ingest_calls += 1
        return super().ingest_files(paths)


@pytest.fixture
def services(
    settings: Settings, embeddings: CountingEmbeddings, monkeypatch: pytest.MonkeyPatch
) -> AppServices:
    monkeypatch.setattr(app_module, "build_embeddings", lambda _settings: embeddings)
    return AppServices(
        settings=settings,
        ingestor=SpyIngestor(settings),
        pipeline=StubPipeline(),  # type: ignore[arg-type]
    )


def _run(
    question: str, files: Any, state: dict[str, Any], services: AppServices
) -> tuple[str, str, dict[str, Any]]:
    """Drive the streaming handler to completion and return its final update."""
    *_, final = handle_question_submission(question, files, state, services)
    return final


def _state() -> dict[str, Any]:
    return {"documents_key": None, "retriever": None}


def test_render_verification_report_formats_all_fields() -> None:
    report = VerificationReport(
        supported=False,
        relevant=True,
        unsupported_claims=["x", "y"],
        additional_details="check x",
    )
    text = render_verification_report(report)
    assert "**Supported:** NO" in text
    assert "**Unsupported Claims:** x, y" in text
    assert "**Contradictions:** None" in text
    assert "**Relevant:** YES" in text
    assert "check x" in text


def test_render_verification_report_without_a_report() -> None:
    assert "No verification" in render_verification_report(None)


def test_render_scope_note_mentions_revisions() -> None:
    assert render_scope_note(ScopeVerdict.PARTIAL, 0) == "Scope: PARTIAL"
    assert "1 revision after" in render_scope_note(ScopeVerdict.ANSWERABLE, 1)
    assert "2 revisions after" in render_scope_note(ScopeVerdict.ANSWERABLE, 2)


def test_load_example_returns_files_and_question(services: AppServices) -> None:
    name, case = next(iter(EXAMPLE_CASES.items()))
    examples_dir = services.settings.examples_dir
    examples_dir.mkdir(parents=True)
    for filename in case.filenames:
        (examples_dir / filename).write_bytes(b"%PDF")
    files, question = load_example_into_inputs(name, services)
    assert question == case.question
    assert [Path(f).name for f in files] == list(case.filenames)


def test_load_example_with_no_selection_is_empty(services: AppServices) -> None:
    assert load_example_into_inputs(None, services) == ([], "")


def test_empty_question_and_missing_files_give_readable_errors(
    services: AppServices, tmp_path: Path
) -> None:
    answer, report, _ = _run("  ", [], _state(), services)
    assert "Question cannot be empty" in answer and report == ""
    answer, _, _ = _run("What?", [], _state(), services)
    assert "No documents uploaded" in answer


def test_successful_submission_renders_answer_and_report(
    services: AppServices, tmp_path: Path
) -> None:
    document = tmp_path / "doc.md"
    document.write_text("# Facts\nThe sky is blue.\n")
    answer, report, state = _run("What colour is the sky?", [str(document)], _state(), services)
    assert answer == "The answer."
    assert "**Supported:** YES" in report
    assert "Scope: CAN_ANSWER" in report and "1 revision" in report
    assert state["retriever"] is not None


def test_same_files_reuse_the_retriever_and_changed_files_rebuild(
    services: AppServices, tmp_path: Path
) -> None:
    document = tmp_path / "doc.md"
    document.write_text("# Facts\nThe sky is blue.\n")
    ingestor = services.ingestor
    assert isinstance(ingestor, SpyIngestor)

    _, _, state = _run("q1", [str(document)], _state(), services)
    _, _, state = _run("q2", [str(document)], state, services)
    assert ingestor.ingest_calls == 1

    document.write_text("# Facts\nThe grass is green.\n")
    _run("q3", [str(document)], state, services)
    assert ingestor.ingest_calls == 2


def test_ingestion_problems_are_reported_not_raised(services: AppServices, tmp_path: Path) -> None:
    unsupported = tmp_path / "data.csv"
    unsupported.write_text("a,b")
    answer, report, _ = _run("q?", [str(unsupported)], _state(), services)
    assert answer.startswith("**Error:**") and "unsupported type" in answer
    assert report == ""


def test_interface_builds_with_the_expected_components(services: AppServices) -> None:
    import gradio as gr

    demo = build_interface(services)
    kinds = {type(block).__name__ for block in demo.blocks.values()}
    assert {"Dropdown", "Files", "Textbox", "Button", "Markdown"} <= kinds
    assert isinstance(demo, gr.Blocks)


def test_first_submission_streams_progress_then_the_final_answer(
    services: AppServices, tmp_path: Path
) -> None:
    document = tmp_path / "doc.md"
    document.write_text("# Facts\nThe sky is blue.\n")
    updates = list(handle_question_submission("q?", [str(document)], _state(), services))
    messages = [answer for answer, _, _ in updates]
    assert "Reading your documents" in messages[0]
    assert "Indexing" in messages[1]
    assert "verifying" in messages[2]
    assert messages[-1] == "The answer."
    assert len(updates) == 4


def test_repeat_submission_skips_the_document_reading_status(
    services: AppServices, tmp_path: Path
) -> None:
    document = tmp_path / "doc.md"
    document.write_text("# Facts\nThe sky is blue.\n")
    *_, (_, _, state) = handle_question_submission("q1", [str(document)], _state(), services)
    updates = list(handle_question_submission("q2", [str(document)], state, services))
    assert not any("Reading your documents" in answer for answer, _, _ in updates)
    assert updates[-1][0] == "The answer."
