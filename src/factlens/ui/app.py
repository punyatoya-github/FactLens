"""Gradio application: upload documents, ask questions, inspect the verification report."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import gradio as gr
from loguru import logger  # pyright: ignore[reportMissingImports]

from factlens.core.constants import ALLOWED_EXTENSIONS
from factlens.core.errors import ConfigurationError, FactLensError
from factlens.core.logging import configure_logging
from factlens.core.settings import Settings, get_settings
from factlens.ingestion import DocumentIngestor
from factlens.ingestion.hashing import combine_hashes, compute_file_hashes
from factlens.ingestion.uploads import resolve_upload_paths
from factlens.llm.clients import build_embeddings
from factlens.pipeline import AnswerPipeline
from factlens.retrieval import build_hybrid_retriever
from factlens.ui.examples import EXAMPLE_CASES, resolve_example_files
from factlens.ui.rendering import render_scope_note, render_verification_report

APP_TITLE = "FactLens"
APP_TAGLINE = (
    "An advanced RAG-based document analysis assistant focused on explainable, verifiable answers."
)


@dataclass
class AppServices:
    """The long-lived objects the UI handlers depend on."""

    settings: Settings
    ingestor: DocumentIngestor
    pipeline: AnswerPipeline

    @classmethod
    def from_settings(cls, settings: Settings) -> AppServices:
        return cls(
            settings=settings,
            ingestor=DocumentIngestor(settings),
            pipeline=AnswerPipeline.from_settings(settings),
        )


def _new_session_state() -> dict[str, Any]:
    return {"documents_key": None, "retriever": None}


def load_example_into_inputs(
    example_name: str | None, services: AppServices
) -> tuple[list[str], str]:
    """Return the files and question for a bundled example (empty if none is selected)."""
    case = EXAMPLE_CASES.get(example_name or "")
    if case is None:
        return [], ""
    files = resolve_example_files(case, services.settings.examples_dir)
    if not files:
        logger.warning("Example files not found in {}", services.settings.examples_dir)
    return files, case.question


def _status(message: str, state: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """An in-progress UI update: a status line in the answer pane and an empty report."""
    return f"_{message}_", "", state


def handle_question_submission(
    question: str,
    uploaded_files: list[Any] | None,
    state: dict[str, Any],
    services: AppServices,
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Answer ``question`` about the uploaded files, yielding status updates along the way.

    The retriever is rebuilt only when the files changed. The last item yielded is the final
    ``(answer, verification report, session state)``.
    """
    try:
        if not question or not question.strip():
            raise FactLensError("Question cannot be empty.")
        paths = resolve_upload_paths(uploaded_files)
        if not paths:
            raise FactLensError("No documents uploaded.")

        documents_key = combine_hashes(
            compute_file_hashes(paths), services.settings.embedding_model
        )
        if state.get("retriever") is None or state.get("documents_key") != documents_key:
            logger.info("Processing new or changed documents")
            yield _status(
                "Reading your documents. The first time this can take several minutes for long "
                "PDFs; the result is cached, so asking again about the same files is fast.",
                state,
            )
            chunks = services.ingestor.ingest_files(paths)
            yield _status(f"Indexing {len(chunks)} passages for search…", state)
            retriever = build_hybrid_retriever(
                chunks, settings=services.settings, embeddings=build_embeddings(services.settings)
            )
            state = {"documents_key": documents_key, "retriever": retriever}

        yield _status("Finding relevant passages, drafting an answer and verifying it…", state)
        result = services.pipeline.run(question.strip(), state["retriever"])
        note = render_scope_note(result.scope_verdict, result.revisions_used)
        report = f"{render_verification_report(result.verification_report)}\n\n_{note}_"
        yield result.answer, report, state
    except FactLensError as error:
        logger.error("Request failed: {}", error)
        yield f"**Error:** {error}", "", state
    except Exception as error:  # last-resort guard so the UI never shows a raw traceback
        logger.exception("Unexpected error while answering")
        yield f"**Unexpected error:** {error}", "", state


def build_interface(services: AppServices) -> gr.Blocks:
    """Create the Gradio layout and wire up its event handlers."""
    demo = gr.Blocks(title=APP_TITLE)
    with demo:
        gr.Markdown(f"# {APP_TITLE}\n{APP_TAGLINE}")
        gr.Markdown(
            "Upload one or more documents (.pdf, .docx, .txt, .md), ask a question, "
            "and review the answer alongside its verification report. "
            "Or load a bundled example and press Submit."
        )
        session_state = gr.State(_new_session_state())

        with gr.Row():
            with gr.Column():
                example_dropdown = gr.Dropdown(
                    label="Example", choices=list(EXAMPLE_CASES), value=None
                )
                load_example_button = gr.Button("Load example")
                files = gr.Files(label="Documents", file_types=list(ALLOWED_EXTENSIONS))
                question = gr.Textbox(label="Question", lines=3)
                submit_button = gr.Button("Submit", variant="primary")
            with gr.Column():
                answer_output = gr.Markdown(label="Answer")
                verification_output = gr.Markdown(label="Verification report")

        load_example_button.click(
            fn=lambda name: load_example_into_inputs(name, services),
            inputs=[example_dropdown],
            outputs=[files, question],
        )

        def submit(
            question_text: str, uploaded: list[Any] | None, state: dict[str, Any]
        ) -> Iterator[tuple[str, str, dict[str, Any]]]:
            yield from handle_question_submission(question_text, uploaded, state, services)

        submit_button.click(
            fn=submit,
            inputs=[question, files, session_state],
            outputs=[answer_output, verification_output, session_state],
        )
    return demo


def main() -> None:
    """Start the FactLens web app."""
    settings = get_settings()
    configure_logging(settings)
    try:
        settings.require_openai_api_key()
        services = AppServices.from_settings(settings)
    except ConfigurationError as error:
        sys.exit(f"Configuration error: {error}")

    build_interface(services).launch(
        server_name=settings.server_host,
        server_port=settings.server_port,
        share=settings.share,
    )


if __name__ == "__main__":
    main()
