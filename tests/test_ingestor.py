from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from factlens.core.errors import IngestionError
from factlens.core.settings import Settings
from factlens.ingestion import DocumentIngestor
from factlens.ingestion.ingestor import _extracted_text_length

MARKDOWN = "# Title\nIntro text.\n\n## Results\nPUE was 1.10 in 2022.\n"


class FakeConverter:
    """Stands in for Docling so no models are loaded."""

    def __init__(self, markdown: str = MARKDOWN, fail_on: str | None = None) -> None:
        self.markdown = markdown
        self.fail_on = fail_on
        self.calls: list[str] = []

    def convert(self, path: str) -> Any:
        self.calls.append(path)
        if self.fail_on and self.fail_on in path:
            raise RuntimeError("cannot parse")
        return SimpleNamespace(document=SimpleNamespace(export_to_markdown=lambda: self.markdown))


def _ingestor(settings: Settings, converter: FakeConverter | None = None) -> DocumentIngestor:
    fake = converter or FakeConverter()
    never_ocr = settings.model_copy(update={"ocr_mode": "never"})
    return DocumentIngestor(never_ocr, converter_factory=lambda _ocr: fake)


def test_markdown_is_split_by_header_and_tagged_with_source(
    settings: Settings, tmp_path: Path
) -> None:
    path = tmp_path / "report.md"
    path.write_text(MARKDOWN)
    chunks = _ingestor(settings).ingest_files([path])
    assert [c.page_content for c in chunks] == ["Intro text.", "PUE was 1.10 in 2022."]
    assert all(c.metadata["source"] == "report.md" for c in chunks)
    assert all(len(c.metadata["file_hash"]) == 64 for c in chunks)
    assert chunks[1].metadata["Header 2"] == "Results"


def test_plain_text_files_do_not_load_docling(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("just some notes")
    converter = FakeConverter()
    _ingestor(settings, converter).ingest_files([path])
    assert converter.calls == []


def test_pdf_goes_through_converter_once_then_uses_cache(
    settings: Settings, tmp_path: Path
) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF fake")
    converter = FakeConverter()
    ingestor = _ingestor(settings, converter)
    first = ingestor.ingest_files([path])
    second = ingestor.ingest_files([path])
    assert len(converter.calls) == 1
    assert [c.page_content for c in first] == [c.page_content for c in second]


def test_oversized_sections_are_split_to_the_size_limit(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        openai_api_key="k",
        data_dir=tmp_path / ".factlens",
        chunk_max_chars=300,
        chunk_overlap_chars=30,
    )
    path = tmp_path / "long.md"
    path.write_text("# Big\n" + " ".join(f"word{i}" for i in range(500)))
    chunks = DocumentIngestor(settings).ingest_files([path])
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 300 for c in chunks)
    assert all(c.metadata["Header 1"] == "Big" for c in chunks)


def test_identical_content_across_files_is_deduplicated(settings: Settings, tmp_path: Path) -> None:
    first, second = tmp_path / "a.md", tmp_path / "b.md"
    first.write_text(MARKDOWN)
    second.write_text(MARKDOWN)
    assert len(_ingestor(settings).ingest_files([first, second])) == 2


def test_unsupported_file_type_is_reported(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a,b")
    with pytest.raises(IngestionError, match="unsupported type"):
        _ingestor(settings).ingest_files([path])


def test_one_bad_file_does_not_sink_the_upload(settings: Settings, tmp_path: Path) -> None:
    good, bad = tmp_path / "good.md", tmp_path / "bad.pdf"
    good.write_text(MARKDOWN)
    bad.write_bytes(b"%PDF broken")
    chunks = _ingestor(settings, FakeConverter(fail_on="bad.pdf")).ingest_files([good, bad])
    assert {c.metadata["source"] for c in chunks} == {"good.md"}


def test_all_files_failing_raises_with_reasons(settings: Settings, tmp_path: Path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF broken")
    with pytest.raises(IngestionError, match="cannot parse"):
        _ingestor(settings, FakeConverter(fail_on="bad.pdf")).ingest_files([bad])


def test_empty_upload_raises(settings: Settings) -> None:
    with pytest.raises(IngestionError, match="No documents"):
        _ingestor(settings).ingest_files([])


def test_per_file_and_total_size_limits(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        openai_api_key="k",
        data_dir=tmp_path / ".factlens",
        max_file_size_bytes=100,
        max_total_size_bytes=150,
    )
    big, small_a, small_b = tmp_path / "big.md", tmp_path / "a.md", tmp_path / "b.md"
    big.write_text("x" * 101)
    small_a.write_text("y" * 80)
    small_b.write_text("z" * 80)
    ingestor = DocumentIngestor(settings)
    with pytest.raises(IngestionError, match="per-file limit"):
        ingestor.validate_upload_sizes([big])
    with pytest.raises(IngestionError, match="Total upload size"):
        ingestor.validate_upload_sizes([small_a, small_b])
    ingestor.validate_upload_sizes([small_a])


def test_changing_chunk_settings_invalidates_the_cache(tmp_path: Path) -> None:
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF fake")
    converter = FakeConverter()

    def build(max_chars: int) -> DocumentIngestor:
        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            openai_api_key="k",
            data_dir=tmp_path / ".factlens",
            chunk_max_chars=max_chars,
            ocr_mode="never",
        )
        return DocumentIngestor(settings, converter_factory=lambda _ocr: converter)

    build(2000).ingest_files([path])
    build(500).ingest_files([path])
    assert len(converter.calls) == 2


# --- OCR modes ---------------------------------------------------------------------------
RICH_MARKDOWN = "# Report\n" + "A real extracted sentence about energy use. " * 20
SCANNED_MARKDOWN = "<!-- image -->\n\n<!-- image -->"


def _ocr_ingestor(
    settings: Settings, mode: str, markdown_by_ocr: dict[bool, str]
) -> tuple[DocumentIngestor, dict[bool, FakeConverter], list[bool]]:
    converters = {ocr: FakeConverter(markdown) for ocr, markdown in markdown_by_ocr.items()}
    created: list[bool] = []

    def factory(ocr: bool) -> FakeConverter:
        created.append(ocr)
        return converters[ocr]

    configured = settings.model_copy(update={"ocr_mode": mode})
    return DocumentIngestor(configured, converter_factory=factory), converters, created


def _pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
    path = tmp_path / name
    path.write_bytes(b"%PDF fake")
    return path


def test_extracted_text_length_ignores_whitespace_and_image_placeholders() -> None:
    assert _extracted_text_length(SCANNED_MARKDOWN) == 0
    assert _extracted_text_length("<!-- image -->\n abc  def \n") == 6


def test_auto_mode_skips_ocr_when_the_pdf_has_text(settings: Settings, tmp_path: Path) -> None:
    ingestor, converters, created = _ocr_ingestor(
        settings, "auto", {False: RICH_MARKDOWN, True: MARKDOWN}
    )
    ingestor.ingest_files([_pdf(tmp_path)])
    assert created == [False]
    assert converters[False].calls and not converters[True].calls


def test_auto_mode_retries_with_ocr_for_scanned_pdfs(settings: Settings, tmp_path: Path) -> None:
    ingestor, converters, created = _ocr_ingestor(
        settings, "auto", {False: SCANNED_MARKDOWN, True: MARKDOWN}
    )
    chunks = ingestor.ingest_files([_pdf(tmp_path)])
    assert created == [False, True]
    assert len(converters[False].calls) == 1 and len(converters[True].calls) == 1
    assert [c.page_content for c in chunks] == ["Intro text.", "PUE was 1.10 in 2022."]


def test_never_mode_does_not_ocr_even_without_text(settings: Settings, tmp_path: Path) -> None:
    ingestor, converters, created = _ocr_ingestor(
        settings, "never", {False: "# Tiny\nstill some text", True: MARKDOWN}
    )
    ingestor.ingest_files([_pdf(tmp_path)])
    assert created == [False]
    assert not converters[True].calls


def test_always_mode_ocrs_pdfs_directly(settings: Settings, tmp_path: Path) -> None:
    ingestor, converters, created = _ocr_ingestor(
        settings, "always", {False: RICH_MARKDOWN, True: RICH_MARKDOWN}
    )
    ingestor.ingest_files([_pdf(tmp_path)])
    assert created == [True]
    assert not converters[False].calls


def test_ocr_never_applies_to_docx(settings: Settings, tmp_path: Path) -> None:
    ingestor, _, created = _ocr_ingestor(settings, "always", {False: RICH_MARKDOWN})
    path = tmp_path / "report.docx"
    path.write_bytes(b"PK fake docx")
    ingestor.ingest_files([path])
    assert created == [False]


def test_each_converter_is_created_once_and_reused(settings: Settings, tmp_path: Path) -> None:
    ingestor, _, created = _ocr_ingestor(settings, "never", {False: RICH_MARKDOWN})
    first = _pdf(tmp_path, "a.pdf")
    second = tmp_path / "b.pdf"
    second.write_bytes(b"%PDF different bytes")
    ingestor.ingest_files([first, second])
    assert created == [False]


def test_changing_ocr_mode_invalidates_the_cache(settings: Settings, tmp_path: Path) -> None:
    path = _pdf(tmp_path)
    first, _, created_first = _ocr_ingestor(settings, "never", {False: RICH_MARKDOWN})
    second, _, created_second = _ocr_ingestor(
        settings, "always", {True: RICH_MARKDOWN, False: RICH_MARKDOWN}
    )
    first.ingest_files([path])
    second.ingest_files([path])
    assert created_first == [False]
    assert created_second == [True]  # not served from the "never" cache entry
