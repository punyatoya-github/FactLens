"""Parsing uploaded files into overlap-aware, size-bounded chunks."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

from factlens.core import constants
from factlens.core.errors import IngestionError
from factlens.core.settings import Settings
from factlens.ingestion.hashing import hash_file, hash_text
from factlens.storage.chunk_cache import ChunkCache

# Bump when chunking behaviour changes so stale cache entries are not reused.
_CHUNKER_VERSION = "1"


_NON_TEXT_PATTERN = re.compile(r"<!--.*?-->|\s+", re.DOTALL)


def _default_converter_factory(ocr: bool) -> Any:
    """Create a Docling converter for PDFs. Imported lazily because Docling is heavy to load.

    OCR is the slowest stage on image-heavy PDFs, so it is only enabled when asked for.
    Table-structure recognition stays on either way.
    """
    # Windows without Developer Mode can't create the symlinks the model cache prefers;
    # caching still works, so the warning is just noise.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(do_ocr=ocr)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def _extracted_text_length(markdown: str) -> int:
    """Count real text characters, ignoring whitespace and Docling's image placeholders."""
    return len(_NON_TEXT_PATTERN.sub("", markdown))


class DocumentIngestor:
    """Validates, parses, chunks and caches uploaded documents."""

    def __init__(
        self,
        settings: Settings,
        cache: ChunkCache | None = None,
        converter_factory: Callable[[bool], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._cache = cache or ChunkCache(settings.cache_dir, settings.cache_expire_days)
        self._converter_factory = converter_factory or _default_converter_factory
        self._converters: dict[bool, Any] = {}
        self._header_splitter = MarkdownHeaderTextSplitter(list(constants.MARKDOWN_HEADERS))
        self._size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_max_chars,
            chunk_overlap=settings.chunk_overlap_chars,
        )
        self._fingerprint = hash_text(
            f"{_CHUNKER_VERSION}|{settings.chunk_max_chars}|{settings.chunk_overlap_chars}"
            f"|{settings.ocr_mode}|{settings.ocr_fallback_min_chars}"
        )[:12]

    def validate_upload_sizes(self, paths: Sequence[Path]) -> None:
        """Raise ``IngestionError`` if any file, or the whole upload, exceeds the size limits."""
        limit_mb = self._settings.max_file_size_bytes // constants.BYTES_PER_MB
        total = 0
        for path in paths:
            size = path.stat().st_size
            if size > self._settings.max_file_size_bytes:
                raise IngestionError(f"{path.name} exceeds the {limit_mb}MB per-file limit.")
            total += size
        if total > self._settings.max_total_size_bytes:
            total_limit_mb = self._settings.max_total_size_bytes // constants.BYTES_PER_MB
            raise IngestionError(f"Total upload size exceeds the {total_limit_mb}MB limit.")

    def ingest_files(self, paths: Sequence[Path]) -> list[Document]:
        """Return unique chunks for ``paths``, using the cache for files seen before."""
        if not paths:
            raise IngestionError("No documents were provided.")
        self.validate_upload_sizes(paths)

        chunks: list[Document] = []
        seen_chunk_hashes: set[str] = set()
        failures: list[str] = []

        for path in paths:
            if path.suffix.lower() not in constants.ALLOWED_EXTENSIONS:
                logger.warning("Skipping unsupported file type: {}", path.name)
                failures.append(f"{path.name} (unsupported type)")
                continue
            try:
                file_chunks = self._load_or_build_chunks(path)
            except Exception as error:  # one bad file must not sink the whole upload
                logger.error("Failed to process {}: {}", path.name, error)
                failures.append(f"{path.name} ({error})")
                continue
            for chunk in file_chunks:
                chunk_hash = hash_text(chunk.page_content)
                if chunk.page_content.strip() and chunk_hash not in seen_chunk_hashes:
                    seen_chunk_hashes.add(chunk_hash)
                    chunks.append(chunk)

        if not chunks:
            detail = "; ".join(failures) if failures else "no extractable text"
            raise IngestionError(f"No content could be extracted: {detail}.")
        logger.info("Ingested {} unique chunks from {} file(s)", len(chunks), len(paths))
        return chunks

    def _load_or_build_chunks(self, path: Path) -> list[Document]:
        file_hash = hash_file(path)
        cache_key = f"{file_hash[:32]}-{self._fingerprint}"
        cached = self._cache.load(cache_key)
        if cached is not None:
            logger.info("Loaded {} from cache", path.name)
            return cached
        logger.info("Parsing {}", path.name)
        chunks = self._parse_and_chunk_file(path, file_hash)
        self._cache.save(cache_key, chunks)
        return chunks

    def _parse_and_chunk_file(self, path: Path, file_hash: str) -> list[Document]:
        markdown = self._parse_to_markdown(path)
        sections = self._header_splitter.split_text(markdown)
        chunks = self._size_splitter.split_documents(sections)
        for chunk in chunks:
            chunk.metadata["source"] = path.name
            chunk.metadata["file_hash"] = file_hash
        return chunks

    def _parse_to_markdown(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in constants.PLAIN_TEXT_EXTENSIONS:
            return path.read_text(encoding="utf-8", errors="replace")
        if suffix != ".pdf" or self._settings.ocr_mode == "never":
            return self._convert(path, ocr=False)
        if self._settings.ocr_mode == "always":
            return self._convert(path, ocr=True)

        markdown = self._convert(path, ocr=False)
        if _extracted_text_length(markdown) < self._settings.ocr_fallback_min_chars:
            logger.info("{} has almost no extractable text; retrying with OCR", path.name)
            markdown = self._convert(path, ocr=True)
        return markdown

    def _convert(self, path: Path, *, ocr: bool) -> str:
        """Run Docling on ``path`` and return Markdown, logging how long it took."""
        if ocr not in self._converters:
            self._converters[ocr] = self._converter_factory(ocr)
        logger.info(
            "Converting {} (OCR {}). Long PDFs can take several minutes on CPU.",
            path.name,
            "on" if ocr else "off",
        )
        started = time.perf_counter()
        result = self._converters[ocr].convert(str(path))
        markdown = str(result.document.export_to_markdown())
        logger.info("Converted {} in {:.0f}s", path.name, time.perf_counter() - started)
        return markdown
