"""Static values shared across FactLens."""

from __future__ import annotations

BYTES_PER_MB = 1024 * 1024

# Hard limits for uploads (overridable through Settings).
DEFAULT_MAX_FILE_SIZE_BYTES = 50 * BYTES_PER_MB
DEFAULT_MAX_TOTAL_SIZE_BYTES = 200 * BYTES_PER_MB

# File types that can be ingested.
ALLOWED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".txt", ".md")

# Extensions read as plain text instead of going through Docling.
PLAIN_TEXT_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md"})

# Markdown headers used to split parsed documents into sections.
MARKDOWN_HEADERS: tuple[tuple[str, str], ...] = (("#", "Header 1"), ("##", "Header 2"))

# Message returned when a question cannot be answered from the uploaded documents.
OUT_OF_SCOPE_MESSAGE = (
    "This question isn't related to the uploaded document(s), or they don't contain the "
    "data needed to answer it. Please ask a question relevant to your documents."
)

# Message used when the synthesizer cannot produce an answer.
NO_ANSWER_MESSAGE = "I cannot answer this question based on the provided documents."
