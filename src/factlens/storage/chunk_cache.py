"""On-disk cache of parsed document chunks.

Chunks are stored as JSON (never pickle), so a cache file can't execute code when loaded.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from langchain_core.documents import Document
from loguru import logger

_SCHEMA_VERSION = 1


class ChunkCache:
    """Stores and retrieves lists of ``Document`` chunks keyed by a content-derived string."""

    def __init__(self, directory: Path, expire_days: int) -> None:
        self._directory = directory
        self._expire_after = timedelta(days=expire_days)
        self._directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        """Return the cache file path for ``key``."""
        return self._directory / f"{key}.json"

    def is_fresh(self, key: str) -> bool:
        """Whether a cache entry exists and has not expired."""
        path = self.path_for(key)
        if not path.exists():
            return False
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        return datetime.now(tz=UTC) - modified < self._expire_after

    def load(self, key: str) -> list[Document] | None:
        """Return cached chunks, or ``None`` if the entry is missing, stale or unreadable."""
        if not self.is_fresh(key):
            return None
        try:
            payload = json.loads(self.path_for(key).read_text(encoding="utf-8"))
            if payload.get("schema_version") != _SCHEMA_VERSION:
                return None
            return [
                Document(page_content=item["page_content"], metadata=item["metadata"])
                for item in payload["chunks"]
            ]
        except (OSError, ValueError, KeyError, TypeError) as error:
            logger.warning("Ignoring unreadable cache entry {}: {}", key, error)
            return None

    def save(self, key: str, chunks: list[Document]) -> None:
        """Write chunks to the cache atomically."""
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "created_at": datetime.now(tz=UTC).isoformat(),
            "chunks": [
                {"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks
            ],
        }
        descriptor, temp_name = tempfile.mkstemp(dir=self._directory, suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            Path(temp_name).replace(self.path_for(key))
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise
