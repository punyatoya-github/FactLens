from __future__ import annotations

import os
import time
from pathlib import Path

from langchain_core.documents import Document

from factlens.storage.chunk_cache import ChunkCache


def _chunks() -> list[Document]:
    return [
        Document(page_content="alpha", metadata={"source": "a.md", "Header 1": "Intro"}),
        Document(page_content="beta ünïcode", metadata={"source": "a.md"}),
    ]


def test_round_trip_preserves_content_and_metadata(tmp_path: Path) -> None:
    cache = ChunkCache(tmp_path, expire_days=7)
    cache.save("key", _chunks())
    loaded = cache.load("key")
    assert loaded is not None
    assert [(c.page_content, c.metadata) for c in loaded] == [
        (c.page_content, c.metadata) for c in _chunks()
    ]


def test_missing_entry_returns_none(tmp_path: Path) -> None:
    assert ChunkCache(tmp_path, expire_days=7).load("nope") is None


def test_expired_entry_is_ignored(tmp_path: Path) -> None:
    cache = ChunkCache(tmp_path, expire_days=7)
    cache.save("key", _chunks())
    old = time.time() - 8 * 24 * 3600
    os.utime(cache.path_for("key"), (old, old))
    assert not cache.is_fresh("key")
    assert cache.load("key") is None


def test_corrupt_entry_is_ignored_not_raised(tmp_path: Path) -> None:
    cache = ChunkCache(tmp_path, expire_days=7)
    cache.path_for("key").write_text("{not json", encoding="utf-8")
    assert cache.load("key") is None


def test_wrong_schema_version_is_ignored(tmp_path: Path) -> None:
    cache = ChunkCache(tmp_path, expire_days=7)
    cache.path_for("key").write_text('{"schema_version": 999, "chunks": []}', encoding="utf-8")
    assert cache.load("key") is None


def test_save_leaves_no_temp_files(tmp_path: Path) -> None:
    cache = ChunkCache(tmp_path, expire_days=7)
    cache.save("key", _chunks())
    assert [p.name for p in tmp_path.iterdir()] == ["key.json"]
