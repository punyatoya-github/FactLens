from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from factlens.core.errors import IngestionError
from factlens.ingestion.hashing import combine_hashes, compute_file_hashes, hash_bytes, hash_file
from factlens.ingestion.uploads import resolve_upload_paths


def test_hash_bytes_matches_sha256() -> None:
    assert hash_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_hash_file_matches_hash_bytes(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_bytes(b"hello world" * 1000)
    assert hash_file(path) == hash_bytes(b"hello world" * 1000)


def test_compute_file_hashes_ignores_order_and_detects_changes(tmp_path: Path) -> None:
    first, second = tmp_path / "1.txt", tmp_path / "2.txt"
    first.write_text("one")
    second.write_text("two")
    assert compute_file_hashes([first, second]) == compute_file_hashes([second, first])
    second.write_text("changed")
    assert compute_file_hashes([first, second]) != compute_file_hashes([first, first])


def test_combine_hashes_is_order_independent_and_sensitive_to_extras() -> None:
    assert combine_hashes(["a", "b"]) == combine_hashes(["b", "a"])
    assert combine_hashes(["a", "b"], "model-1") != combine_hashes(["a", "b"], "model-2")


def test_resolve_upload_paths_accepts_strings_paths_and_objects(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("x")
    resolved = resolve_upload_paths([str(path), path, SimpleNamespace(name=str(path))])
    assert resolved == [path.resolve()]


def test_resolve_upload_paths_handles_none() -> None:
    assert resolve_upload_paths(None) == []


def test_resolve_upload_paths_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="not found"):
        resolve_upload_paths([tmp_path / "missing.pdf"])


def test_resolve_upload_paths_rejects_unknown_reference_type() -> None:
    with pytest.raises(IngestionError, match="Unsupported upload reference"):
        resolve_upload_paths([42])
