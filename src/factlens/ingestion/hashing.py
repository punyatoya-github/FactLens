"""Content hashing helpers used for caching and change detection."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

_READ_BLOCK_BYTES = 1024 * 1024


def hash_bytes(content: bytes) -> str:
    """Return the SHA-256 hex digest of ``content``."""
    return hashlib.sha256(content).hexdigest()


def hash_text(text: str) -> str:
    """Return the SHA-256 hex digest of ``text`` encoded as UTF-8."""
    return hash_bytes(text.encode("utf-8"))


def hash_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, reading it in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_READ_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_file_hashes(paths: Iterable[Path]) -> frozenset[str]:
    """Return the set of content hashes for ``paths`` (order-independent)."""
    return frozenset(hash_file(path) for path in paths)


def combine_hashes(hashes: Iterable[str], *extra: str) -> str:
    """Return one stable digest for a set of hashes plus extra identifying strings."""
    digest = hashlib.sha256()
    for item in sorted(hashes):
        digest.update(item.encode("utf-8"))
    for item in extra:
        digest.update(b"\x00")
        digest.update(item.encode("utf-8"))
    return digest.hexdigest()
