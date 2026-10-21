"""Normalising whatever the UI hands us into plain file paths."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from factlens.core.errors import IngestionError


def resolve_upload_paths(files: Iterable[Any] | None) -> list[Path]:
    """Convert uploaded file references to existing, de-duplicated ``Path`` objects.

    Accepts strings, ``os.PathLike`` objects, or objects exposing a ``name`` path attribute.
    """
    paths: list[Path] = []
    seen: set[Path] = set()
    for item in files or []:
        raw = item if isinstance(item, str | os.PathLike) else getattr(item, "name", None)
        if raw is None:
            raise IngestionError(f"Unsupported upload reference: {item!r}")
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise IngestionError(f"Uploaded file not found: {path.name}")
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths
