"""Show how FactLens parses and chunks a file, without calling any model.

Usage:
    python scripts/inspect_parsing.py path/to/document.pdf [--preview 3]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from factlens.core.settings import Settings
from factlens.ingestion import DocumentIngestor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="A .pdf, .docx, .txt or .md file")
    parser.add_argument("--preview", type=int, default=3, help="How many chunks to print")
    args = parser.parse_args()

    settings = Settings(openai_api_key="unused")
    chunks = DocumentIngestor(settings).ingest_files([args.path.resolve()])

    lengths = [len(chunk.page_content) for chunk in chunks]
    print(f"{len(chunks)} chunks, {min(lengths)}-{max(lengths)} characters each")
    for index, chunk in enumerate(chunks[: args.preview], start=1):
        print(f"\n--- chunk {index} | {chunk.metadata}")
        print(chunk.page_content[:500])


if __name__ == "__main__":
    main()
