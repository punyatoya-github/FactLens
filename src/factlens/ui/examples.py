"""Bundled demo documents with a sample question for each."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExampleCase:
    """A sample question together with the bundled files it is meant to be asked about."""

    question: str
    filenames: tuple[str, ...]


EXAMPLE_CASES: dict[str, ExampleCase] = {
    "Google 2024 Environmental Report": ExampleCase(
        question=(
            "Retrieve the data center PUE efficiency values in Singapore 2nd facility in 2019 "
            "and 2022. Also retrieve regional average CFE in Asia pacific in 2023"
        ),
        filenames=("google-2024-environmental-report.pdf",),
    ),
    "DeepSeek-R1 Technical Report": ExampleCase(
        question=(
            "Summarize DeepSeek-R1 model's performance evaluation on all coding tasks against "
            "OpenAI o1-mini model"
        ),
        filenames=("deepseek-r1-technical-report.pdf",),
    ),
}


def resolve_example_files(case: ExampleCase, examples_dir: Path) -> list[str]:
    """Return the existing file paths for ``case`` inside ``examples_dir``."""
    paths = [examples_dir / name for name in case.filenames]
    return [str(path) for path in paths if path.is_file()]
