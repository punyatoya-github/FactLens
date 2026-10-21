# Contributing to FactLens

Thanks for helping improve FactLens. This guide covers setup, conventions and how to submit changes.

## Getting started

1. Fork the repository and clone your fork.
2. Create a branch: `feat/<short-description>`, `fix/<short-description>` or `docs/<short-description>`.
3. Install dependencies with `uv sync --extra dev` (or `pip install -e ".[dev]"`).
4. Copy `.env.example` to `.env` and add your `OPENAI_API_KEY`. Never commit `.env`.
5. Optional: `uv run pre-commit install` to run lint, format and type checks on each commit.

## Before opening a pull request

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

All four must pass; CI runs the same commands. Tests must not need network access or an API key:
use the fakes in `tests/conftest.py` (scripted chat models, deterministic embeddings) rather than
calling OpenAI. Don't commit large binaries.

## Conventions

- **Layout:** code lives in `src/factlens/<area>/`. Put new code in the area that owns it
  (ingestion, retrieval, agents, pipeline, ui) rather than a generic `utils` module.
- **Functions and methods:** `snake_case`, starting with a verb (`parse_document`,
  `build_hybrid_retriever`, `verify`). Private helpers start with `_`.
- **Classes:** `PascalCase` nouns named for their role (`ClaimVerifier`, `ChunkCache`).
- **Constants:** `UPPER_SNAKE_CASE`, in `core/constants.py` if shared.
- **Configuration:** add options to `core/settings.py` with a `FACTLENS_` prefix and document them
  in `.env.example` and the README. Don't read `os.environ` elsewhere.
- **Typing and docs:** full type hints (mypy enforces `disallow_untyped_defs`) and a docstring on
  every public class and function.
- **Typed results over strings:** return enums and Pydantic models (`ScopeVerdict`,
  `VerificationReport`), not magic strings or ad-hoc dicts.
- **Logging:** use `loguru`'s `logger`, never `print`.
- **Errors:** raise the types in `core/errors.py`. Never swallow a model failure by returning a
  plausible-looking default.

## Pull requests

Keep PRs small and focused. Describe the summary, motivation, implementation and how you tested
it, and call out any behaviour change or breaking change.

## Reporting issues

Include a short summary, steps to reproduce, expected and actual behaviour, your environment (OS,
Python version, `uv.lock` commit) and relevant logs or a small sample file.

## Security

Never commit credentials or private documents. If you accidentally commit a secret, remove it and
rotate the key immediately.

## License

By contributing you agree that your contributions are licensed under the project's MIT License.
