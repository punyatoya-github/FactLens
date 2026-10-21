"""Exception types raised by FactLens."""

from __future__ import annotations


class FactLensError(Exception):
    """Base class for all FactLens errors."""


class ConfigurationError(FactLensError):
    """Raised when required configuration (such as an API key) is missing or invalid."""


class IngestionError(FactLensError):
    """Raised when uploaded documents cannot be validated or parsed."""


class LLMError(FactLensError):
    """Raised when a language-model call fails."""
