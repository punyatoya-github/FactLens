"""The three agents of the answer pipeline."""

from factlens.agents.answer_synthesizer import AnswerSynthesizer
from factlens.agents.claim_verifier import ClaimVerifier
from factlens.agents.scope_classifier import ScopeClassifier

__all__ = ["AnswerSynthesizer", "ClaimVerifier", "ScopeClassifier"]
