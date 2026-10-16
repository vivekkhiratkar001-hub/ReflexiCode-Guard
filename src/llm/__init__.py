"""LLM providers, review data models, and the shared-contract adapter."""

from .context import ReviewContext, ReviewContextBuilder
from .evaluation import (
    CaseEvaluationResult,
    EvaluationCase,
    EvaluationResult,
    GroundTruthFinding,
    ReviewEvaluator,
    evaluate_predictions,
)
from .mock_provider import MockProvider
from .ollama_provider import OllamaProvider
from .parser import ReviewResponseError, ReviewResponseParser
from .pipeline import ReviewPipeline
from .prompts import ReviewPromptBuilder
from .provider import LLMProvider
from .reviewer import OllamaReviewer
from .schemas import ReviewFinding, ReviewFindings

__all__ = [
    "CaseEvaluationResult",
    "EvaluationCase",
    "EvaluationResult",
    "GroundTruthFinding",
    "LLMProvider",
    "MockProvider",
    "OllamaProvider",
    "OllamaReviewer",
    "ReviewContext",
    "ReviewContextBuilder",
    "ReviewEvaluator",
    "ReviewFinding",
    "ReviewFindings",
    "ReviewPipeline",
    "ReviewPromptBuilder",
    "ReviewResponseError",
    "ReviewResponseParser",
    "evaluate_predictions",
]
