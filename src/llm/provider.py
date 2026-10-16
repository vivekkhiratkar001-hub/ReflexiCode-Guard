"""Base interface for language-model review providers."""

from abc import ABC, abstractmethod

from .context import ReviewContext


class LLMProvider(ABC):
	"""Common interface implemented by code-review language models."""

	@abstractmethod
	def review(self, context: ReviewContext | dict | str) -> str:
		"""Return raw model output for a context object or ready-built prompt."""
		raise NotImplementedError
