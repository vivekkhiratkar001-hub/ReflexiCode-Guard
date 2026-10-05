"""Base interface for language-model review providers."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
	"""Common interface implemented by code-review language models."""

	@abstractmethod
	def review(self, context: dict) -> str:
		"""Review the supplied pull-request context and return feedback."""
		raise NotImplementedError
