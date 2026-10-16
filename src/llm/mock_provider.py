"""Configurable provider for tests that do not need a running Ollama server."""

from .context import ReviewContext
from .provider import LLMProvider


class MockProvider(LLMProvider):
    """Return a configured response and retain the last submitted prompt."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_input: str | ReviewContext | dict | None = None
        self.inputs: list[str | ReviewContext | dict] = []

    def review(self, context: ReviewContext | dict | str) -> str:
        self.last_input = context
        self.inputs.append(context)
        return self.response