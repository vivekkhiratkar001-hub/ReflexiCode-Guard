"""Small end-to-end orchestration for semantic PR review."""

from .context import ReviewContext
from .parser import ReviewResponseParser
from .prompts import ReviewPromptBuilder
from .provider import LLMProvider
from .schemas import ReviewFinding


class ReviewPipeline:
    """Connect context, prompt, provider, and validated review findings."""

    def __init__(
        self,
        provider: LLMProvider,
        prompt_builder: ReviewPromptBuilder | None = None,
        parser: ReviewResponseParser | None = None,
    ) -> None:
        self.provider = provider
        self.prompt_builder = prompt_builder or ReviewPromptBuilder()
        self.parser = parser or ReviewResponseParser()

    def review(self, context: ReviewContext) -> list[ReviewFinding]:
        """Run a review and return only schema- and context-validated findings."""
        prompt = self.prompt_builder.build(context)
        raw_response = self.provider.review(prompt)
        return self.parser.parse(raw_response, context)