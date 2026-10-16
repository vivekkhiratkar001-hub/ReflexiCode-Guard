"""Small end-to-end orchestration for semantic PR review."""

import logging

from .context import ReviewContext
from .parser import ReviewResponseParser
from .prompts import ReviewPromptBuilder
from .provider import LLMProvider
from .schemas import ReviewFinding


logger = logging.getLogger(__name__)


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
        self.last_context_limitations: tuple[str, ...] = ()

    def review(self, context: ReviewContext) -> list[ReviewFinding]:
        """Run a review and return only schema- and context-validated findings."""
        built_prompt = self.prompt_builder.build_with_metadata(context)
        self.last_context_limitations = built_prompt.limitations
        prompt = built_prompt.prompt
        logger.info(
            "Semantic review context prepared: files_available=%d files_included=%d "
            "changed_lines_available=%d changed_lines_included=%d findings_available=%d "
            "findings_supplied=%d findings_included=%d prompt_chars=%d "
            "estimated_prompt_tokens=%d",
            built_prompt.available_files,
            len(built_prompt.included_files),
            built_prompt.available_changed_lines,
            built_prompt.included_changed_lines,
            built_prompt.available_findings,
            built_prompt.supplied_findings,
            built_prompt.included_findings,
            len(prompt),
            (len(prompt) + 3) // 4,
        )
        raw_response = self.provider.review(prompt)
        return self.parser.parse(raw_response, context)