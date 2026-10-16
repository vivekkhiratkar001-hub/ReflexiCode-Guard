"""Small end-to-end orchestration for semantic PR review."""

import logging
from time import monotonic

from .context import ReviewContext
from .parser import ReviewResponseError, ReviewResponseParser, deduplicate_findings
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
        max_files_per_batch: int = 3,
    ) -> None:
        if max_files_per_batch < 1:
            raise ValueError("max_files_per_batch must be positive.")
        self.provider = provider
        self.prompt_builder = prompt_builder or ReviewPromptBuilder()
        self.parser = parser or ReviewResponseParser()
        self.last_context_limitations: tuple[str, ...] = ()
        self.last_review_failures: tuple[str, ...] = ()
        self.last_reviewed_files: tuple[str, ...] = ()
        self.last_unreviewed_files: tuple[str, ...] = ()
        self.last_batch_count = 0
        self.last_failed_batch_count = 0
        self.max_files_per_batch = max_files_per_batch

    def review(self, context: ReviewContext) -> list[ReviewFinding]:
        """Review every changed file in bounded requests and preserve successes."""
        paths = self._ranked_paths(context)
        batches = [
            paths[index : index + self.max_files_per_batch]
            for index in range(0, len(paths), self.max_files_per_batch)
        ]
        if not batches:
            batches = [[]]

        findings: list[ReviewFinding] = []
        limitations: list[str] = []
        failures: list[str] = []
        reviewed: list[str] = []
        unreviewed: list[str] = []
        self.last_batch_count = len(batches)

        for batch_number, batch_paths in enumerate(batches, start=1):
            batch_context = self._context_for_files(context, batch_paths)
            path_label = ", ".join(batch_paths) or "(no changed files)"
            preparation_started = monotonic()
            provider_started: float | None = None
            parsing_started: float | None = None
            try:
                built_prompt = self.prompt_builder.build_with_metadata(batch_context)
                limitations.extend(built_prompt.limitations)
                preparation_seconds = monotonic() - preparation_started
                provider_started = monotonic()
                raw_response = self.provider.review(built_prompt.prompt)
                provider_seconds = monotonic() - provider_started
                parsing_started = monotonic()
                batch_findings = self.parser.parse(raw_response, batch_context)
                parsing_seconds = monotonic() - parsing_started
            except (RuntimeError, ReviewResponseError, ValueError) as error:
                logger.warning(
                    "Semantic review batch failed: batch=%d/%d prompt_preparation_seconds=%.3f "
                    "model_wait_seconds=%.3f response_parse_seconds=%.3f exception=%s",
                    batch_number,
                    len(batches),
                    monotonic() - preparation_started,
                    monotonic() - provider_started if provider_started is not None else 0.0,
                    monotonic() - parsing_started if parsing_started is not None else 0.0,
                    type(error).__name__,
                )
                reason = (
                    f"Member 3 rejected the model response: {error}"
                    if isinstance(error, ReviewResponseError)
                    else f"failed ({type(error).__name__}): {error}"
                )
                failures.append(
                    f"Semantic batch {batch_number}/{len(batches)} "
                    f"[{path_label}] {reason}"
                )
                unreviewed.extend(batch_paths)
                continue

            logger.info(
                "Semantic review batch completed: batch=%d/%d prompt_chars=%d "
                "selected_files=%d prompt_preparation_seconds=%.3f model_wait_seconds=%.3f "
                "response_parse_seconds=%.3f findings=%d "
                "files_available=%d files_included=%d "
                "changed_lines_available=%d changed_lines_included=%d "
                "findings_available=%d findings_supplied=%d findings_included=%d "
                "estimated_prompt_tokens=%d",
                batch_number,
                len(batches),
                len(built_prompt.prompt),
                len(built_prompt.included_files),
                preparation_seconds,
                provider_seconds,
                parsing_seconds,
                len(batch_findings),
                len(batch_context.changed_code),
                len(built_prompt.included_files),
                sum(len(set(lines)) for lines in (batch_context.changed_lines or {}).values()),
                built_prompt.included_changed_lines,
                len(batch_context.analysis_findings),
                built_prompt.supplied_findings,
                built_prompt.included_findings,
                (len(built_prompt.prompt) + 3) // 4,
            )

            findings.extend(batch_findings)
            reviewed.extend(built_prompt.included_files)
            unreviewed.extend(
                path for path in batch_paths if path not in built_prompt.included_files
            )
            if not built_prompt.included_files and batch_paths:
                failures.append(
                    f"Semantic batch {batch_number}/{len(batches)} [{path_label}] "
                    "did not include any changed files in its prompt."
                )

        self.last_context_limitations = tuple(dict.fromkeys(limitations))
        self.last_failed_batch_count = len(failures)
        if unreviewed:
            failures.append(
                "Semantic review did not cover these changed files: "
                + ", ".join(dict.fromkeys(unreviewed))
            )
        self.last_review_failures = tuple(failures)
        self.last_reviewed_files = tuple(dict.fromkeys(reviewed))
        self.last_unreviewed_files = tuple(dict.fromkeys(unreviewed))
        return deduplicate_findings(findings)

    @staticmethod
    def _ranked_paths(context: ReviewContext) -> list[str]:
        findings_by_file: dict[str, list[dict]] = {}
        for finding in context.analysis_findings:
            key = str(finding.get("file", "")).replace("\\", "/").casefold()
            findings_by_file.setdefault(key, []).append(finding)

        def priority(path: str) -> tuple[int, int, int, int]:
            key = path.replace("\\", "/").casefold()
            findings = findings_by_file.get(key, [])
            security = int(
                any(str(item.get("category", "")).casefold() == "security" for item in findings)
            )
            severity = max(
                ({"critical": 4, "high": 3, "medium": 2, "low": 1}.get(
                    str(item.get("severity", "")).casefold(), 0
                ) for item in findings),
                default=0,
            )
            production = int(
                not key.startswith(("tests/", "scripts/", "docs/"))
            )
            changed = len(context.changed_lines.get(path, [])) if context.changed_lines else 0
            return security, severity, production, len(findings) * 1000 + changed

        return sorted(context.changed_code, key=priority, reverse=True)

    @staticmethod
    def _context_for_files(context: ReviewContext, paths: list[str]) -> ReviewContext:
        path_keys = {path.replace("\\", "/").casefold() for path in paths}

        def subset(mapping: dict[str, object] | None) -> dict[str, object] | None:
            if mapping is None:
                return None
            return {
                path: value
                for path, value in mapping.items()
                if path.replace("\\", "/").casefold() in path_keys
            }

        findings = [
            finding
            for finding in context.analysis_findings
            if str(finding.get("file", "")).replace("\\", "/").casefold() in path_keys
        ]
        return context.model_copy(
            update={
                "changed_code": subset(context.changed_code) or {},
                "changed_lines": subset(context.changed_lines),
                "changed_line_content": subset(context.changed_line_content) or {},
                "analysis_findings": findings,
            }
        )
