"""Small end-to-end orchestration for semantic PR review."""

import logging
from math import isfinite
from time import monotonic

from .context import ReviewContext
from .parser import ReviewResponseError, ReviewResponseParser, deduplicate_findings
from .prompts import ReviewPromptBuilder
from .provider import LLMProvider
from .schemas import ReviewFinding

logger = logging.getLogger(__name__)

DEFAULT_REVIEW_BUDGET_SECONDS = 240.0
MAX_REVIEW_REQUESTS = 5
MIN_REVIEW_REQUEST_BUDGET_SECONDS = 1.0
REQUEST_TIMEOUT_SAFETY_MARGIN_SECONDS = 5.0


class ReviewPipeline:
    """Connect context, prompt, provider, and validated review findings."""

    def __init__(
        self,
        provider: LLMProvider,
        prompt_builder: ReviewPromptBuilder | None = None,
        parser: ReviewResponseParser | None = None,
        *,
        review_budget_seconds: float = DEFAULT_REVIEW_BUDGET_SECONDS,
    ) -> None:
        if (
            isinstance(review_budget_seconds, bool)
            or not isinstance(review_budget_seconds, (int, float))
            or not isfinite(review_budget_seconds)
            or review_budget_seconds <= 0
        ):
            raise ValueError("review_budget_seconds must be positive.")
        self.provider = provider
        self.prompt_builder = prompt_builder or ReviewPromptBuilder()
        self.parser = parser or ReviewResponseParser(
            max_findings=self.prompt_builder.max_findings
        )
        self.last_context_limitations: tuple[str, ...] = ()
        self.last_review_failures: tuple[str, ...] = ()
        self.last_reviewed_files: tuple[str, ...] = ()
        self.last_partially_reviewed_files: tuple[str, ...] = ()
        self.last_unreviewed_files: tuple[str, ...] = ()
        self.last_batch_count = 0
        self.last_failed_batch_count = 0
        self.last_findings_count = 0
        self.last_elapsed_seconds = 0.0
        self.review_budget_seconds = float(review_budget_seconds)

    def review(self, context: ReviewContext) -> list[ReviewFinding]:
        """Review prioritized batches within one aggregate wall-clock budget."""
        started = monotonic()
        deadline = started + self.review_budget_seconds
        paths = self._ranked_paths(context)
        if not paths:
            self._set_result((), (), (), (), (), 0, 0, 0, monotonic() - started)
            return []

        limitations: list[str] = []
        failures: list[str] = []
        reviewed: set[str] = set()
        partially_reviewed: set[str] = set()
        findings: list[ReviewFinding] = []
        attempted = 0
        failed = 0
        batch_size = self.prompt_builder.max_files
        batches = [
            paths[index : index + batch_size]
            for index in range(0, len(paths), batch_size)
        ]

        for batch_index, batch_paths in enumerate(
            batches[:MAX_REVIEW_REQUESTS], start=1
        ):
            if attempted >= MAX_REVIEW_REQUESTS:
                break
            if deadline - monotonic() < (
                MIN_REVIEW_REQUEST_BUDGET_SECONDS
                + REQUEST_TIMEOUT_SAFETY_MARGIN_SECONDS
            ):
                failures.append(
                    f"Semantic review budget exhausted before batch {batch_index}; "
                    "no additional model request was started."
                )
                break

            preparation_started = monotonic()
            provider_started: float | None = None
            parsing_started: float | None = None
            try:
                batch_context = self._context_for_files(context, batch_paths)
                built_prompt = self.prompt_builder.build_with_metadata(batch_context)
                limitations.extend(built_prompt.limitations)
                remaining = deadline - monotonic()
                if remaining < (
                    MIN_REVIEW_REQUEST_BUDGET_SECONDS
                    + REQUEST_TIMEOUT_SAFETY_MARGIN_SECONDS
                ):
                    raise RuntimeError(
                        "review budget exhausted during prompt preparation; "
                        "no model request was started"
                    )

                included = list(built_prompt.included_files)
                if not included:
                    raise RuntimeError(
                        "No changed source lines fit in the semantic-review prompt."
                    )
                eligible_context = self._context_for_files(batch_context, included)
                eligible_lines = built_prompt.eligible_changed_lines or {}
                eligible_keys = {
                    path.replace("\\", "/"): set(lines)
                    for path, lines in eligible_lines.items()
                }
                eligible_context = eligible_context.model_copy(
                    update={
                        "changed_lines": {
                            path: [
                                line
                                for line in lines
                                if line
                                in eligible_keys.get(path.replace("\\", "/"), set())
                            ]
                            for path, lines in (eligible_context.changed_lines or {}).items()
                        },
                        "changed_line_content": {
                            path: {
                                line: text
                                for line, text in line_map.items()
                                if line
                                in eligible_keys.get(path.replace("\\", "/"), set())
                            }
                            for path, line_map in eligible_context.changed_line_content.items()
                        },
                        "analysis_findings": list(built_prompt.included_analysis_findings),
                    }
                )

                remaining = deadline - monotonic()
                if remaining < (
                    MIN_REVIEW_REQUEST_BUDGET_SECONDS
                    + REQUEST_TIMEOUT_SAFETY_MARGIN_SECONDS
                ):
                    raise RuntimeError(
                        "review budget exhausted before model execution; "
                        "no model request was started"
                    )
                configured_timeout = getattr(self.provider, "timeout", None)
                timeout_adjusted = (
                    isinstance(configured_timeout, (int, float))
                    and not isinstance(configured_timeout, bool)
                    and configured_timeout > 0
                )
                if timeout_adjusted:
                    setattr(
                        self.provider,
                        "timeout",
                        min(
                            float(configured_timeout),
                            remaining - REQUEST_TIMEOUT_SAFETY_MARGIN_SECONDS,
                        ),
                    )
                try:
                    logger.info(
                        "Semantic request prepared: batch=%d/%d model=%s prompt_chars=%d "
                        "estimated_prompt_tokens=%d selected_files=%d "
                        "changed_lines_available=%d changed_lines_included=%d "
                        "static_findings_available=%d static_findings_supplied=%d "
                        "static_findings_included=%d generation_options=%s "
                        "request_timeout_seconds=%s aggregate_budget_remaining_seconds=%.2f",
                        batch_index,
                        min(len(batches), MAX_REVIEW_REQUESTS),
                        getattr(self.provider, "model", "custom"),
                        len(built_prompt.prompt),
                        (len(built_prompt.prompt) + 3) // 4,
                        len(built_prompt.included_files),
                        sum(
                            len(set(lines))
                            for lines in (batch_context.changed_lines or {}).values()
                        ),
                        built_prompt.included_changed_lines,
                        len(batch_context.analysis_findings),
                        built_prompt.supplied_findings,
                        built_prompt.included_findings,
                        {
                            "num_predict": getattr(self.provider, "num_predict", "provider-default"),
                            "num_ctx": getattr(self.provider, "num_ctx", "provider-default"),
                        },
                        getattr(self.provider, "timeout", "provider-default"),
                        remaining,
                    )
                    if deadline - monotonic() < (
                        MIN_REVIEW_REQUEST_BUDGET_SECONDS
                        + REQUEST_TIMEOUT_SAFETY_MARGIN_SECONDS
                    ):
                        raise RuntimeError(
                            "review budget exhausted before model execution; "
                            "no model request was started"
                        )
                    attempted += 1
                    provider_started = monotonic()
                    raw_response = self.provider.review(built_prompt.prompt)
                finally:
                    if timeout_adjusted:
                        setattr(self.provider, "timeout", configured_timeout)
                provider_seconds = monotonic() - provider_started
                if deadline - monotonic() <= 0:
                    raise RuntimeError("review budget exhausted during model execution")

                parsing_started = monotonic()
                batch_findings = self.parser.parse(raw_response, eligible_context)
                findings.extend(batch_findings)
                if deadline - monotonic() <= 0:
                    raise RuntimeError("review budget exhausted during response parsing")

                line_lookup = {
                    path.replace("\\", "/"): set(lines)
                    for path, lines in (context.changed_lines or {}).items()
                }
                eligible_lookup = {
                    path.replace("\\", "/"): set(lines)
                    for path, lines in eligible_lines.items()
                }
                for path in included:
                    key = path.replace("\\", "/")
                    changed = line_lookup.get(key, set())
                    supplied = eligible_lookup.get(key, set())
                    if changed and changed.issubset(supplied):
                        reviewed.add(path)
                    elif changed.intersection(supplied):
                        partially_reviewed.add(path)
                    else:
                        failures.append(
                            f"Semantic review did not include changed lines for {path}."
                        )
                logger.info(
                    "Semantic batch completed: batch=%d prompt_preparation_seconds=%.3f "
                    "model_wait_seconds=%.3f response_parse_seconds=%.3f "
                    "semantic_findings=%d",
                    batch_index,
                    provider_started - preparation_started,
                    provider_seconds,
                    monotonic() - parsing_started,
                    len(batch_findings),
                )
            except (RuntimeError, ReviewResponseError, ValueError, AttributeError) as error:
                failed += 1
                logger.warning(
                    "Semantic batch failed: batch=%d prompt_preparation_seconds=%.3f "
                    "model_wait_seconds=%.3f response_parse_seconds=%.3f exception=%s",
                    batch_index,
                    monotonic() - preparation_started,
                    monotonic() - provider_started if provider_started is not None else 0.0,
                    monotonic() - parsing_started if parsing_started is not None else 0.0,
                    type(error).__name__,
                )
                if isinstance(error, ReviewResponseError):
                    reason = f"Member 3 rejected the model response: {error}"
                elif isinstance(error, RuntimeError):
                    reason = f"failed (RuntimeError): {error}"
                else:
                    reason = f"failed ({type(error).__name__}): {error}"
                failures.append(f"Semantic review batch {batch_index} {reason}")

        if len(batches) > MAX_REVIEW_REQUESTS and attempted >= MAX_REVIEW_REQUESTS:
            failures.append(
                f"Semantic review request limit ({MAX_REVIEW_REQUESTS}) reached; "
                "remaining batches were not started."
            )
        reviewed.difference_update(partially_reviewed)
        unreviewed = [
            path
            for path in paths
            if path not in reviewed and path not in partially_reviewed
        ]
        if unreviewed:
            failures.append(
                "Semantic review scope omitted changed files: "
                + ", ".join(unreviewed)
            )
        if partially_reviewed:
            failures.append(
                "Semantic review only partially covered changed lines in: "
                + ", ".join(path for path in paths if path in partially_reviewed)
            )
        final_findings = deduplicate_findings(findings)
        self._set_result(
            limitations,
            failures,
            [path for path in paths if path in reviewed],
            [path for path in paths if path in partially_reviewed],
            unreviewed,
            attempted,
            failed,
            len(final_findings),
            monotonic() - started,
        )
        return final_findings

    def _set_result(
        self,
        limitations: tuple[str, ...] | list[str],
        failures: tuple[str, ...] | list[str],
        reviewed: tuple[str, ...] | list[str],
        partially_reviewed: tuple[str, ...] | list[str],
        unreviewed: tuple[str, ...] | list[str],
        batch_count: int,
        failed_batch_count: int,
        findings_count: int = 0,
        elapsed_seconds: float = 0.0,
    ) -> None:
        self.last_context_limitations = tuple(dict.fromkeys(limitations))
        self.last_review_failures = tuple(failures)
        self.last_reviewed_files = tuple(dict.fromkeys(reviewed))
        self.last_partially_reviewed_files = tuple(dict.fromkeys(partially_reviewed))
        self.last_unreviewed_files = tuple(dict.fromkeys(unreviewed))
        self.last_batch_count = batch_count
        self.last_failed_batch_count = failed_batch_count
        self.last_findings_count = findings_count
        self.last_elapsed_seconds = max(0.0, elapsed_seconds)

    @staticmethod
    def _ranked_paths(context: ReviewContext) -> list[str]:
        findings_by_file: dict[str, list[dict]] = {}
        for finding in context.analysis_findings:
            key = str(finding.get("file", "")).replace("\\", "/")
            findings_by_file.setdefault(key, []).append(finding)

        def priority(path: str) -> tuple[int, int, int, int]:
            key = path.replace("\\", "/")
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
        path_keys = {path.replace("\\", "/") for path in paths}

        def subset(mapping: dict[str, object] | None) -> dict[str, object] | None:
            if mapping is None:
                return None
            return {
                path: value
                for path, value in mapping.items()
                if path.replace("\\", "/") in path_keys
            }

        findings = [
            finding
            for finding in context.analysis_findings
            if str(finding.get("file", "")).replace("\\", "/") in path_keys
        ]
        return context.model_copy(
            update={
                "changed_code": subset(context.changed_code) or {},
                "changed_lines": subset(context.changed_lines),
                "changed_line_content": subset(context.changed_line_content) or {},
                "analysis_findings": findings,
            }
        )
