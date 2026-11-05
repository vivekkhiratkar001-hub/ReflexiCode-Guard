"""Small end-to-end orchestration for semantic PR review."""

import logging
from collections.abc import Sequence
from time import monotonic
from typing import Callable

from .context import ReviewContext
from .parser import (
    FindingRejection,
    ReviewResponseError,
    ReviewResponseParser,
    deduplicate_findings,
    validate_suggestion,
)
from .prompts import ReviewPromptBuilder
from .provider import LLMProvider
from .schemas import ReviewFinding

logger = logging.getLogger(__name__)
MIN_REQUEST_WINDOW_SECONDS = 60
SAFETY_MARGIN_SECONDS = 5
AGGREGATE_CLEANUP_RESERVE_SECONDS = 3.0
# Seconds held back from every request to absorb deadline-enforcement overhead
# (timer resolution, socket close propagation, and request teardown) so that a
# started request cannot push the aggregate elapsed time past the configured
# budget. Without this margin, a request allocated nearly all remaining time can
# overrun its deadline by several seconds and breach the aggregate budget.
AGGREGATE_REQUEST_OVERRUN_MARGIN_SECONDS = 5.0


class ReviewPipeline:
    """Connect context, prompt, provider, and validated review findings."""

    def __init__(
        self,
        provider: LLMProvider,
        prompt_builder: ReviewPromptBuilder | None = None,
        parser: ReviewResponseParser | None = None,
        *,
        max_requests: int = 3,
        review_budget_seconds: int = 240,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 1:
            raise ValueError("max_requests must be a positive integer.")
        if (
            isinstance(review_budget_seconds, bool)
            or not isinstance(review_budget_seconds, int)
            or review_budget_seconds < 1
        ):
            raise ValueError("review_budget_seconds must be a positive integer.")
        self.provider = provider
        self.prompt_builder = prompt_builder or ReviewPromptBuilder()
        self.parser = parser or ReviewResponseParser(
            max_findings=self.prompt_builder.max_findings
        )
        self.last_context_limitations: tuple[str, ...] = ()
        self.last_review_failures: tuple[str, ...] = ()
        self.last_rejections: tuple[FindingRejection, ...] = ()
        self.last_reviewed_files: tuple[str, ...] = ()
        self.last_partially_reviewed_files: tuple[str, ...] = ()
        self.last_unreviewed_files: tuple[str, ...] = ()
        self.last_batch_count = 0
        self.last_failed_batch_count = 0
        self.last_successful_batch_count = 0
        self.last_skipped_batch_count = 0
        self.last_total_chunks = 0
        self.last_findings_count = 0
        self.last_review_elapsed_seconds = 0.0
        self.last_root_failure_category: str | None = None
        self.last_root_failure_message: str | None = None
        self.last_budget_exhausted: bool = False
        self.review_budget_seconds = review_budget_seconds
        self.max_requests = max_requests
        self._clock = clock

    def review(self, context: ReviewContext) -> list[ReviewFinding]:
        """Review prioritized chunks sequentially within request and scheduling budgets."""
        review_started = self._clock()
        self.last_root_failure_category = None
        self.last_root_failure_message = None
        self.last_budget_exhausted = False
        self.last_rejections = ()
        paths = self._ranked_paths(context)
        if not paths:
            self.last_review_elapsed_seconds = 0.0
            self._set_result((), (), (), (), (), 0, 0)
            return []

        limitations: list[str] = []
        failures: list[str] = []
        fully_reviewed: list[str] = []
        partially_reviewed: list[str] = []
        findings: list[ReviewFinding] = []
        rejections: list[FindingRejection] = []
        failed_requests = 0
        attempted_requests = 0
        chunk_size = self.prompt_builder.max_files
        chunks = [paths[index : index + chunk_size] for index in range(0, len(paths), chunk_size)]
        request_limit = min(self.max_requests, len(chunks))

        for request_index, chunk_paths in enumerate(chunks[:request_limit], start=1):
            elapsed = self._clock() - review_started
            remaining = self.review_budget_seconds - elapsed
            configured_timeout = getattr(self.provider, "timeout", None)
            provider_min_window = getattr(self.provider, "min_request_window_seconds", None)
            base_window = (
                provider_min_window
                if isinstance(provider_min_window, (int, float)) and provider_min_window > 0
                else MIN_REQUEST_WINDOW_SECONDS
            )
            # The guard must be the fixed minimum window, not the full provider timeout:
            # otherwise remaining >= timeout always holds past this check and the
            # timeout-tightening below can never shorten a request.
            minimum_window = (
                min(base_window, configured_timeout)
                if isinstance(configured_timeout, (int, float))
                else base_window
            )
            # The minimum total time needed to safely start another request is the
            # fixed request window plus the cleanup reserve and an overrun margin.
            # Without this, a request allocated nearly all remaining time can
            # overrun its deadline (timer/socket race conditions) and breach the
            # aggregate budget.
            minimum_total = (
                minimum_window
                + AGGREGATE_CLEANUP_RESERVE_SECONDS
                + AGGREGATE_REQUEST_OVERRUN_MARGIN_SECONDS
            )
            if remaining < minimum_total:
                self.last_budget_exhausted = True
                limitations.append(
                    f"Aggregate semantic-review budget of {self.review_budget_seconds} seconds "
                    f"left only {max(0.0, remaining):.1f} seconds, below the "
                    f"{minimum_window}-second minimum request window plus "
                    f"{AGGREGATE_CLEANUP_RESERVE_SECONDS:g} seconds reserve and "
                    f"{AGGREGATE_REQUEST_OVERRUN_MARGIN_SECONDS:g} seconds overrun "
                    "margin; the next request was not started."
                )
                break

            preparation_started = self._clock()
            provider_started: float | None = None
            parsing_started: float | None = None
            original_timeout = getattr(self.provider, "timeout", None)
            provider_timeout_changed = False
            try:
                chunk_context = self._context_for_files(context, chunk_paths)
                built_prompt = self.prompt_builder.build_with_metadata(chunk_context)
                limitations.extend(built_prompt.limitations)
                included = list(built_prompt.included_files)
                if not included:
                    raise RuntimeError("No changed source lines fit in this review chunk.")
                eligible_context = self._eligible_context(
                    chunk_context, built_prompt, included
                )

                # Prompt construction is synchronous, so calculate the usable request
                # window after it. Reserve time for transport teardown and parsing.
                remaining = self.review_budget_seconds - (
                    self._clock() - review_started
                )
                provider_timeout = (
                    float(original_timeout)
                    if isinstance(original_timeout, (int, float))
                    else remaining
                )
                request_timeout = min(
                    provider_timeout,
                    remaining - AGGREGATE_CLEANUP_RESERVE_SECONDS,
                )
                minimum_window = min(base_window, provider_timeout)
                if request_timeout < minimum_window:
                    limitation = (
                        f"Aggregate semantic-review budget of {self.review_budget_seconds} seconds "
                        f"left only {max(0.0, remaining):.1f} seconds after prompt preparation, "
                        f"below the {minimum_window:g}-second minimum request window plus "
                        f"{AGGREGATE_CLEANUP_RESERVE_SECONDS:g} seconds reserved for cleanup; "
                        "the next model request was not started."
                    )
                    limitations.append(limitation)
                    failures.append(limitation)
                    break

                deadline_review = getattr(self.provider, "review_with_deadline", None)
                if not callable(deadline_review) and isinstance(original_timeout, (int, float)):
                    # Compatibility providers are synchronous and cannot be cancelled;
                    # shorten their configured timeout and reject late returns.
                    self.provider.timeout = max(1, int(request_timeout))
                    provider_timeout_changed = True
                logger.info(
                    "Semantic request prepared: request=%d/%d model=%s prompt_chars=%d "
                    "estimated_prompt_tokens=%d selected_files=%d "
                    "changed_lines_available=%d changed_lines_included=%d "
                    "static_findings_available=%d static_findings_supplied=%d "
                    "static_findings_included=%d generation_options=%s request_timeout_seconds=%s "
                    "aggregate_remaining_seconds=%.1f",
                    request_index,
                    request_limit,
                    getattr(self.provider, "model", "custom"),
                    len(built_prompt.prompt),
                    (len(built_prompt.prompt) + 3) // 4,
                    len(included),
                    sum(len(set(lines)) for lines in (chunk_context.changed_lines or {}).values()),
                    built_prompt.included_changed_lines,
                    len(chunk_context.analysis_findings),
                    built_prompt.supplied_findings,
                    built_prompt.included_findings,
                    {
                        "num_predict": getattr(self.provider, "num_predict", "provider-default"),
                        "num_ctx": getattr(self.provider, "num_ctx", "provider-default"),
                    },
                    request_timeout,
                    remaining,
                )
                attempted_requests += 1
                provider_started = self._clock()
                raw_response: str | None = None
                try:
                    if callable(deadline_review):
                        raw_response = deadline_review(
                            built_prompt.prompt,
                            timeout_seconds=request_timeout,
                        )
                    else:
                        raw_response = self.provider.review(built_prompt.prompt)
                finally:
                    if provider_timeout_changed:
                        self.provider.timeout = original_timeout
                        provider_timeout_changed = False
                provider_seconds = self._clock() - provider_started
                if provider_seconds > request_timeout:
                    raise RuntimeError(
                        f"Provider exceeded its {request_timeout:.1f}-second wall-clock "
                        f"request deadline (elapsed {provider_seconds:.2f} seconds)."
                    )
                if self._clock() - review_started >= self.review_budget_seconds:
                    raise RuntimeError(
                        "Aggregate semantic-review deadline expired during model execution."
                    )
                parsing_started = self._clock()
                try:
                    chunk_findings = self.parser.parse(raw_response, eligible_context)
                except ReviewResponseError as first_error:
                    # One correction retry for schema/parse failures only.
                    # Transport, timeout, and auth errors (RuntimeError) are not retried.
                    retry_remaining = self.review_budget_seconds - (self._clock() - review_started)
                    minimum_retry_window = AGGREGATE_CLEANUP_RESERVE_SECONDS + 2.0
                    if retry_remaining < minimum_retry_window:
                        raise  # Budget too tight to retry safely.
                    correction_prompt = (
                        "The previous response failed schema validation with the following error:\n"
                        f"{first_error}\n\n"
                        "Return a complete corrected JSON response in exactly this shape, "
                        "with all evidence strings copied verbatim from the 'source' field "
                        "of the relevant changed-code JSONL records that were provided:\n"
                        '{"findings":[{"file":"path/to/file","line":1,"category":"Bug",'
                        '"severity":"High","message":"Concise actionable explanation",'
                        '"evidence":"Exact complete source string from that eligible record",'
                        '"suggestion":"Practical correction or null"}]}\n'
                        "If no defect can be supported by a changed-code source line, "
                        'return {"findings":[]}. Do not invent evidence.'
                    )
                    current_provider_timeout = getattr(self.provider, "timeout", None)
                    provider_timeout = (
                        float(current_provider_timeout)
                        if isinstance(current_provider_timeout, (int, float))
                        else retry_remaining
                    )
                    retry_timeout = min(
                        provider_timeout,
                        retry_remaining - AGGREGATE_CLEANUP_RESERVE_SECONDS,
                    )
                    logger.warning(
                        "Semantic request schema validation failed; attempting one correction retry: "
                        "request=%d/%d error=%s retry_timeout_seconds=%.1f",
                        request_index,
                        request_limit,
                        first_error,
                        retry_timeout,
                    )
                    attempted_requests += 1
                    retry_provider_started = self._clock()
                    deadline_review_retry = getattr(self.provider, "review_with_deadline", None)
                    retry_timeout_changed = False
                    try:
                        try:
                            if callable(deadline_review_retry):
                                raw_response = deadline_review_retry(
                                    correction_prompt,
                                    timeout_seconds=retry_timeout,
                                )
                            else:
                                original_timeout_retry = getattr(self.provider, "timeout", None)
                                if isinstance(original_timeout_retry, (int, float)):
                                    self.provider.timeout = max(1, int(retry_timeout))
                                    retry_timeout_changed = True
                                raw_response = self.provider.review(correction_prompt)
                        finally:
                            if retry_timeout_changed:
                                self.provider.timeout = original_timeout_retry
                        retry_provider_seconds = self._clock() - retry_provider_started
                        if retry_provider_seconds > retry_timeout:
                            raise RuntimeError(
                                f"Provider exceeded its {retry_timeout:.1f}-second wall-clock "
                                f"request deadline (elapsed {retry_provider_seconds:.2f} seconds)."
                            )
                        if self._clock() - review_started >= self.review_budget_seconds:
                            raise RuntimeError(
                                "Aggregate semantic-review deadline expired during model execution."
                            )
                        logger.info(
                            "Correction retry completed: request=%d/%d elapsed_seconds=%.2f",
                            request_index,
                            request_limit,
                            retry_provider_seconds,
                        )
                        chunk_findings = self.parser.parse(raw_response, eligible_context)
                    except ReviewResponseError:
                        raise
                    except Exception as retry_err:
                        logger.warning(
                            "Correction retry failed: %s; falling back to original validation error",
                            retry_err,
                        )
                        raise first_error from retry_err
                rejections.extend(getattr(self.parser, "last_rejections", ()))
                parsing_seconds = self._clock() - parsing_started
                findings.extend(chunk_findings)
                eligible_lines = built_prompt.eligible_changed_lines or {}
                partial_chunk: list[str] = []
                full_chunk: list[str] = []
                changed_lines_by_file = chunk_context.changed_lines or {}
                for path in included:
                    expected_lines = set(changed_lines_by_file.get(path, ()))
                    supplied_lines = set(eligible_lines.get(path, ()))
                    if expected_lines and supplied_lines.issuperset(expected_lines):
                        full_chunk.append(path)
                    elif not expected_lines and path in included:
                        full_chunk.append(path)
                    elif supplied_lines:
                        partial_chunk.append(path)
                fully_reviewed.extend(full_chunk)
                partially_reviewed.extend(partial_chunk)
                if partial_chunk:
                    failures.append(
                        "Semantic review included only some changed lines in: "
                        + ", ".join(partial_chunk)
                    )
                omitted_from_chunk = [path for path in chunk_paths if path not in set(included)]
                if omitted_from_chunk:
                    limitations.append(
                        "Changed files did not fit in the prompt: " + ", ".join(omitted_from_chunk)
                    )
                logger.info(
                    "Semantic request completed: request=%d/%d elapsed_seconds=%.2f "
                    "parse_seconds=%.3f findings=%d files_reviewed=%d",
                    request_index,
                    request_limit,
                    provider_seconds,
                    parsing_seconds,
                    len(chunk_findings),
                    len(full_chunk) + len(partial_chunk),
                )
                if self._clock() - review_started >= self.review_budget_seconds:
                    deadline_message = (
                        "Aggregate semantic-review deadline expired during response parsing; "
                        "validated findings were retained, but no further request will start."
                    )
                    limitations.append(deadline_message)
                    failures.append(deadline_message)
                    failed_requests += 1
                    break
            except (RuntimeError, ReviewResponseError, ValueError) as error:
                if parsing_started is not None:
                    rejections.extend(getattr(self.parser, "last_rejections", ()))
                failed_requests += 1
                if self.last_root_failure_category is None:
                    cat = getattr(error, "error_category", None)
                    if cat:
                        self.last_root_failure_category = str(cat)
                    elif isinstance(error, ReviewResponseError):
                        self.last_root_failure_category = "RESPONSE_VALIDATION_ERROR"
                    elif "timeout" in str(error).lower():
                        self.last_root_failure_category = "TIMEOUT"
                    else:
                        self.last_root_failure_category = type(error).__name__
                    self.last_root_failure_message = " ".join(str(error).split()).rstrip(".,; ")
                if provider_timeout_changed:
                    self.provider.timeout = original_timeout
                    provider_timeout_changed = False
                logger.warning(
                    "Semantic request failed: request=%d/%d preparation_seconds=%.3f "
                    "model_wait_seconds=%.3f parse_seconds=%.3f exception=%s",
                    request_index,
                    request_limit,
                    self._clock() - preparation_started,
                    self._clock() - provider_started if provider_started is not None else 0.0,
                    self._clock() - parsing_started if parsing_started is not None else 0.0,
                    f"{type(error).__name__}: {error}",
                )
                reason = (
                    f"Member 3 rejected the model response: {error}"
                    if isinstance(error, ReviewResponseError)
                    else f"failed ({type(error).__name__}): {error}"
                )
                failures.append(
                    f"Semantic request {request_index}/{request_limit} "
                    f"[{', '.join(chunk_paths)}] {reason}"
                )
                continue

        if rejections:
            limitations.append(
                f"Semantic review rejected {len(rejections)} invalid finding(s) individually."
            )
        accounted = set(fully_reviewed) | set(partially_reviewed)
        unreviewed = [path for path in paths if path not in accounted]
        if len(chunks) > request_limit:
            limitations.append(
                f"Request limit of {self.max_requests} left {len(paths) - len(accounted)} "
                "changed files outside semantic review."
            )
        if unreviewed:
            failures.append(
                "Semantic review scope omitted changed files: " + ", ".join(unreviewed)
            )
        if partially_reviewed:
            limitations.append(
                "Partially reviewed files (some changed lines did not fit): "
                + ", ".join(dict.fromkeys(partially_reviewed))
            )
        self.last_review_elapsed_seconds = max(0.0, self._clock() - review_started)
        if unreviewed or partially_reviewed:
            logger.info(
                "Semantic coverage: fully_reviewed_files=%d partially_reviewed_files=%d "
                "not_reviewed_files=%d "
                "requests_attempted=%d requests_failed=%d elapsed_seconds=%.2f "
                "budget_seconds=%d",
                len(set(fully_reviewed)),
                len(set(partially_reviewed)),
                len(set(unreviewed)),
                attempted_requests,
                failed_requests,
                self.last_review_elapsed_seconds,
                self.review_budget_seconds,
            )
        filtered = self._filter_unsupported_findings(findings, context=context)
        self.last_rejections = tuple(rejections)
        self._set_result(
            limitations,
            failures,
            fully_reviewed,
            partially_reviewed,
            unreviewed,
            attempted_requests,
            failed_requests,
            len(filtered),
            len(chunks),
        )
        return filtered

    @staticmethod
    def _filter_unsupported_findings(
        findings: Sequence[ReviewFinding],
        context: ReviewContext | None = None,
    ) -> list[ReviewFinding]:
        """Filter out findings that are duplicate, weak, or lacking actionable defect evidence."""
        deduped = deduplicate_findings(findings)
        valid: list[ReviewFinding] = []
        for finding in deduped:
            ev = (finding.evidence or "").strip()
            if not ev or len(ev) < 4:
                continue
            if finding.category.casefold() in ("bug", "security", "correctness"):
                if ev.startswith(("#", "//", "/*", "*", '"""', "'''")):
                    continue
            msg = (finding.message or "").strip()
            if len(msg) < 8:
                continue
            # Re-verify suggestion does not repeat buggy evidence or violate rules
            reason = validate_suggestion(finding, context, strict=False)
            if reason is not None:
                finding.suggestion = None
            valid.append(finding)
        return valid

    @staticmethod
    def _eligible_context(context, built_prompt, included: list[str]):
        eligible_context = ReviewPipeline._context_for_files(context, included)
        eligible_lines = built_prompt.eligible_changed_lines or {}
        eligible_keys = {
            path.replace("\\", "/"): set(lines)
            for path, lines in eligible_lines.items()
        }
        return eligible_context.model_copy(
            update={
                "changed_lines": {
                    path: [
                        line
                        for line in lines
                        if line in eligible_keys.get(path.replace("\\", "/"), set())
                    ]
                    for path, lines in (eligible_context.changed_lines or {}).items()
                },
                "changed_line_content": {
                    path: {
                        line: text
                        for line, text in line_map.items()
                        if line in eligible_keys.get(path.replace("\\", "/"), set())
                    }
                    for path, line_map in eligible_context.changed_line_content.items()
                },
                "analysis_findings": list(built_prompt.included_analysis_findings),
            }
        )

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
        total_chunks: int = 0,
    ) -> None:
        self.last_context_limitations = tuple(dict.fromkeys(limitations))
        self.last_review_failures = tuple(failures)
        self.last_reviewed_files = tuple(dict.fromkeys(reviewed))
        self.last_partially_reviewed_files = tuple(dict.fromkeys(partially_reviewed))
        self.last_unreviewed_files = tuple(dict.fromkeys(unreviewed))
        self.last_batch_count = batch_count
        self.last_failed_batch_count = failed_batch_count
        self.last_successful_batch_count = max(0, batch_count - failed_batch_count)
        self.last_total_chunks = total_chunks or batch_count
        self.last_skipped_batch_count = max(0, self.last_total_chunks - batch_count)
        self.last_findings_count = findings_count

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
