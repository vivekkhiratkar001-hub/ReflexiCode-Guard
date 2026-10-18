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
    ) -> None:
        self.provider = provider
        self.prompt_builder = prompt_builder or ReviewPromptBuilder()
        self.parser = parser or ReviewResponseParser(
            max_findings=self.prompt_builder.max_findings
        )
        self.last_context_limitations: tuple[str, ...] = ()
        self.last_review_failures: tuple[str, ...] = ()
        self.last_reviewed_files: tuple[str, ...] = ()
        self.last_unreviewed_files: tuple[str, ...] = ()
        self.last_batch_count = 0
        self.last_failed_batch_count = 0
        self.last_findings_count = 0

    def review(self, context: ReviewContext) -> list[ReviewFinding]:
        """Review one prioritized, bounded subset and report omissions explicitly."""
        paths = self._ranked_paths(context)
        if not paths:
            self._set_result((), (), (), (), 0, 0)
            return []

        preparation_started = monotonic()
        provider_started: float | None = None
        parsing_started: float | None = None
        limitations: list[str] = []
        failures: list[str] = []
        reviewed: list[str] = []
        findings: list[ReviewFinding] = []
        built_prompt = None
        request_failed = False
        try:
            built_prompt = self.prompt_builder.build_with_metadata(context)
            limitations.extend(built_prompt.limitations)
            included = list(built_prompt.included_files)
            eligible_context = self._context_for_files(context, included)
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
                            if line in eligible_keys.get(
                                path.replace("\\", "/"), set()
                            )
                        ]
                        for path, lines in (eligible_context.changed_lines or {}).items()
                    },
                    "changed_line_content": {
                        path: {
                            line: text
                            for line, text in line_map.items()
                            if line
                            in eligible_keys.get(
                                path.replace("\\", "/"), set()
                            )
                        }
                        for path, line_map in eligible_context.changed_line_content.items()
                    },
                    "analysis_findings": list(built_prompt.included_analysis_findings),
                }
            )
            if not included:
                raise RuntimeError("No changed source lines fit in the semantic-review prompt.")

            logger.info(
                "Semantic request prepared: model=%s prompt_chars=%d "
                "estimated_prompt_tokens=%d selected_files=%d changed_lines=%d "
                "static_findings=%d generation_options=%s timeout_seconds=%s",
                getattr(self.provider, "model", "custom"),
                len(built_prompt.prompt),
                (len(built_prompt.prompt) + 3) // 4,
                len(built_prompt.included_files),
                built_prompt.included_changed_lines,
                built_prompt.included_findings,
                {
                    "num_predict": getattr(self.provider, "num_predict", "provider-default"),
                    "num_ctx": getattr(self.provider, "num_ctx", "provider-default"),
                },
                getattr(self.provider, "timeout", "provider-default"),
            )
            provider_started = monotonic()
            raw_response = self.provider.review(built_prompt.prompt)
            provider_seconds = monotonic() - provider_started
            parsing_started = monotonic()
            findings = self.parser.parse(raw_response, eligible_context)
            parsing_seconds = monotonic() - parsing_started
            reviewed.extend(included)
        except (RuntimeError, ReviewResponseError, ValueError) as error:
            request_failed = True
            logger.warning(
                "Semantic review failed: prompt_preparation_seconds=%.3f "
                "model_wait_seconds=%.3f response_parse_seconds=%.3f exception=%s",
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
            failures.append(f"Semantic review {reason}")
            provider_seconds = monotonic() - provider_started if provider_started else 0.0
            parsing_seconds = monotonic() - parsing_started if parsing_started else 0.0
        else:
            logger.info(
                "Semantic review completed: prompt_chars=%d estimated_prompt_tokens=%d "
                "selected_files=%d changed_lines_available=%d changed_lines_included=%d "
                "static_findings_available=%d static_findings_supplied=%d "
                "static_findings_included=%d semantic_findings=%d "
                "prompt_preparation_seconds=%.3f model_wait_seconds=%.3f "
                "response_parse_seconds=%.3f",
                len(built_prompt.prompt),
                (len(built_prompt.prompt) + 3) // 4,
                len(built_prompt.included_files),
                sum(len(set(lines)) for lines in (context.changed_lines or {}).values()),
                built_prompt.included_changed_lines,
                len(context.analysis_findings),
                built_prompt.supplied_findings,
                built_prompt.included_findings,
                len(findings),
                provider_started - preparation_started,
                provider_seconds,
                parsing_seconds,
            )

        unreviewed = [path for path in paths if path not in set(reviewed)]
        if unreviewed:
            failures.append(
                "Semantic review scope omitted changed files: "
                + ", ".join(unreviewed)
            )
        self._set_result(
            limitations,
            failures,
            reviewed,
            unreviewed,
            1,
            int(request_failed),
            len(findings),
        )
        return deduplicate_findings(findings)

    def _set_result(
        self,
        limitations: tuple[str, ...] | list[str],
        failures: tuple[str, ...] | list[str],
        reviewed: tuple[str, ...] | list[str],
        unreviewed: tuple[str, ...] | list[str],
        batch_count: int,
        failed_batch_count: int,
        findings_count: int = 0,
    ) -> None:
        self.last_context_limitations = tuple(dict.fromkeys(limitations))
        self.last_review_failures = tuple(failures)
        self.last_reviewed_files = tuple(dict.fromkeys(reviewed))
        self.last_unreviewed_files = tuple(dict.fromkeys(unreviewed))
        self.last_batch_count = batch_count
        self.last_failed_batch_count = failed_batch_count
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
