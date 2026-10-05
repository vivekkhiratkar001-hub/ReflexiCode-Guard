"""Parsing and deterministic validation of structured model responses."""

import json
import re

from pydantic import ValidationError

from .context import ReviewContext
from .schemas import ReviewFinding, ReviewFindings


class ReviewResponseError(ValueError):
    """Raised when a model response cannot be trusted as review findings."""


def _extract_json_object(raw_response: str) -> str:
    """Extract one JSON object, allowing harmless text around the JSON."""
    decoder = json.JSONDecoder()
    for start, character in enumerate(raw_response):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(raw_response[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return raw_response[start : start + end]
    raise ReviewResponseError("The model response does not contain a valid JSON object.")


def _normalized(value: str) -> str:
    """Normalize whitespace and case for a conservative evidence match."""
    return re.sub(r"\s+", "", value).casefold()


class ReviewResponseParser:
    """Parse JSON, validate the schema, and check context-backed locations."""

    def parse(
        self,
        raw_response: str,
        context: ReviewContext | None = None,
    ) -> list[ReviewFinding]:
        """Return validated findings or raise a useful error for bad output."""
        if not raw_response or not raw_response.strip():
            raise ReviewResponseError("The model returned an empty response.")

        json_text = _extract_json_object(raw_response)
        try:
            envelope = ReviewFindings.model_validate_json(json_text)
        except ValidationError as exc:
            raise ReviewResponseError(
                f"The model response failed review-schema validation: {exc}"
            ) from exc

        findings = envelope.findings
        if context is not None:
            self._validate_against_context(findings, context)
        return findings

    @staticmethod
    def _validate_against_context(
        findings: list[ReviewFinding], context: ReviewContext
    ) -> None:
        """Reject findings for unknown files, unchanged lines, or unsupported code."""
        for finding in findings:
            if context.changed_code and finding.file not in context.changed_code:
                raise ReviewResponseError(
                    f"Finding refers to file {finding.file!r}, which is not in changed_code."
                )

            if context.changed_lines is not None:
                changed_lines = context.changed_lines.get(finding.file)
                if changed_lines is None or finding.line not in changed_lines:
                    raise ReviewResponseError(
                        f"Finding at {finding.file}:{finding.line} is not listed as a changed line."
                    )

            code = context.changed_code.get(finding.file, "")
            matching_analysis = any(
                analysis_finding.get("file") == finding.file
                and _normalized(finding.evidence)
                in _normalized(json.dumps(analysis_finding, ensure_ascii=False))
                for analysis_finding in context.analysis_findings
            )
            if (
                code
                and _normalized(finding.evidence) not in _normalized(code)
                and not matching_analysis
            ):
                raise ReviewResponseError(
                    f"Finding evidence for {finding.file}:{finding.line} does not match supplied changed code."
                )