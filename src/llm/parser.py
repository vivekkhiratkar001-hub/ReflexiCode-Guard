"""Parsing and deterministic validation of structured model responses."""

import json
import re
import unicodedata
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Any

from pydantic import ValidationError

from .context import ReviewContext
from .schemas import ReviewFinding, ReviewFindings


class ReviewResponseError(ValueError):
    """Raised when a model response cannot be trusted as review findings."""


def _extract_json_object(raw_response: str) -> str:
    """Prefer the structured review payload over earlier stray brace pairs."""
    text = raw_response.strip()
    fenced = re.match(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, str]] = []
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        payload = text[start : start + end]
        score = 0
        lower = payload.casefold()
        for keyword in ("findings", "file", "line", "category", "severity", "message", "evidence"):
            if keyword in lower:
                score += 5
        if value == {} and score == 0:
            continue
        candidates.append((score, start, payload))

    if not candidates:
        raise ReviewResponseError("The model response does not contain a valid JSON object.")

    _, _, best_payload = max(candidates, key=lambda item: (item[0], item[1]))
    return best_payload


def _normalized(value: str) -> str:
    """Normalize whitespace and case for a conservative evidence match."""
    return re.sub(r"\s+", "", value).casefold()


def _normalize_path(value: str) -> str:
    """Use a simple cross-platform canonical form for file paths."""
    return value.replace("\\", "/").strip().lstrip("./").casefold()


def _normalize_message(value: str) -> str:
    """Flatten messages for duplicate detection without heavy NLP."""
    text = unicodedata.normalize("NFKD", value.casefold())
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return " ".join(text.split())


def _looks_meaningful_evidence(value: str) -> bool:
    """Reject obvious weak evidence like a bare keyword or single-character token."""
    text = value.strip()
    if not text:
        return False
    if len(text) < 4:
        return False
    tokens = [token for token in re.split(r"\s+", text) if token]
    if len(tokens) == 1 and len(text) < 8 and not any(ch in text for ch in "+-*/=<>!():[]{};"):
        return False
    return True


def _canonical_file_lookup(mapping: dict[str, Any]) -> dict[str, Any]:
    """Map normalized file paths to the original key in the context."""
    return {
        _normalize_path(str(path)): value
        for path, value in mapping.items()
    }


def _analysis_matches_finding(
    finding: ReviewFinding,
    analysis_findings: Sequence[dict[str, Any]],
) -> bool:
    """Allow evidence to be justified by supplied static-analysis metadata."""
    if not analysis_findings:
        return False

    normalized_file = _normalize_path(finding.file)
    normalized_evidence = _normalized(finding.evidence)
    for analysis in analysis_findings:
        if not isinstance(analysis, dict):
            continue
        if _normalize_path(str(analysis.get("file", ""))) != normalized_file:
            continue
        payload = json.dumps(analysis, ensure_ascii=False)
        if normalized_evidence in _normalized(payload):
            return True
        message = analysis.get("message")
        if isinstance(message, str) and _normalized(message) in normalized_evidence:
            return True
    return False


def validate_file(finding: ReviewFinding, context: ReviewContext) -> None:
    """Reject findings for files not present in the changed-code context."""
    if not context.changed_code:
        return
    file_lookup = _canonical_file_lookup(context.changed_code)
    if _normalize_path(finding.file) not in file_lookup:
        raise ReviewResponseError(
            f"Finding refers to file {finding.file!r}, which is not in changed_code."
        )


def validate_line(finding: ReviewFinding, context: ReviewContext) -> None:
    """Reject findings for lines outside the supplied changed-line map."""
    if context.changed_lines is None or not context.changed_lines:
        return
    file_lookup = _canonical_file_lookup(context.changed_lines)
    changed_lines = file_lookup.get(_normalize_path(finding.file))
    if changed_lines is None or not changed_lines or finding.line not in changed_lines:
        raise ReviewResponseError(
            f"Finding at {finding.file}:{finding.line} is not listed as a changed line."
        )


def validate_evidence(finding: ReviewFinding, context: ReviewContext) -> None:
    """Reject unsupported findings when the evidence is absent from the relevant changed code."""
    if not _looks_meaningful_evidence(finding.evidence):
        raise ReviewResponseError(
            f"Finding evidence for {finding.file}:{finding.line} is too weak or generic to support the claim."
        )

    if not context.changed_code:
        return
    file_lookup = _canonical_file_lookup(context.changed_code)
    code = file_lookup.get(_normalize_path(finding.file), "")
    if not isinstance(code, str) or not code:
        return

    relevant_code = code
    if context.changed_lines:
        file_line_lookup = _canonical_file_lookup(context.changed_lines)
        changed_lines = file_line_lookup.get(_normalize_path(finding.file), [])
        if changed_lines and finding.line in changed_lines:
            lines = code.splitlines()
            if 1 <= finding.line <= len(lines):
                relevant_code = lines[finding.line - 1]

    if _normalized(finding.evidence) in _normalized(relevant_code):
        return
    if _analysis_matches_finding(finding, context.analysis_findings):
        return
    raise ReviewResponseError(
        f"Finding evidence for {finding.file}:{finding.line} does not match supplied changed code."
    )


def deduplicate_findings(findings: Sequence[ReviewFinding]) -> list[ReviewFinding]:
    """Remove obvious duplicates while preserving clearly distinct issues."""
    unique: list[ReviewFinding] = []
    for finding in findings:
        normalized_message = _normalize_message(finding.message)
        if not normalized_message:
            unique.append(finding)
            continue

        duplicate = False
        for existing in unique:
            if (
                finding.file != existing.file
                or finding.line != existing.line
                or finding.category.casefold() != existing.category.casefold()
            ):
                continue
            existing_message = _normalize_message(existing.message)
            if not existing_message:
                continue
            if normalized_message == existing_message:
                duplicate = True
                break
            if min(len(normalized_message), len(existing_message)) <= 5:
                continue
            if SequenceMatcher(None, normalized_message, existing_message).ratio() >= 0.85:
                duplicate = True
                break
        if not duplicate:
            unique.append(finding)
    return unique


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
        return deduplicate_findings(findings)

    @staticmethod
    def _validate_against_context(
        findings: list[ReviewFinding], context: ReviewContext
    ) -> None:
        """Reject findings for unknown files, unchanged lines, or unsupported code."""
        for finding in findings:
            validate_file(finding, context)
            validate_line(finding, context)
            validate_evidence(finding, context)
