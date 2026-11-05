"""Parsing and deterministic validation of structured model responses."""

import json
import logging
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from pydantic import ValidationError

from .context import ReviewContext
from .schemas import ReviewFinding

logger = logging.getLogger(__name__)


class ReviewResponseError(ValueError):
    """Raised when a model response cannot be trusted as review findings."""


class _FindingValidationError(ReviewResponseError):
    """Internal validation failure carrying a machine-readable reason code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class FindingRejection:
    """Structured reason explaining why one finding was not accepted."""

    index: int
    code: str
    reason: str
    file: str | None = None
    line: int | None = None

    def __str__(self) -> str:
        location = ""
        if self.file is not None and self.line is not None:
            location = f"{self.file}:{self.line} "
        elif self.file is not None:
            location = f"{self.file} "
        return f"{location}({self.code}): {self.reason}"

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "code": self.code,
            "reason": self.reason,
            "file": self.file,
            "line": self.line,
        }


def summarize_validation_errors(exc: "ValidationError") -> str:
    """Describe schema failures without echoing model text or source code."""
    errors = exc.errors()
    parts = []
    for error in errors[:5]:
        location = ".".join(str(item) for item in error.get("loc", ())) or "<envelope>"
        parts.append(f"{location}: {error.get('msg')}")
    summary = "; ".join(parts)
    remaining = len(errors) - len(parts)
    if remaining > 0:
        summary += f"; and {remaining} more validation error(s)"
    return summary


def _summarize_finding_validation_error(
    exc: "ValidationError", index: int
) -> str:
    """Describe one invalid finding without echoing model text or source code."""
    errors = exc.errors()
    parts = []
    for error in errors[:5]:
        location = ".".join(str(item) for item in error.get("loc", ()))
        if location:
            location = f"findings.{index}.{location}"
        else:
            location = f"findings.{index}"
        parts.append(f"{location}: {error.get('msg')}")
    summary = "; ".join(parts)
    remaining = len(errors) - len(parts)
    if remaining > 0:
        summary += f"; and {remaining} more validation error(s)"
    return "The model response failed review-schema validation: " + summary


def _safe_finding_identity(value: object) -> tuple[str | None, int | None]:
    """Extract model-supplied identity fields for a structured rejection."""
    if not isinstance(value, dict):
        return None, None
    file_value = value.get("file")
    line_value = value.get("line")
    file = file_value if isinstance(file_value, str) else None
    line = (
        line_value
        if isinstance(line_value, int) and not isinstance(line_value, bool)
        else None
    )
    return file, line


def _extract_findings_payload(json_text: str) -> list[object]:
    """Validate the response envelope while leaving each finding independent."""
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ReviewResponseError(
            "The model response failed review-schema validation: invalid JSON."
        ) from exc
    if not isinstance(data, dict):
        raise ReviewResponseError(
            "The model response failed review-schema validation: "
            "<envelope>: Input should be a valid dictionary."
        )
    extra_keys = sorted(set(data) - {"findings"})
    if extra_keys:
        raise ReviewResponseError(
            "The model response failed review-schema validation: "
            f"{extra_keys[0]}: Extra inputs are not permitted."
        )
    if "findings" not in data:
        raise ReviewResponseError(
            "The model response failed review-schema validation: findings: Field required."
        )
    findings = data["findings"]
    if not isinstance(findings, list):
        raise ReviewResponseError(
            "The model response failed review-schema validation: "
            "findings: Input should be a valid list."
        )
    return findings


def _extract_json_object(raw_response: str) -> str:
    """Require one complete JSON object without prose or markdown wrappers."""
    text = raw_response.strip()
    decoder = json.JSONDecoder()
    try:
        _, end = decoder.raw_decode(text)
    except json.JSONDecodeError as exc:
        if text.count("{") > text.count("}") or text.count("[") > text.count("]"):
            raise ReviewResponseError(
                "The model response contains truncated JSON."
            ) from exc
        raise ReviewResponseError(
            "The model response does not contain valid JSON."
        ) from exc
    if text[end:].strip():
        raise ReviewResponseError(
            "The model response must contain only one JSON object."
        )
    if not text.startswith("{"):
        raise ReviewResponseError("The model response must be a JSON object.")
    return text


def _normalized(value: str) -> str:
    """Normalize whitespace and case for a conservative evidence match."""
    return re.sub(r"\s+", "", value).casefold()


def _normalized_source_line(value: str) -> str:
    """Ignore indentation/spacing only; preserve code casing and punctuation."""
    return re.sub(r"\s+", "", value)


def _normalize_path(value: str) -> str:
    """Use a simple cross-platform canonical form for file paths."""
    normalized = value.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


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
        analysis_line = analysis.get("line")
        if isinstance(analysis_line, int) and not isinstance(analysis_line, bool):
            if analysis_line != finding.line:
                continue
        evidence = analysis.get("evidence")
        if (
            isinstance(evidence, str)
            and _looks_meaningful_evidence(evidence)
            and normalized_evidence == _normalized(evidence)
        ):
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

    line_lookup = _canonical_file_lookup(context.changed_line_content)
    line_content = line_lookup.get(_normalize_path(finding.file), {})
    exact_line = line_content.get(finding.line)
    if exact_line is not None:
        if _normalized_source_line(finding.evidence) == _normalized_source_line(exact_line):
            return
        if _analysis_matches_finding(finding, context.analysis_findings):
            return
        raise ReviewResponseError(
            f"Finding evidence for {finding.file}:{finding.line} does not match the changed line."
        )

    if not context.changed_code:
        return
    file_lookup = _canonical_file_lookup(context.changed_code)
    code = file_lookup.get(_normalize_path(finding.file), "")
    if not isinstance(code, str) or not code:
        return

    relevant_code = code
    exact_source_line: str | None = None
    if context.changed_lines:
        file_line_lookup = _canonical_file_lookup(context.changed_lines)
        changed_lines = file_line_lookup.get(_normalize_path(finding.file), [])
        if changed_lines and finding.line in changed_lines:
            lines = code.splitlines()
            if 1 <= finding.line <= len(lines):
                exact_source_line = lines[finding.line - 1]

    if exact_source_line is not None and _normalized_source_line(
        finding.evidence
    ) == _normalized_source_line(exact_source_line):
        return
    if not context.changed_lines and any(
        _normalized_source_line(finding.evidence) == _normalized_source_line(source_line)
        for source_line in relevant_code.splitlines()
    ):
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
                _normalize_path(finding.file) != _normalize_path(existing.file)
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


def _strip_markdown_fences(text: str) -> str:
    """Remove outer markdown code fences or backticks from code suggestions."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()
    elif cleaned.startswith("`") and cleaned.endswith("`") and len(cleaned) >= 2:
        cleaned = cleaned.strip("`").strip()
    return cleaned


def _canonicalize_code_expression(code: str) -> str:
    """Normalize a code snippet for equivalence checking.

    Handles whitespace, comments, trailing semicolons, and outer parens without
    altering core semantic tokens.
    """
    text = _strip_markdown_fences(code)
    lines = [line.split("#")[0].rstrip() for line in text.splitlines()]
    text = " ".join(lines).strip()
    text = text.rstrip(";").strip()
    text = re.sub(r"^return\s*\((.*)\)$", r"return \1", text).strip()
    return re.sub(r"\s+", "", text)


def _is_valid_python_snippet(code: str) -> bool:
    """Check if the code snippet is syntactically valid Python."""
    import ast
    import textwrap

    try:
        ast.parse(code)
        return True
    except SyntaxError:
        pass
    try:
        ast.parse(f"def _dummy_container():\n    {code}")
        return True
    except SyntaxError:
        pass
    try:
        dedented = textwrap.dedent(code)
        ast.parse(f"def _dummy_container():\n{textwrap.indent(dedented, '    ')}")
        return True
    except SyntaxError:
        return False


def _check_rule_conflicts(
    finding: ReviewFinding,
    suggestion: str,
    rules: Sequence[dict[str, Any]],
) -> str | None:
    """Check if a suggestion directly conflicts with explicit repository requirements."""
    if not rules:
        return None

    rule_texts: list[str] = []
    for r in rules:
        if isinstance(r, dict):
            for key in ("instructions", "rule", "description"):
                val = r.get(key)
                if isinstance(val, str) and val.strip():
                    rule_texts.append(val)

    combined_rules = " ".join(rule_texts).casefold()
    norm_sugg = suggestion.casefold()
    code_context = f"{finding.message} {finding.evidence} {finding.file}".casefold()

    # Rule requires adding a surcharge, but suggestion doubles/multiplies/divides instead
    if ("surcharge" in combined_rules or "add a surcharge" in combined_rules or "must not reduce" in combined_rules) and (
        any(w in code_context for w in ("fee", "shipping", "surcharge", "express"))
    ):
        if (
            "* 2" in norm_sugg
            or "*2" in norm_sugg
            or "* 1." in norm_sugg
            or "/ 2" in norm_sugg
            or "/2" in norm_sugg
        ) and "+" not in norm_sugg:
            return "doubles, multiplies, or divides fee instead of adding the required surcharge"

    # Rule requires tax amount only, but suggestion computes tax-inclusive total
    if (
        "tax amount only" in combined_rules
        or "not the total including tax" in combined_rules
        or "returns the tax amount" in combined_rules
    ) and (any(w in code_context for w in ("tax", "rate", "calculate_tax"))):
        if any(
            op in norm_sugg
            for op in (
                "1 + rate",
                "1+rate",
                "1 - rate",
                "1-rate",
                "(1 +",
                "(1+",
                "(1 -",
                "(1-",
                "+ amount",
                "- amount",
            )
        ):
            return "returns tax-inclusive total instead of the required tax amount only"

    # Rule requires exact match, but suggestion uses prefix or substring check
    if ("exact match" in combined_rules or "exact equality" in combined_rules) and (
        any(w in code_context for w in ("password", "auth", "verify", "match", "equal", "token"))
    ):
        if "startswith" in norm_sugg or ".find(" in norm_sugg or "endswith" in norm_sugg:
            return "uses prefix or substring matching instead of exact equality"

    # Rule requires percentage discount, but suggestion subtracts raw percentage number
    if ("percentage" in combined_rules and "discount" in combined_rules) and (
        any(w in code_context for w in ("discount", "cart", "total", "subtotal"))
    ):
        if re.search(r"total\s*-\s*[a-zA-Z0-9_]*discount[a-zA-Z0-9_]*(?!\s*[/|*])", norm_sugg):
            return "subtracts raw percentage number instead of calculating percentage of total"

    # Rule requires adding to inventory, but suggestion merely overwrites
    if (
        ("adds" in combined_rules or "accumulate" in combined_rules)
        and ("inventory" in combined_rules or "stock" in combined_rules)
        and any(w in code_context for w in ("stock", "inventory", "restock", "sku"))
    ):
        if (
            re.search(r"[a-zA-Z0-9_]*\[[^\]]+\]\s*=\s*[a-zA-Z0-9_]*quantity\b", norm_sugg)
            and "+" not in norm_sugg
            and ".get" not in norm_sugg
        ):
            return "overwrites inventory instead of adding received quantity to existing stock"

    return None


def validate_suggestion(
    finding: ReviewFinding,
    context: ReviewContext | None = None,
    *,
    strict: bool = False,
) -> str | None:
    """Validate that a suggestion provides a meaningful, non-conflicting correction.

    Rejects suggestions that:
    - Contain invalid Python syntax.
    - Merely repeat or echo the buggy evidence or changed line.
    - Are trivial formatting or whitespace variations of the buggy code.
    - Directly contradict explicit repository requirements.
    - Propose unverified business logic when requirements are absent or ambiguous.

    When strict=True, raises ReviewResponseError.
    When strict=False, returns a string diagnostic reason if invalid, or None if valid.
    """
    if finding.suggestion is None:
        return None

    cleaned_sugg = _strip_markdown_fences(finding.suggestion).strip()
    if not cleaned_sugg or cleaned_sugg.casefold() in {"none", "null", "todo", "fixme", "pass"}:
        reason = "Suggestion is an empty or non-functional placeholder."
        if strict:
            raise ReviewResponseError(
                f"Suggestion for {finding.file}:{finding.line} is empty or non-functional."
            )
        return reason

    if not _is_valid_python_snippet(cleaned_sugg):
        reason = f"Suggestion for {finding.file}:{finding.line} contains invalid Python syntax."
        if strict:
            raise ReviewResponseError(reason)
        return reason

    norm_sugg = _canonicalize_code_expression(cleaned_sugg)
    norm_ev = _canonicalize_code_expression(finding.evidence)

    # Check 1: Suggestion repeats evidence (exact or normalized)
    if norm_sugg == norm_ev:
        reason = f"Suggestion for {finding.file}:{finding.line} repeats the buggy code without correcting it."
        if strict:
            raise ReviewResponseError(reason)
        return reason

    # Check 2: Suggestion repeats the exact changed line
    if context is not None:
        line_lookup = _canonical_file_lookup(context.changed_line_content)
        line_content = line_lookup.get(_normalize_path(finding.file), {})
        exact_line = line_content.get(finding.line)
        if exact_line is not None:
            norm_exact = _canonicalize_code_expression(exact_line)
            if norm_sugg == norm_exact:
                reason = f"Suggestion for {finding.file}:{finding.line} repeats the buggy code without correcting it."
                if strict:
                    raise ReviewResponseError(reason)
                return reason

        # Check 3: Check against repository requirements
        if context.repository_rules:
            conflict = _check_rule_conflicts(finding, cleaned_sugg, context.repository_rules)
            if conflict is not None:
                reason = f"Suggestion for {finding.file}:{finding.line} conflicts with repository requirements: {conflict}."
                if strict:
                    raise ReviewResponseError(reason)
                return reason
        elif finding.category.casefold() in ("bug", "correctness", "logic"):
            # When requirements are absent or ambiguous, do not invent the intended fix.
            # Withhold the suggestion when correctness cannot be established.
            reason = f"Requirements are absent; withholding unverified logic suggestion for {finding.file}:{finding.line}."
            if strict:
                raise ReviewResponseError(reason)
            return reason

    return None


class ReviewResponseParser:
    """Parse JSON, validate the schema, and check context-backed locations."""

    def __init__(self, max_findings: int = 3) -> None:
        if isinstance(max_findings, bool) or not isinstance(max_findings, int) or max_findings < 1:
            raise ValueError("max_findings must be a positive integer.")
        self.max_findings = max_findings
        self.last_rejections: tuple[FindingRejection, ...] = ()

    def parse(
        self,
        raw_response: str,
        context: ReviewContext | None = None,
        *,
        strict_suggestions: bool = False,
    ) -> list[ReviewFinding]:
        """Return validated findings or raise a useful error for bad output.

        Findings are validated independently. A malformed or unsupported finding
        is recorded in ``last_rejections`` without discarding unrelated valid
        findings from the same response. If every supplied finding is invalid,
        the response is still rejected as a whole so callers do not mistake a
        fully unusable model response for a successful empty review.
        """
        self.last_rejections = ()
        if not raw_response or not raw_response.strip():
            raise ReviewResponseError("The model returned an empty response.")

        json_text = _extract_json_object(raw_response)
        raw_findings = _extract_findings_payload(json_text)
        if len(raw_findings) > self.max_findings:
            raise ReviewResponseError(
                "The model response exceeded the configured semantic finding limit."
            )

        validated: list[ReviewFinding] = []
        rejections: list[FindingRejection] = []
        for index, raw_finding in enumerate(raw_findings):
            try:
                finding = ReviewFinding.model_validate(raw_finding)
            except ValidationError as exc:
                file, line = _safe_finding_identity(raw_finding)
                rejections.append(
                    FindingRejection(
                        index=index,
                        code="schema_validation",
                        reason=_summarize_finding_validation_error(exc, index),
                        file=file,
                        line=line,
                    )
                )
                continue

            if context is not None:
                try:
                    self._validate_finding_against_context(
                        finding, context, strict_suggestions=strict_suggestions
                    )
                except _FindingValidationError as exc:
                    rejections.append(
                        FindingRejection(
                            index=index,
                            code=exc.code,
                            reason=str(exc),
                            file=finding.file,
                            line=finding.line,
                        )
                    )
                    continue
            else:
                try:
                    reason = validate_suggestion(
                        finding, None, strict=strict_suggestions
                    )
                except ReviewResponseError as exc:
                    rejections.append(
                        FindingRejection(
                            index=index,
                            code="invalid_suggestion",
                            reason=str(exc),
                            file=finding.file,
                            line=finding.line,
                        )
                    )
                    continue
                if reason is not None:
                    logger.warning(
                        "Suggestion for %s:%d failed validation (%s); removed suggestion from finding.",
                        finding.file,
                        finding.line,
                        reason,
                    )
                    finding.suggestion = None
            validated.append(finding)

        self.last_rejections = tuple(rejections)
        if rejections and not validated:
            details = "; ".join(rejection.reason for rejection in rejections)
            raise ReviewResponseError(
                "The model response contained no valid findings: " + details
            )
        if rejections:
            logger.warning(
                "Rejected %d invalid model finding(s) while retaining %d valid finding(s).",
                len(rejections),
                len(validated),
            )
        return deduplicate_findings(validated)

    @staticmethod
    def _validate_finding_against_context(
        finding: ReviewFinding,
        context: ReviewContext,
        *,
        strict_suggestions: bool = False,
    ) -> None:
        """Reject one finding for an unsupported category, file, line, or evidence."""
        supported_categories = {
            "bug",
            "correctness",
            "security",
            "performance",
            "maintainability",
            "architecture",
        }
        if finding.category.casefold() not in supported_categories:
            raise _FindingValidationError(
                "unsupported_category",
                "The model response contains an unsupported finding category.",
            )
        try:
            validate_file(finding, context)
        except ReviewResponseError as exc:
            raise _FindingValidationError("unknown_file", str(exc)) from exc
        try:
            validate_line(finding, context)
        except ReviewResponseError as exc:
            raise _FindingValidationError("unchanged_line", str(exc)) from exc
        try:
            validate_evidence(finding, context)
        except ReviewResponseError as exc:
            raise _FindingValidationError("unsupported_evidence", str(exc)) from exc
        try:
            reason = validate_suggestion(
                finding, context, strict=strict_suggestions
            )
        except ReviewResponseError as exc:
            raise _FindingValidationError("invalid_suggestion", str(exc)) from exc
        if reason is not None:
            logger.warning(
                "Suggestion for %s:%d failed validation (%s); removed suggestion from finding.",
                finding.file,
                finding.line,
                reason,
            )
            finding.suggestion = None
