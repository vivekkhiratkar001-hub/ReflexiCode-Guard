"""Typed review input and deterministic context construction.

Repository rules and static-analysis findings are accepted as lists of structured
records. Until the other team interfaces are available, each record is represented
as a mapping (for example, ``{"rule_id": "R1", "description": "..."}``) or a
Pydantic model that can be dumped to a mapping.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


class ReviewContext(BaseModel):
    """All PR, issue, code, rule, and analysis context used for a review."""

    model_config = ConfigDict(extra="forbid", strict=True)

    pr_title: StrictStr = ""
    pr_description: StrictStr = ""
    issue_context: StrictStr = ""
    changed_code: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    repository_rules: list[dict[str, Any]] = Field(default_factory=list)
    analysis_findings: list[dict[str, Any]] = Field(default_factory=list)
    changed_lines: dict[StrictStr, list[int]] | None = None

    @field_validator("changed_code")
    @classmethod
    def validate_changed_code(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not path.strip() for path in value):
            raise ValueError("changed_code file paths must not be empty")
        return value


def _record_to_mapping(record: Any, field_name: str) -> dict[str, Any]:
    """Convert a team-provided structured object into a plain mapping."""
    if isinstance(record, Mapping):
        return dict(record)
    model_dump = getattr(record, "model_dump", None)
    if callable(model_dump):
        value = model_dump()
        if isinstance(value, Mapping):
            return dict(value)
    raise TypeError(
        f"Each {field_name} entry must be a mapping or a Pydantic model."
    )


class ReviewContextBuilder:
    """Build a predictable ReviewContext from pipeline input components."""

    def build(
        self,
        *,
        pr_title: str = "",
        pr_description: str = "",
        issue_context: str = "",
        changed_code: Mapping[str, str] | None = None,
        repository_rules: Sequence[Any] | None = None,
        analysis_findings: Sequence[Any] | None = None,
        changed_lines: Mapping[str, Sequence[int]] | None = None,
    ) -> ReviewContext:
        """Normalize provided inputs without fetching or inventing context."""
        rules = [
            _record_to_mapping(record, "repository_rules")
            for record in (repository_rules or ())
        ]
        findings = [
            _record_to_mapping(record, "analysis_findings")
            for record in (analysis_findings or ())
        ]
        line_map = (
            {path: list(lines) for path, lines in changed_lines.items()}
            if changed_lines is not None
            else None
        )
        return ReviewContext(
            pr_title=pr_title,
            pr_description=pr_description,
            issue_context=issue_context,
            changed_code=dict(changed_code or {}),
            repository_rules=rules,
            analysis_findings=findings,
            changed_lines=line_map,
        )