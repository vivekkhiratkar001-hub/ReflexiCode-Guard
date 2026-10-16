"""Adapters between GitHub's API models and the shared review contracts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from src.analysis import analyze_pr
from src.context import load_repository_rules
from src.llm.context import ReviewContextBuilder
from src.llm.parser import ReviewResponseError
from src.llm.pipeline import ReviewPipeline
from src.llm.schemas import ReviewFinding
from src.shared.enums import FindingCategory
from src.shared.models import AnalysisFinding, PRContext

from .models import FileChange, ParsedPatch, PullRequestInfo


class RepositoryRulesProvider(Protocol):
    def get_rules(self, pr: PullRequestInfo) -> list[dict[str, Any]]:
        """Return repository rules as mappings accepted by ReviewContextBuilder."""


class StaticAnalysisProvider(Protocol):
    def get_findings(
        self,
        pr: PullRequestInfo,
        files: Sequence[FileChange],
        changed_code: dict[str, str],
    ) -> list[AnalysisFinding | dict[str, Any]]:
        """Return shared analysis findings or mappings from an explicit fixture."""


class NullRepositoryRulesProvider:
    """Used when Member 2 rules are not available."""

    def get_rules(self, pr: PullRequestInfo) -> list[dict[str, Any]]:
        return []


class LocalRepositoryRulesProvider:
    """Load plain-text repository guidance through the shared context module."""

    def __init__(self, repository_root: Path = Path(".")) -> None:
        self.repository_root = repository_root

    def get_rules(self, pr: PullRequestInfo) -> list[dict[str, Any]]:
        rules = load_repository_rules(self.repository_root)
        return [{"instructions": rules}] if rules else []


class NullStaticAnalysisProvider:
    """Used when Member 2 static analysis is not available."""

    def get_findings(
        self,
        pr: PullRequestInfo,
        files: Sequence[FileChange],
        changed_code: dict[str, str],
    ) -> list[dict[str, Any]]:
        return []


def build_shared_pr_context(
    pr: PullRequestInfo,
    files: Sequence[FileChange],
) -> PRContext:
    """Adapt GitHub file patches to the shared analyzer's unified-diff input."""
    diff_parts: list[str] = []
    for file_change in files:
        if not file_change.reviewable or not file_change.patch:
            continue
        old_path = file_change.previous_path or file_change.path
        new_path = file_change.path
        old_header = "/dev/null" if file_change.status == "added" else f"a/{old_path}"
        new_header = "/dev/null" if file_change.status == "removed" else f"b/{new_path}"
        diff_parts.extend(
            (
                f"diff --git a/{old_path} b/{new_path}\n",
                f"--- {old_header}\n",
                f"+++ {new_header}\n",
                file_change.patch.rstrip("\n") + "\n",
            )
        )
    return PRContext(
        repository_owner=pr.owner,
        repository_name=pr.repo,
        pr_number=pr.number,
        title=pr.title,
        description=pr.body,
        base_sha=pr.base_sha or "",
        head_sha=pr.head_sha,
        changed_files=[item.path for item in files if item.reviewable],
        diff="".join(diff_parts),
    )


class SharedStaticAnalysisProvider:
    """Run Member 2's deterministic analyzer against collected GitHub patches."""

    def get_findings(
        self,
        pr: PullRequestInfo,
        files: Sequence[FileChange],
        changed_code: dict[str, str],
    ) -> list[AnalysisFinding]:
        return analyze_pr(build_shared_pr_context(pr, files))


class SharedContractReviewer:
    """Adapt Member 3's validated semantic review to the shared review contract."""

    def __init__(
        self,
        pipeline: ReviewPipeline,
        changed_code: dict[str, str],
        changed_lines: dict[str, list[int]],
        changed_line_content: dict[str, dict[int, str]],
        context_builder: ReviewContextBuilder | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.changed_code = changed_code
        self.changed_lines = changed_lines
        self.changed_line_content = changed_line_content
        self.context_builder = context_builder or ReviewContextBuilder()

    def review(
        self,
        context: PRContext,
        findings: list[AnalysisFinding],
        repository_rules: str = "",
    ) -> list[AnalysisFinding]:
        try:
            decoded_rules = json.loads(repository_rules) if repository_rules else []
        except json.JSONDecodeError:
            decoded_rules = [{"instructions": repository_rules}]
        if not isinstance(decoded_rules, list) or any(
            not isinstance(rule, dict) for rule in decoded_rules
        ):
            decoded_rules = [{"instructions": repository_rules}]

        llm_context = self.context_builder.build(
            pr_title=context.title,
            pr_description=context.description,
            issue_context=context.issue_context,
            changed_code=self.changed_code,
            repository_rules=decoded_rules,
            analysis_findings=[
                {
                    "file": finding.file_path,
                    "line": finding.line_number,
                    "category": finding.category,
                    "severity": finding.severity,
                    "message": finding.message,
                    "evidence": finding.evidence,
                    "suggestion": finding.suggestion,
                }
                for finding in findings
            ],
            changed_lines=self.changed_lines,
            changed_line_content=self.changed_line_content,
        )
        try:
            semantic_findings = self.pipeline.review(llm_context)
        except ReviewResponseError as error:
            raise RuntimeError(f"Member 3 rejected the model response: {error}") from error

        static_evidence = {
            (
                finding.file_path.replace("\\", "/").casefold(),
                finding.line_number,
                "".join(finding.evidence.split()).casefold(),
            )
            for finding in findings
        }
        unique_semantic = [
            finding
            for finding in semantic_findings
            if (
                finding.file.replace("\\", "/").casefold(),
                finding.line,
                "".join(finding.evidence.split()).casefold(),
            )
            not in static_evidence
        ]
        return findings + [
            AnalysisFinding(
                file_path=finding.file,
                line_number=finding.line,
                category=finding.category.casefold()
                or FindingCategory.SEMANTIC.value,
                severity=finding.severity.casefold(),
                message=finding.message,
                evidence=finding.evidence,
                suggestion=finding.suggestion or "",
            )
            for finding in unique_semantic
        ]


def build_changed_line_content(
    files: Sequence[FileChange],
    patches: dict[str, ParsedPatch],
) -> dict[str, dict[int, str]]:
    """Preserve patch line numbers for strict semantic-evidence validation."""
    result: dict[str, dict[int, str]] = {}
    for file_change in files:
        patch = patches.get(file_change.path)
        if patch is None:
            continue
        if file_change.status == "removed":
            line_content = {
                line.old_lineno: line.content
                for line in patch.lines
                if line.kind == "delete" and line.old_lineno is not None
            }
        else:
            line_content = {
                line.new_lineno: line.content
                for line in patch.lines
                if line.kind in {"context", "add"} and line.new_lineno is not None
            }
        result[file_change.path] = line_content
    return result


def normalize_analysis_findings(
    records: Sequence[AnalysisFinding | dict[str, Any]],
) -> list[AnalysisFinding]:
    """Validate explicit analysis fixtures against the shared finding schema."""
    normalized: list[AnalysisFinding] = []
    for record in records:
        if isinstance(record, AnalysisFinding):
            normalized.append(record)
            continue
        if not isinstance(record, dict):
            raise TypeError("Analysis findings must be shared findings or JSON objects.")
        values = dict(record)
        if "file_path" not in values and "file" in values:
            values["file_path"] = values.pop("file")
        if "line_number" not in values and "line" in values:
            values["line_number"] = values.pop("line")
        values.pop("rule", None)
        values.pop("rule_id", None)
        if values.get("suggestion") is None:
            values["suggestion"] = ""
        values.setdefault("suggestion", "")
        required_strings = (
            "file_path",
            "category",
            "severity",
            "message",
            "evidence",
            "suggestion",
        )
        if any(not isinstance(values.get(name), str) for name in required_strings):
            raise TypeError("Analysis finding fields must match the shared contract.")
        line_number = values.get("line_number")
        if line_number is not None and (
            not isinstance(line_number, int) or isinstance(line_number, bool)
        ):
            raise TypeError("Analysis finding line_number must be an integer or null.")
        if (
            not values["file_path"].strip()
            or not values["category"].strip()
            or not values["message"].strip()
            or values["severity"].casefold()
            not in {"low", "medium", "high", "critical"}
        ):
            raise TypeError("Analysis finding contains an empty or unsupported field.")
        normalized.append(AnalysisFinding(**values))
    return normalized


class FileRepositoryRulesProvider:
    """Load Member-2-shaped rules from a JSON list of objects."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get_rules(self, pr: PullRequestInfo) -> list[dict[str, Any]]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"Rules file {self.path} must contain a JSON list.")
        if any(not isinstance(item, dict) for item in payload):
            raise TypeError(f"Every rule in {self.path} must be a JSON object.")
        return [dict(item) for item in payload]


class FileStaticAnalysisProvider:
    """Load Member-2-shaped static findings from a JSON list of objects."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get_findings(
        self,
        pr: PullRequestInfo,
        files: Sequence[FileChange],
        changed_code: dict[str, str],
    ) -> list[dict[str, Any]]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"Analysis file {self.path} must contain a JSON list.")
        if any(not isinstance(item, dict) for item in payload):
            raise TypeError(f"Every finding in {self.path} must be a JSON object.")
        return [dict(item) for item in payload]


def select_rules_provider(rules_file: Path | None) -> RepositoryRulesProvider:
    if rules_file is None:
        return LocalRepositoryRulesProvider()
    return FileRepositoryRulesProvider(rules_file)


def select_analysis_provider(analysis_file: Path | None) -> StaticAnalysisProvider:
    if analysis_file is None:
        return SharedStaticAnalysisProvider()
    return FileStaticAnalysisProvider(analysis_file)
