"""Temporary adapters for Member 2 (and future Member 4) inputs.

Member 2's repository-rules and static-analysis modules are not in this
repository yet. These adapters let Member 1 run locally and in tests without
inventing a second analysis engine.

When Member 2 publishes a real Python API, replace the file-backed providers
with thin wrappers around that API. Do not call these adapters from Member 3.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from .models import FileChange, PullRequestInfo


class RepositoryRulesProvider(Protocol):
    def get_rules(self, pr: PullRequestInfo) -> list[dict[str, Any]]:
        """Return repository rules as mappings accepted by ReviewContextBuilder."""


class StaticAnalysisProvider(Protocol):
    def get_findings(
        self,
        pr: PullRequestInfo,
        files: Sequence[FileChange],
        changed_code: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Return static-analysis findings as mappings accepted by ReviewContextBuilder."""


class NullRepositoryRulesProvider:
    """Used when Member 2 rules are not available."""

    def get_rules(self, pr: PullRequestInfo) -> list[dict[str, Any]]:
        return []


class NullStaticAnalysisProvider:
    """Used when Member 2 static analysis is not available."""

    def get_findings(
        self,
        pr: PullRequestInfo,
        files: Sequence[FileChange],
        changed_code: dict[str, str],
    ) -> list[dict[str, Any]]:
        return []


class FileRepositoryRulesProvider:
    """Load Member-2-shaped rules from a JSON list of objects."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get_rules(self, pr: PullRequestInfo) -> list[dict[str, Any]]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"Rules file {self.path} must contain a JSON list.")
        return [dict(item) for item in payload if isinstance(item, dict)]


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
        return [dict(item) for item in payload if isinstance(item, dict)]


def select_rules_provider(rules_file: Path | None) -> RepositoryRulesProvider:
    if rules_file is None:
        return NullRepositoryRulesProvider()
    return FileRepositoryRulesProvider(rules_file)


def select_analysis_provider(analysis_file: Path | None) -> StaticAnalysisProvider:
    if analysis_file is None:
        return NullStaticAnalysisProvider()
    return FileStaticAnalysisProvider(analysis_file)
