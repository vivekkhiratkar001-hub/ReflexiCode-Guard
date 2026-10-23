"""Plain data models used by the GitHub integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.shared.models import AnalysisFinding


CommentSide = Literal["LEFT", "RIGHT"]
FileStatus = Literal[
    "added",
    "removed",
    "modified",
    "renamed",
    "copied",
    "changed",
    "unchanged",
    "unknown",
]


@dataclass(frozen=True)
class PullRequestRef:
    """Identifies one pull request on GitHub."""

    owner: str
    repo: str
    number: int


@dataclass(frozen=True)
class PullRequestInfo:
    """Metadata collected from the GitHub Pulls API."""

    owner: str
    repo: str
    number: int
    title: str
    body: str
    base_branch: str
    head_branch: str
    head_sha: str
    html_url: str
    state: str
    base_sha: str | None = None


@dataclass
class FileChange:
    """One changed file in a pull request."""

    path: str
    status: str
    previous_path: str | None = None
    patch: str | None = None
    additions: int = 0
    deletions: int = 0
    changes: int = 0
    sha: str | None = None
    reviewable: bool = True
    skip_reason: str | None = None


@dataclass(frozen=True)
class HunkLine:
    """One line inside a unified-diff hunk."""

    kind: Literal["context", "add", "delete"]
    content: str
    old_lineno: int | None
    new_lineno: int | None


@dataclass
class ParsedPatch:
    """Parsed unified diff for a single file."""

    lines: list[HunkLine] = field(default_factory=list)

    def added_line_map(self) -> dict[int, str]:
        return {
            line.new_lineno: line.content
            for line in self.lines
            if line.kind == "add" and line.new_lineno is not None
        }

    def deleted_line_map(self) -> dict[int, str]:
        return {
            line.old_lineno: line.content
            for line in self.lines
            if line.kind == "delete" and line.old_lineno is not None
        }

    def right_commentable_lines(self) -> set[int]:
        """GitHub RIGHT-side lines that appear in the diff (context + added)."""
        return {
            line.new_lineno
            for line in self.lines
            if line.new_lineno is not None and line.kind in {"context", "add"}
        }

    def left_commentable_lines(self) -> set[int]:
        """GitHub LEFT-side lines that appear in the diff (context + deleted)."""
        return {
            line.old_lineno
            for line in self.lines
            if line.old_lineno is not None and line.kind in {"context", "delete"}
        }


@dataclass(frozen=True)
class InlineComment:
    """A GitHub review comment that is ready to publish."""

    path: str
    line: int
    side: CommentSide
    body: str
    fingerprint: str
    finding_file: str
    finding_line: int


@dataclass
class ReviewSummary:
    """Human-readable result of one review run."""

    dry_run: bool
    owner: str
    repo: str
    pr_number: int
    head_sha: str
    total_findings: int = 0
    findings_by_severity: dict[str, int] = field(default_factory=dict)
    findings: list[AnalysisFinding] = field(default_factory=list)
    comments_intended: int = 0
    comments_published: int = 0
    comments_skipped_duplicate: int = 0
    unmapped_findings: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    reviewed_files: list[str] = field(default_factory=list)
    partially_reviewed_files: list[str] = field(default_factory=list)
    unreviewed_files: list[str] = field(default_factory=list)
    semantic_batches_total: int = 0
    semantic_batches_failed: int = 0
    semantic_status: str = "not_run"
    semantic_findings: int = 0
    static_findings: int = 0
    semantic_elapsed_seconds: float = 0.0
    semantic_budget_seconds: float = 240.0
    block_merge: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    intended_comments: list[InlineComment] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render a summary suitable for the console or GitHub Actions."""
        mode = "DRY RUN (no GitHub comments were created)" if self.dry_run else "PUBLISH"
        published_label = (
            "Comments that would be published"
            if self.dry_run
            else "Comments published"
        )
        severity_lines = (
            "\n".join(
                f"- {name}: {count}"
                for name, count in sorted(self.findings_by_severity.items())
            )
            or "- none"
        )
        unmapped = "\n".join(f"- {item}" for item in self.unmapped_findings) or "- none"
        skipped = "\n".join(f"- {item}" for item in self.skipped_files) or "- none"
        reviewed = "\n".join(f"- {item}" for item in self.reviewed_files) or "- none"
        partially_reviewed = (
            "\n".join(f"- {item}" for item in self.partially_reviewed_files)
            or "- none"
        )
        unreviewed = "\n".join(f"- {item}" for item in self.unreviewed_files) or "- none"
        errors = "\n".join(f"- {item}" for item in self.errors) or "- none"
        warnings = "\n".join(f"- {item}" for item in self.warnings) or "- none"
        return f"""## ReflexiCode-Guard review summary

- Mode: {mode}
- Pull request: `{self.owner}/{self.repo}#{self.pr_number}`
- Head SHA: `{self.head_sha}`
- Total findings: {self.total_findings}
- Block merge: {"yes" if self.block_merge else "no"}
- {published_label}: {self.comments_intended if self.dry_run else self.comments_published}
- Duplicates skipped: {self.comments_skipped_duplicate}
- Unmapped findings: {len(self.unmapped_findings)}

### Findings by severity
{severity_lines}

### Unmapped findings
{unmapped}

### Files skipped (binary / missing patch / unsupported)
{skipped}

### Semantic review coverage
- Semantic status: {self.semantic_status}
- Semantic findings: {self.semantic_findings}
- Deterministic findings: {self.static_findings}
- Batches: {self.semantic_batches_total}
- Failed batches: {self.semantic_batches_failed}
- Elapsed: {self.semantic_elapsed_seconds:.2f}s of {self.semantic_budget_seconds:.0f}s

#### Fully reviewed files
{reviewed}

#### Partially reviewed files
{partially_reviewed}

#### Not reviewed files
{unreviewed}

### Warnings
{warnings}

### Errors
{errors}
"""
