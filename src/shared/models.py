from dataclasses import dataclass, field
from typing import List, Optional

from .enums import ReviewStatus


@dataclass
class PRContext:
    repository_owner: str
    repository_name: str
    pr_number: int
    title: str
    description: str
    base_sha: str
    head_sha: str
    issue_context: str = ""
    changed_files: List[str] = field(default_factory=list)
    diff: str = ""


@dataclass
class AnalysisFinding:
    file_path: str
    line_number: Optional[int]
    category: str
    severity: str
    message: str
    evidence: str
    suggestion: str


@dataclass
class ReviewResult:
    pr_number: int
    findings: List[AnalysisFinding]
    summary: str
    status: ReviewStatus
