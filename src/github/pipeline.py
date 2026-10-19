import os
from collections import Counter
from pathlib import Path
from typing import List, Optional, Protocol, Union

from src.analysis import analyze_pr
from src.context import load_repository_rules
from src.shared.enums import ReviewStatus
from src.shared.models import AnalysisFinding, PRContext, ReviewResult

from .client import GitHubClient
from .comments import ReviewPublisher
from .models import PullRequestInfo, ReviewSummary


class Reviewer(Protocol):
    def review(
        self,
        context: PRContext,
        findings: List[AnalysisFinding],
        repository_rules: str = "",
    ) -> List[AnalysisFinding]:
        ...


def review_pr(
    context: PRContext,
    repository_root: Union[str, Path] = ".",
    reviewer: Optional[Reviewer] = None,
    *,
    repository_rules: Optional[str] = None,
    initial_findings: Optional[List[AnalysisFinding]] = None,
) -> ReviewResult:
    findings = (
        list(initial_findings)
        if initial_findings is not None
        else analyze_pr(context)
    )
    rules = (
        load_repository_rules(repository_root)
        if repository_rules is None
        else repository_rules
    )
    status = ReviewStatus.COMPLETE
    reviewer_error: Optional[str] = None
    semantic_scope_partial = False
    if reviewer is not None:
        try:
            findings = reviewer.review(context, findings, rules)
            reviewer_failures = tuple(getattr(reviewer, "review_failures", ()))
            if reviewer_failures:
                status = ReviewStatus.PARTIAL
                reviewer_error = " ".join(reviewer_failures)
                pipeline = getattr(reviewer, "pipeline", None)
                semantic_scope_partial = (
                    getattr(pipeline, "last_failed_batch_count", None) == 0
                    and bool(
                        getattr(reviewer, "unreviewed_files", ())
                        or getattr(reviewer, "partially_reviewed_files", ())
                    )
                )
        except RuntimeError as error:
            status = ReviewStatus.PARTIAL
            reviewer_error = str(error)

    summary = f"Review completed with {len(findings)} finding(s)."
    if status is ReviewStatus.PARTIAL:
        if semantic_scope_partial:
            summary += (
                f" Semantic review was partial: {reviewer_error}. "
                "Static findings are included."
            )
        else:
            summary += (
                f" Semantic review failed: {reviewer_error}. "
                "Static findings are included."
            )

    severity_counts = dict(
        Counter(item.severity.title() for item in findings if item.severity)
    )
    has_blocking = any(
        item.severity.casefold() in {"high", "critical"} for item in findings
    )

    return ReviewResult(
        pr_number=context.pr_number,
        findings=findings,
        summary=summary,
        status=status,
        head_sha=context.head_sha,
        comments=[],
        block_merge=has_blocking,
        severity_counts=severity_counts,
    )


def publish_review(
    pr_context: PRContext,
    review: ReviewResult,
    *,
    client: Optional[GitHubClient] = None,
    dry_run: Optional[bool] = None,
) -> None:
    """Publish a ReviewResult to GitHub based on PRContext and established contracts.

    Guards against stale PR revisions, respects dry_run, and prevents duplicate publications.
    """
    if pr_context.pr_number != review.pr_number:
        raise ValueError(
            f"PR number mismatch: pr_context has #{pr_context.pr_number}, "
            f"review has #{review.pr_number}."
        )

    if (
        review.head_sha
        and pr_context.head_sha
        and review.head_sha != pr_context.head_sha
    ):
        raise ValueError(
            f"Stale pull request revision: review head SHA {review.head_sha} "
            f"does not match PR context head SHA {pr_context.head_sha}."
        )

    head_sha = review.head_sha or pr_context.head_sha
    token = os.environ.get("GITHUB_TOKEN", "").strip() or None
    is_dry_run = dry_run if dry_run is not None else (token is None and client is None)

    github_client = client or GitHubClient(token=token)
    pr_info = PullRequestInfo(
        owner=pr_context.repository_owner,
        repo=pr_context.repository_name,
        number=pr_context.pr_number,
        title=pr_context.title,
        body=pr_context.description,
        base_branch="",
        head_branch="",
        head_sha=head_sha,
        html_url="",
        state="open",
        base_sha=pr_context.base_sha,
    )

    summary = ReviewSummary(
        dry_run=is_dry_run,
        owner=pr_context.repository_owner,
        repo=pr_context.repository_name,
        pr_number=pr_context.pr_number,
        head_sha=head_sha,
        total_findings=len(review.findings),
        findings_by_severity=dict(review.severity_counts),
        block_merge=review.block_merge,
    )

    publisher = ReviewPublisher(github_client)
    comments = list(review.comments) if review.comments else []
    publisher.publish(pr_info, comments, summary)

