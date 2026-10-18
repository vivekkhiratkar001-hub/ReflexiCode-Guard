from pathlib import Path
from typing import List, Optional, Protocol, Union

from src.analysis import analyze_pr
from src.context import load_repository_rules
from src.shared.enums import ReviewStatus
from src.shared.models import AnalysisFinding, PRContext, ReviewResult


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
                    and bool(getattr(reviewer, "unreviewed_files", ()))
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
    return ReviewResult(
        pr_number=context.pr_number,
        findings=findings,
        summary=summary,
        status=status,
    )
