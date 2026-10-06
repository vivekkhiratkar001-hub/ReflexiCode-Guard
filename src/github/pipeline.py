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
) -> ReviewResult:
    findings = analyze_pr(context)
    rules = load_repository_rules(repository_root)
    status = ReviewStatus.COMPLETE
    reviewer_error: Optional[str] = None
    if reviewer is not None:
        try:
            findings = reviewer.review(context, findings, rules)
        except RuntimeError as error:
            status = ReviewStatus.PARTIAL
            reviewer_error = str(error)

    summary = f"Review completed with {len(findings)} finding(s)."
    if status is ReviewStatus.PARTIAL:
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
