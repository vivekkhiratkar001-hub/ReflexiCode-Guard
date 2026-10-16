"""End-to-end Member 1 flow: GitHub PR → Member 3 pipeline → comments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from src.llm.context import ReviewContextBuilder
from src.llm.parser import ReviewResponseError
from src.llm.pipeline import ReviewPipeline
from src.llm.schemas import ReviewFinding

from .adapters import (
    NullRepositoryRulesProvider,
    NullStaticAnalysisProvider,
    RepositoryRulesProvider,
    StaticAnalysisProvider,
)
from .client import GitHubClient
from .collector import PullRequestCollector, prepare_review_inputs
from .comments import ReviewPublisher
from .exceptions import GitHubIntegrationError
from .mapper import FindingCommentMapper
from .models import InlineComment, PullRequestInfo, ReviewSummary


class ReviewOrchestrator:
    """Connect GitHub collection, Member 3 review, mapping, and publishing."""

    def __init__(
        self,
        client: GitHubClient,
        pipeline: ReviewPipeline,
        *,
        rules_provider: RepositoryRulesProvider | None = None,
        analysis_provider: StaticAnalysisProvider | None = None,
        publisher: ReviewPublisher | None = None,
        context_builder: ReviewContextBuilder | None = None,
        dry_run: bool = True,
    ) -> None:
        self.client = client
        self.pipeline = pipeline
        self.rules_provider = rules_provider or NullRepositoryRulesProvider()
        self.analysis_provider = analysis_provider or NullStaticAnalysisProvider()
        self.publisher = publisher or ReviewPublisher(client)
        self.context_builder = context_builder or ReviewContextBuilder()
        self.dry_run = dry_run
        self.collector = PullRequestCollector(client)

    def run(self, owner: str, repo: str, number: int) -> ReviewSummary:
        try:
            pr, files = self.collector.collect(owner, repo, number)
        except GitHubIntegrationError as exc:
            return ReviewSummary(
                dry_run=self.dry_run,
                owner=owner,
                repo=repo,
                pr_number=number,
                head_sha="",
                errors=[str(exc)],
            )

        changed_code, changed_lines, patches, skipped = prepare_review_inputs(files)
        summary = ReviewSummary(
            dry_run=self.dry_run,
            owner=pr.owner,
            repo=pr.repo,
            pr_number=pr.number,
            head_sha=pr.head_sha,
            skipped_files=skipped,
        )
        if not changed_code:
            summary.warnings.append(
                "No reviewable text diffs were found, so the LLM pipeline was not invoked."
            )
            return summary

        try:
            rules = self.rules_provider.get_rules(pr)
            analysis = self.analysis_provider.get_findings(pr, files, changed_code)
        except Exception as exc:  # noqa: BLE001 - adapters must not crash the run
            summary.errors.append(f"Supporting context provider failed: {exc}")
            rules, analysis = [], []
            summary.warnings.append(
                "Continuing without repository rules and/or static-analysis findings."
            )

        context = self.context_builder.build(
            pr_title=pr.title,
            pr_description=pr.body,
            issue_context="",
            changed_code=changed_code,
            repository_rules=rules,
            analysis_findings=analysis,
            changed_lines=changed_lines,
        )

        try:
            findings: Sequence[ReviewFinding] = self.pipeline.review(context)
        except ReviewResponseError as exc:
            summary.errors.append(f"Member 3 rejected the model response: {exc}")
            findings = []
        except Exception as exc:  # noqa: BLE001 - surface provider failures in the summary
            summary.errors.append(f"Review pipeline failed: {exc}")
            findings = []

        return self._complete(pr, files, patches, findings, summary)

    def _complete(
        self,
        pr: PullRequestInfo,
        files: list,
        patches: dict,
        findings: Sequence[ReviewFinding],
        summary: ReviewSummary,
    ) -> ReviewSummary:
        summary.total_findings = len(findings)
        summary.findings_by_severity = dict(Counter(item.severity for item in findings))

        comments, unmapped = FindingCommentMapper(files, patches).map_findings(findings)
        summary.unmapped_findings = unmapped
        summary.intended_comments = list(comments)

        try:
            existing = self.publisher.existing_fingerprints(pr)
        except GitHubIntegrationError as exc:
            message = f"Could not load existing review comments for duplicate detection: {exc}"
            if not summary.dry_run:
                summary.errors.append(
                    f"Aborting publication to avoid creating duplicate comments. {message}"
                )
                return summary
            summary.warnings.append(message)
            existing = set()

        unique, duplicates = self.publisher.filter_duplicates(comments, existing)
        summary.comments_skipped_duplicate = len(duplicates)
        summary.comments_intended = len(unique)
        summary.intended_comments = unique

        try:
            result = self.publisher.publish(pr, unique, summary)
        except GitHubIntegrationError as exc:
            summary.errors.append(f"Publishing review comments failed: {exc}")
            return summary

        if summary.dry_run:
            summary.comments_published = 0
        else:
            published = result.get("comments") if isinstance(result, dict) else None
            summary.comments_published = int(published if published is not None else len(unique))
        return summary


def comments_preview(comments: Sequence[InlineComment]) -> str:
    if not comments:
        return "No inline comments would be published."
    blocks = []
    for comment in comments:
        blocks.append(
            f"- {comment.path}:{comment.line} ({comment.side}) fingerprint={comment.fingerprint}\n"
            f"{comment.body}"
        )
    return "\n\n".join(blocks)
