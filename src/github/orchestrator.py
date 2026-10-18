"""End-to-end Member 1 flow: GitHub PR → Member 3 pipeline → comments."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from src.analysis.rules.security_rules import redact_secret_text
from src.llm.context import ReviewContextBuilder
from src.llm.pipeline import ReviewPipeline
from src.llm.schemas import ReviewFinding
from src.shared.enums import ReviewStatus
from src.shared.models import AnalysisFinding, PRContext

from .adapters import (
    LocalRepositoryRulesProvider,
    SharedContractReviewer,
    SharedStaticAnalysisProvider,
    RepositoryRulesProvider,
    StaticAnalysisProvider,
    build_changed_line_content,
    build_shared_pr_context,
    normalize_analysis_findings,
)
from .client import GitHubClient
from .collector import PullRequestCollector, prepare_review_inputs
from .comments import ReviewPublisher
from .exceptions import GitHubIntegrationError
from .mapper import FindingCommentMapper
from .models import InlineComment, PullRequestInfo, ReviewSummary
from .pipeline import review_pr as run_shared_review


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
        repository_root: Path = Path("."),
        dry_run: bool = True,
    ) -> None:
        self.client = client
        self.pipeline = pipeline
        self.rules_provider = rules_provider or LocalRepositoryRulesProvider(repository_root)
        self.analysis_provider = analysis_provider or SharedStaticAnalysisProvider()
        self.publisher = publisher or ReviewPublisher(client)
        self.context_builder = context_builder
        self.repository_root = repository_root
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
            analysis = normalize_analysis_findings(
                self.analysis_provider.get_findings(pr, files, changed_code)
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            summary.errors.append(f"Supporting context provider failed: {exc}")
            rules, analysis = [], []
            summary.warnings.append(
                "Continuing without repository rules and/or static-analysis findings."
            )

        shared_context = build_shared_pr_context(pr, files)
        reviewer = SharedContractReviewer(
            self.pipeline,
            changed_code,
            changed_lines,
            build_changed_line_content(files, patches),
            self.context_builder,
        )
        review_result = run_shared_review(
            shared_context,
            repository_root=self.repository_root,
            reviewer=reviewer,
            repository_rules=json.dumps(rules, ensure_ascii=False),
            initial_findings=analysis,
        )
        if reviewer.context_limitations:
            summary.warnings.append(
                "Semantic review context was prioritized and bounded: "
                + " ".join(reviewer.context_limitations)
            )
        summary.reviewed_files = list(reviewer.reviewed_files)
        summary.unreviewed_files = list(reviewer.unreviewed_files)
        summary.semantic_batches_total = self.pipeline.last_batch_count
        summary.semantic_batches_failed = self.pipeline.last_failed_batch_count
        summary.semantic_findings = self.pipeline.last_findings_count
        summary.static_findings = len(analysis)
        if review_result.status is ReviewStatus.PARTIAL:
            if self.pipeline.last_failed_batch_count:
                summary.semantic_status = (
                    "partial_failure" if analysis else "failed"
                )
            else:
                summary.semantic_status = "partial"
        elif summary.semantic_findings:
            summary.semantic_status = "success_with_findings"
        else:
            summary.semantic_status = "success_no_findings"
        if self.pipeline.last_batch_count > 1:
            summary.warnings.append(
                f"Semantic review was split into {self.pipeline.last_batch_count} bounded batches."
            )
        if review_result.status is ReviewStatus.PARTIAL:
            summary.errors.append(review_result.summary)
        findings = self._to_publishable_findings(review_result.findings, summary)

        return self._complete(pr, files, patches, findings, summary)

    @staticmethod
    def _to_publishable_findings(
        findings: Sequence[AnalysisFinding],
        summary: ReviewSummary,
    ) -> list[ReviewFinding]:
        severity_names = {
            "low": "Low",
            "medium": "Medium",
            "high": "High",
            "critical": "Critical",
        }
        category_names = {
            "security": "Security",
            "static_analysis": "Static Analysis",
            "semantic": "Semantic",
        }
        converted: list[ReviewFinding] = []
        for finding in findings:
            if finding.line_number is None or finding.line_number < 1:
                summary.unmapped_findings.append(
                    f"{finding.file_path}:{finding.line_number} ({finding.category})"
                )
                continue
            if not finding.evidence.strip():
                summary.unmapped_findings.append(
                    f"{finding.file_path}:{finding.line_number} "
                    f"({finding.category}; missing evidence)"
                )
                continue
            severity = severity_names.get(finding.severity.casefold())
            if severity is None:
                summary.errors.append(
                    f"Finding {finding.file_path}:{finding.line_number} has unsupported "
                    f"severity {finding.severity!r} and was not published."
                )
                continue
            category = category_names.get(
                finding.category.casefold(),
                finding.category.replace("_", " ").title(),
            )
            try:
                converted.append(
                    ReviewFinding.model_validate(
                        {
                            "file": finding.file_path,
                            "line": finding.line_number,
                            "category": category,
                            "severity": severity,
                            "message": redact_secret_text(finding.message),
                            "evidence": redact_secret_text(finding.evidence),
                            "suggestion": (
                                redact_secret_text(finding.suggestion)
                                if finding.suggestion
                                else None
                            ),
                        }
                    )
                )
            except ValidationError as error:
                summary.errors.append(
                    f"Invalid shared finding {finding.file_path}:{finding.line_number}: "
                    f"{error}"
                )
        return converted

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
        summary.unmapped_findings.extend(unmapped)
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
