"""Publish mapped review comments to GitHub with duplicate prevention."""

from __future__ import annotations

from collections.abc import Sequence

from .client import GitHubClient
from .mapper import extract_fingerprints
from .models import InlineComment, PullRequestInfo, ReviewSummary


class ReviewPublisher:
    """Create a GitHub pull-request review, or describe what would be created."""

    def __init__(self, client: GitHubClient) -> None:
        self.client = client

    def existing_fingerprints(self, pr: PullRequestInfo) -> set[str]:
        comments = self.client.list_review_comments(pr.owner, pr.repo, pr.number)
        bodies = [str(item.get("body") or "") for item in comments]
        return extract_fingerprints(bodies)

    def filter_duplicates(
        self,
        comments: Sequence[InlineComment],
        existing: set[str],
    ) -> tuple[list[InlineComment], list[InlineComment]]:
        unique: list[InlineComment] = []
        duplicates: list[InlineComment] = []
        seen = set(existing)
        for comment in comments:
            if comment.fingerprint in seen:
                duplicates.append(comment)
                continue
            unique.append(comment)
            seen.add(comment.fingerprint)
        return unique, duplicates

    def publish(
        self,
        pr: PullRequestInfo,
        comments: Sequence[InlineComment],
        summary: ReviewSummary,
    ) -> dict:
        """Submit one GitHub review containing the new inline comments.

        Dry-run mode never calls the write API.
        """
        if summary.dry_run:
            return {"dry_run": True, "comments": len(comments)}
        if not comments:
            return {"dry_run": False, "comments": 0, "skipped": True}

        payload = {
            "commit_id": pr.head_sha,
            "event": "COMMENT",
            "body": summary.to_markdown(),
            "comments": [
                {
                    "path": comment.path,
                    "body": comment.body,
                    "line": comment.line,
                    "side": comment.side,
                }
                for comment in comments
            ],
        }
        return self.client.create_pull_review(pr.owner, pr.repo, pr.number, payload)
