"""Collect pull-request metadata and diffs from GitHub."""

from __future__ import annotations

from typing import Any

from .client import GitHubClient
from .diff import build_source_text, is_probably_binary_path, parse_patch
from .exceptions import GitHubAPIError
from .models import FileChange, ParsedPatch, PullRequestInfo


class PullRequestCollector:
    """Turn GitHub API payloads into review-ready structures."""

    def __init__(self, client: GitHubClient) -> None:
        self.client = client

    def collect(self, owner: str, repo: str, number: int) -> tuple[PullRequestInfo, list[FileChange]]:
        raw_pr = self.client.get_pull_request(owner, repo, number)
        info = self._parse_pull_request(owner, repo, number, raw_pr)
        raw_files = self.client.list_pull_request_files(owner, repo, number)
        files = [self._parse_file(item) for item in raw_files]
        return info, files

    @staticmethod
    def _parse_pull_request(
        owner: str,
        repo: str,
        number: int,
        payload: dict[str, Any],
    ) -> PullRequestInfo:
        base = payload.get("base") if isinstance(payload.get("base"), dict) else {}
        head = payload.get("head") if isinstance(payload.get("head"), dict) else {}
        sha = str(head.get("sha") or "")
        if not sha:
            raise GitHubAPIError("Pull request payload is missing the head commit SHA.")
        title = payload.get("title")
        body = payload.get("body")
        return PullRequestInfo(
            owner=owner,
            repo=repo,
            number=int(payload.get("number") or number),
            title=str(title or ""),
            body=str(body or ""),
            base_branch=str(base.get("ref") or ""),
            head_branch=str(head.get("ref") or ""),
            head_sha=sha,
            html_url=str(payload.get("html_url") or ""),
            state=str(payload.get("state") or ""),
            base_sha=str(base.get("sha")) if base.get("sha") else None,
        )

    @staticmethod
    def _parse_file(payload: dict[str, Any]) -> FileChange:
        path = str(payload.get("filename") or "").strip()
        status = str(payload.get("status") or "unknown")
        previous = payload.get("previous_filename")
        patch = payload.get("patch")
        patch_text = patch if isinstance(patch, str) and patch.strip() else None
        skip_reason = None
        reviewable = True
        if not path:
            reviewable = False
            skip_reason = "GitHub omitted the file path."
        elif is_probably_binary_path(path):
            reviewable = False
            skip_reason = "Binary or unsupported file type."
        elif patch_text is None:
            reviewable = False
            skip_reason = "No patch is available (binary, too large, or empty diff)."
        return FileChange(
            path=path or "(unknown)",
            status=status,
            previous_path=str(previous) if previous else None,
            patch=patch_text,
            additions=int(payload.get("additions") or 0),
            deletions=int(payload.get("deletions") or 0),
            changes=int(payload.get("changes") or 0),
            sha=str(payload.get("sha")) if payload.get("sha") else None,
            reviewable=reviewable,
            skip_reason=skip_reason,
        )


def prepare_review_inputs(
    files: list[FileChange],
) -> tuple[dict[str, str], dict[str, list[int]], dict[str, ParsedPatch], list[str]]:
    """Build Member 3 changed_code / changed_lines from reviewable patches.

    Only changed hunks are sent to the LLM. Binary files and files without a
    patch are skipped.
    """
    changed_code: dict[str, str] = {}
    changed_lines: dict[str, list[int]] = {}
    patches: dict[str, ParsedPatch] = {}
    skipped: list[str] = []

    for file_change in files:
        if not file_change.reviewable or not file_change.patch:
            reason = file_change.skip_reason or "not reviewable"
            skipped.append(f"{file_change.path} ({file_change.status}): {reason}")
            continue

        parsed = parse_patch(file_change.patch)
        patches[file_change.path] = parsed
        if file_change.status == "removed":
            line_map = parsed.deleted_line_map()
            source = build_source_text(line_map)
            lines = sorted(line_map)
        else:
            line_map = parsed.added_line_map()
            # Include enough surrounding context so evidence can mention nearby code.
            context_map = {
                line.new_lineno: line.content
                for line in parsed.lines
                if line.kind == "context" and line.new_lineno is not None
            }
            source = build_source_text({**context_map, **line_map})
            lines = sorted(line_map)
        if not source.strip() or not lines:
            skipped.append(
                f"{file_change.path} ({file_change.status}): patch contained no reviewable changed lines."
            )
            continue
        changed_code[file_change.path] = source
        changed_lines[file_change.path] = lines
    return changed_code, changed_lines, patches, skipped
