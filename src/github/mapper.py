"""Map Member 3 ReviewFinding objects onto GitHub inline comments."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from src.llm.schemas import ReviewFinding

from .models import FileChange, InlineComment, ParsedPatch

FINGERPRINT_MARKER = "reflexicode-guard:fp="
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip().lstrip("./").casefold()


def finding_fingerprint(finding: ReviewFinding, side: str) -> str:
    """Stable id used to skip duplicate comments on workflow reruns."""
    message = _NON_ALNUM.sub(" ", finding.message.casefold()).strip()
    raw = "|".join(
        [
            normalize_path(finding.file),
            str(finding.line),
            side,
            finding.category.casefold(),
            message,
        ]
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return digest


def format_comment_body(finding: ReviewFinding, fingerprint: str) -> str:
    suggestion = finding.suggestion or "No suggestion provided."
    return (
        f"**[{finding.severity}] {finding.category}** in `{finding.file}:{finding.line}`\n\n"
        f"{finding.message}\n\n"
        f"**Evidence:** `{finding.evidence}`\n\n"
        f"**Suggestion:** {suggestion}\n\n"
        f"<!-- {FINGERPRINT_MARKER}{fingerprint} -->"
    )


def extract_fingerprints(comment_bodies: Sequence[str]) -> set[str]:
    found: set[str] = set()
    marker = FINGERPRINT_MARKER
    for body in comment_bodies:
        start = 0
        while True:
            index = body.find(marker, start)
            if index == -1:
                break
            value = body[index + len(marker) :]
            fingerprint = "".join(ch for ch in value[:16] if ch.isalnum())
            if fingerprint:
                found.add(fingerprint)
            start = index + len(marker)
    return found


class FindingCommentMapper:
    """Convert validated findings into GitHub review comments when the line is in the diff."""

    def __init__(self, files: Sequence[FileChange], patches: dict[str, ParsedPatch]) -> None:
        self._files_by_path = {normalize_path(item.path): item for item in files}
        self._patches = {normalize_path(path): patch for path, patch in patches.items()}

    def map_findings(
        self,
        findings: Sequence[ReviewFinding],
    ) -> tuple[list[InlineComment], list[str]]:
        comments: list[InlineComment] = []
        unmapped: list[str] = []
        for finding in findings:
            comment = self.map_finding(finding)
            if comment is None:
                unmapped.append(f"{finding.file}:{finding.line} ({finding.category})")
            else:
                comments.append(comment)
        return comments, unmapped

    def map_finding(self, finding: ReviewFinding) -> InlineComment | None:
        key = normalize_path(finding.file)
        file_change = self._files_by_path.get(key)
        patch = self._patches.get(key)
        if file_change is None or patch is None or not file_change.reviewable:
            return None

        right_lines = patch.right_commentable_lines()
        left_lines = patch.left_commentable_lines()
        status = file_change.status
        side: str | None = None
        if status == "removed":
            if finding.line in left_lines:
                side = "LEFT"
        elif finding.line in right_lines:
            side = "RIGHT"
        elif finding.line in left_lines:
            # Deleted line inside a modified file.
            side = "LEFT"
        if side is None:
            return None

        fingerprint = finding_fingerprint(finding, side)
        return InlineComment(
            path=file_change.path,
            line=finding.line,
            side=side,  # type: ignore[arg-type]
            body=format_comment_body(finding, fingerprint),
            fingerprint=fingerprint,
            finding_file=finding.file,
            finding_line=finding.line,
        )
