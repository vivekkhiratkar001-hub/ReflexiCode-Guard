"""Unified-diff parsing and GitHub comment line mapping."""

from __future__ import annotations

import re

from .models import HunkLine, ParsedPatch

HUNK_HEADER = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)

# Reconstructing full-file strings for Member 3 evidence checks uses padding.
# Student PRs are small; above this size we keep a compact snippet instead.
MAX_PADDED_LINES = 4000

UNSUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".woff",
    ".woff2",
    ".ttf",
    ".mp3",
    ".mp4",
    ".lock",
}


def is_probably_binary_path(path: str) -> bool:
    lowered = path.lower()
    return any(lowered.endswith(ext) for ext in UNSUPPORTED_EXTENSIONS)


def parse_patch(patch: str | None) -> ParsedPatch:
    """Parse a GitHub file `patch` field into old/new line numbers."""
    parsed = ParsedPatch()
    if not patch:
        return parsed

    old_lineno = 0
    new_lineno = 0
    in_hunk = False
    for raw_line in patch.splitlines():
        header = HUNK_HEADER.match(raw_line)
        if header:
            old_lineno = int(header.group("old_start"))
            new_lineno = int(header.group("new_start"))
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if raw_line.startswith("\\"):
            # "\ No newline at end of file"
            continue
        if raw_line.startswith("+"):
            parsed.lines.append(
                HunkLine("add", raw_line[1:], None, new_lineno)
            )
            new_lineno += 1
        elif raw_line.startswith("-"):
            parsed.lines.append(
                HunkLine("delete", raw_line[1:], old_lineno, None)
            )
            old_lineno += 1
        elif raw_line.startswith(" "):
            parsed.lines.append(
                HunkLine("context", raw_line[1:], old_lineno, new_lineno)
            )
            old_lineno += 1
            new_lineno += 1
        else:
            # GitHub sometimes omits the leading space on context lines.
            parsed.lines.append(
                HunkLine("context", raw_line, old_lineno, new_lineno)
            )
            old_lineno += 1
            new_lineno += 1
    return parsed


def build_source_text(line_map: dict[int, str]) -> str:
    """Build a 1-indexed source string so Member 3 can check evidence by line.

    Missing lines become empty strings. If the file would be huge, only the
    known changed lines are joined and Member 3 falls back to whole-snippet
    evidence matching.
    """
    if not line_map:
        return ""
    max_line = max(line_map)
    if max_line <= MAX_PADDED_LINES:
        rows = [line_map.get(index, "") for index in range(1, max_line + 1)]
        return "\n".join(rows)
    return "\n".join(line_map[key] for key in sorted(line_map))
