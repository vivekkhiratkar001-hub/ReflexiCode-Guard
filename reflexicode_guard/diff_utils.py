"""Unified-diff parsing: map changed lines and find valid review targets.

GitHub only accepts a line comment on a line that appears inside a diff hunk
(RIGHT side = new file). Anything else returns HTTP 422, so we filter first.
"""
from __future__ import annotations

import re

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_patch(patch: str | None) -> dict:
    """Return {'added': {new_line: text}, 'context': {new_line: text}, 'removed': [old_line,...]}."""
    out = {"added": {}, "context": {}, "removed": []}
    if not patch:
        return out
    old_ln = new_ln = 0
    in_hunk = False
    for raw in patch.splitlines():
        m = HUNK_RE.match(raw)
        if m:
            old_ln, new_ln = int(m.group(1)), int(m.group(3))
            in_hunk = True
            continue
        if not in_hunk or raw.startswith("\\"):      # "\ No newline at end of file"
            continue
        if raw.startswith("+"):
            out["added"][new_ln] = raw[1:]
            new_ln += 1
        elif raw.startswith("-"):
            out["removed"].append(old_ln)
            old_ln += 1
        else:                                         # context line (leading space)
            out["context"][new_ln] = raw[1:] if raw.startswith(" ") else raw
            old_ln += 1
            new_ln += 1
    return out


def changed_lines(patch: str | None) -> set[int]:
    return set(parse_patch(patch)["added"])


def commentable_lines(patch: str | None) -> set[int]:
    p = parse_patch(patch)
    return set(p["added"]) | set(p["context"])


def split_diff_by_file(diff_text: str) -> dict[str, str]:
    """Split a full `git diff` into {path: patch_hunks}. Used for offline/local runs."""
    files: dict[str, list[str]] = {}
    current = None
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            current = None
        elif line.startswith("+++ "):
            path = line[4:].strip()
            if path == "/dev/null":
                continue
            current = path[2:] if path.startswith("b/") else path
            files[current] = []
        elif current is not None and not line.startswith("--- "):
            files[current].append(line)
    return {p: "\n".join(ls) for p, ls in files.items()}


def hunk_ranges(patch: str | None) -> list[tuple[int, int]]:
    """New-file (start, end) line range of each hunk. Used for hunk-level vs line-level comparison."""
    out = []
    for raw in (patch or "").splitlines():
        m = HUNK_RE.match(raw)
        if m:
            start, length = int(m.group(3)), int(m.group(4)) if m.group(4) is not None else 1
            if length:
                out.append((start, start + length - 1))
    return out
