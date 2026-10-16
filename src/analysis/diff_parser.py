"""Unified diff parser for PR review and changed-line localization."""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


_HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
)


@dataclass(frozen=True)
class DiffLine:
    """Represents a single line within a diff hunk."""
    number: Optional[int]       # Line number in the new/target file (None for deleted lines)
    old_number: Optional[int]   # Line number in the old file (None for added lines)
    text: str                   # Line content with hunk prefix removed
    raw_prefix: str             # '+', '-', ' ', or '\'
    is_added: bool
    is_deleted: bool
    is_context: bool


@dataclass
class DiffHunk:
    """Represents a contiguous hunk within a file diff."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    header_comment: str = ""
    lines: List[DiffLine] = field(default_factory=list)

    @property
    def added_lines(self) -> List[DiffLine]:
        return [line for line in self.lines if line.is_added]

    @property
    def added_line_numbers(self) -> Set[int]:
        return {line.number for line in self.lines if line.is_added and line.number is not None}


@dataclass
class FileDiff:
    """Represents all changes made to a single file."""
    old_path: Optional[str] = None
    new_path: Optional[str] = None
    change_type: str = "modified"  # "added", "modified", "deleted", "renamed"
    is_binary: bool = False
    hunks: List[DiffHunk] = field(default_factory=list)

    @property
    def target_path(self) -> str:
        """The primary file path for reporting findings."""
        if self.new_path and self.new_path != "/dev/null":
            return self.new_path
        if self.old_path and self.old_path != "/dev/null":
            return self.old_path
        return "unknown"

    @property
    def added_line_numbers(self) -> Set[int]:
        """All line numbers in the target file that were added or modified."""
        result: Set[int] = set()
        for hunk in self.hunks:
            result.update(hunk.added_line_numbers)
        return result

    @property
    def added_lines(self) -> List[DiffLine]:
        """All added lines in the file across all hunks."""
        result: List[DiffLine] = []
        for hunk in self.hunks:
            result.extend(hunk.added_lines)
        return result

    def is_line_added(self, line_number: Optional[int]) -> bool:
        """Check if a line number was added/modified in this diff."""
        if line_number is None:
            return False
        return line_number in self.added_line_numbers


def _clean_diff_path(raw: str) -> Optional[str]:
    path = raw.strip()
    if not path or path == "/dev/null":
        return None
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def parse_unified_diff(diff_text: str) -> List[FileDiff]:
    """Parse a unified git diff into structured FileDiff models.
    
    Correctly tracks:
    - Added, modified, deleted, and renamed files
    - Multiple hunks per file
    - Target file line numbering
    - Context lines and deleted lines
    - Binary files
    """
    if not diff_text or not diff_text.strip():
        return []

    files: List[FileDiff] = []
    current_file: Optional[FileDiff] = None
    current_hunk: Optional[DiffHunk] = None

    old_line_num = 0
    new_line_num = 0
    old_remaining = 0
    new_remaining = 0

    lines = diff_text.splitlines()
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Check for start of git diff header
        if line.startswith("diff --git "):
            current_hunk = None
            current_file = FileDiff()
            files.append(current_file)
            parts = line[11:].strip().split()
            if len(parts) >= 2:
                current_file.old_path = _clean_diff_path(parts[0])
                current_file.new_path = _clean_diff_path(parts[1])
            i += 1
            continue

        if current_file is None:
            # Diff without `diff --git` header (e.g. raw patch)
            if line.startswith("--- ") or line.startswith("+++ "):
                current_file = FileDiff()
                files.append(current_file)

        if current_file is not None:
            if line.startswith("new file mode "):
                current_file.change_type = "added"
                i += 1
                continue
            elif line.startswith("deleted file mode "):
                current_file.change_type = "deleted"
                i += 1
                continue
            elif line.startswith("rename from "):
                current_file.old_path = line[12:].strip()
                current_file.change_type = "renamed"
                i += 1
                continue
            elif line.startswith("rename to "):
                current_file.new_path = line[10:].strip()
                current_file.change_type = "renamed"
                i += 1
                continue
            elif line.startswith("Binary files "):
                current_file.is_binary = True
                i += 1
                continue
            elif line.startswith("--- "):
                path = _clean_diff_path(line[4:])
                current_file.old_path = path
                if path is None:
                    current_file.change_type = "added"
                i += 1
                continue
            elif line.startswith("+++ "):
                path = _clean_diff_path(line[4:])
                current_file.new_path = path
                if path is None:
                    current_file.change_type = "deleted"
                i += 1
                continue

        # Check for hunk header
        if line.startswith("@@"):
            match = _HUNK_HEADER.match(line)
            if match:
                if current_file is None:
                    current_file = FileDiff()
                    files.append(current_file)

                old_start = int(match.group(1))
                old_count = int(match.group(2)) if match.group(2) is not None else 1
                new_start = int(match.group(3))
                new_count = int(match.group(4)) if match.group(4) is not None else 1
                comment = match.group(5).strip()

                current_hunk = DiffHunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    header_comment=comment,
                )
                current_file.hunks.append(current_hunk)

                old_line_num = old_start
                new_line_num = new_start
                old_remaining = old_count
                new_remaining = new_count
                i += 1
                continue

        # Hunk content lines
        if current_hunk is not None:
            if line.startswith("\\"):
                # "\ No newline at end of file"
                i += 1
                continue

            if line.startswith("+"):
                diff_line = DiffLine(
                    number=new_line_num,
                    old_number=None,
                    text=line[1:],
                    raw_prefix="+",
                    is_added=True,
                    is_deleted=False,
                    is_context=False,
                )
                current_hunk.lines.append(diff_line)
                new_line_num += 1
                new_remaining -= 1
            elif line.startswith("-"):
                diff_line = DiffLine(
                    number=None,
                    old_number=old_line_num,
                    text=line[1:],
                    raw_prefix="-",
                    is_added=False,
                    is_deleted=True,
                    is_context=False,
                )
                current_hunk.lines.append(diff_line)
                old_line_num += 1
                old_remaining -= 1
            elif line.startswith(" "):
                diff_line = DiffLine(
                    number=new_line_num,
                    old_number=old_line_num,
                    text=line[1:],
                    raw_prefix=" ",
                    is_added=False,
                    is_deleted=False,
                    is_context=True,
                )
                current_hunk.lines.append(diff_line)
                old_line_num += 1
                new_line_num += 1
                old_remaining -= 1
                new_remaining -= 1
            else:
                # Line does not match hunk syntax; end of hunk
                current_hunk = None
                continue

            if old_remaining <= 0 and new_remaining <= 0:
                current_hunk = None

        i += 1

    # Filter out empty files with no path and no hunks
    valid_files = [f for f in files if f.target_path != "unknown" or f.hunks]
    return valid_files
