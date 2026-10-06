import re
from typing import List, Optional, Tuple

from src.shared.enums import FindingCategory, Severity
from src.shared.models import AnalysisFinding, PRContext


_HUNK_START = re.compile(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_EVAL_CALL = re.compile(r"\beval\s*\(")


def _added_lines(diff: str) -> List[Tuple[Optional[str], Optional[int], str]]:
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    lines: List[Tuple[Optional[str], Optional[int], str]] = []

    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            file_path = path[2:] if path.startswith("b/") else path
        elif line.startswith("@@"):
            match = _HUNK_START.match(line)
            line_number = int(match.group(1)) if match else None
        elif line.startswith("+") and not line.startswith("+++"):
            lines.append((file_path, line_number, line[1:]))
            if line_number is not None:
                line_number += 1
        elif line.startswith(" ") and line_number is not None:
            line_number += 1
        elif line.startswith("-"):
            continue
    return lines


def analyze_pr(context: PRContext) -> List[AnalysisFinding]:
    """Report direct eval calls in added diff lines."""
    findings: List[AnalysisFinding] = []
    for file_path, line_number, added_line in _added_lines(context.diff):
        if _EVAL_CALL.search(added_line):
            findings.append(
                AnalysisFinding(
                    file_path=file_path or "unknown",
                    line_number=line_number,
                    category=FindingCategory.SECURITY.value,
                    severity=Severity.HIGH.value,
                    message="Avoid eval() on input that may be untrusted.",
                    evidence=added_line.strip(),
                    suggestion="Replace eval() with an explicit parser or a safe allowlisted operation.",
                )
            )
    return findings
