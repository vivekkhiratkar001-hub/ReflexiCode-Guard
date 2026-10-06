"""Deterministic, changed-line-focused Python checks for unified PR diffs."""

import ast
import io
import re
import textwrap
import tokenize
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from src.shared.enums import FindingCategory, Severity
from src.shared.models import AnalysisFinding, PRContext


_HUNK_HEADER = re.compile(
    r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:.*)$"
)
_SUBPROCESS_FUNCTIONS = {
    "Popen",
    "call",
    "check_call",
    "check_output",
    "run",
}
_IGNORED_TOKEN_TYPES = {
    tokenize.COMMENT,
    tokenize.DEDENT,
    tokenize.ENCODING,
    tokenize.INDENT,
    tokenize.NEWLINE,
    tokenize.NL,
}


@dataclass(frozen=True)
class _DiffLine:
    number: int
    text: str
    added: bool


@dataclass
class _DiffHunk:
    file_path: str
    lines: List[_DiffLine] = field(default_factory=list)


@dataclass(frozen=True)
class _Rule:
    line_number: int
    category: str
    severity: str
    message: str
    suggestion: str
    evidence: Optional[str] = None


def _diff_hunks(diff: str) -> List[_DiffHunk]:
    """Extract new-file lines and their changed status from unified diff hunks."""
    hunks: List[_DiffHunk] = []
    file_path: Optional[str] = None
    current: Optional[_DiffHunk] = None
    old_remaining = 0
    new_remaining = 0
    new_line_number = 0

    for line in diff.splitlines():
        if current is not None and old_remaining == 0 and new_remaining == 0:
            current = None

        if current is not None and line.startswith("@@"):
            current = None

        if current is None:
            if line.startswith("diff --git "):
                file_path = None
            elif line.startswith("+++ "):
                raw_path = line[4:].strip()
                if raw_path == "/dev/null":
                    file_path = None
                else:
                    file_path = raw_path[2:] if raw_path.startswith("b/") else raw_path
            elif line.startswith("@@") and file_path is not None:
                match = _HUNK_HEADER.match(line)
                if match is None:
                    continue
                new_line_number = int(match.group(2))
                old_remaining = int(match.group(1)) if match.group(1) else 1
                new_remaining = int(match.group(3)) if match.group(3) else 1
                current = _DiffHunk(file_path=file_path)
                hunks.append(current)
            continue

        if line.startswith("\\"):
            continue
        if line.startswith("+"):
            current.lines.append(_DiffLine(new_line_number, line[1:], True))
            new_line_number += 1
            new_remaining -= 1
        elif line.startswith("-"):
            old_remaining -= 1
        elif line.startswith(" "):
            current.lines.append(_DiffLine(new_line_number, line[1:], False))
            new_line_number += 1
            old_remaining -= 1
            new_remaining -= 1
        else:
            # A malformed hunk should not prevent parsing a later file header.
            current = None
            if line.startswith("+++ "):
                raw_path = line[4:].strip()
                file_path = None if raw_path == "/dev/null" else (
                    raw_path[2:] if raw_path.startswith("b/") else raw_path
                )
    return hunks


def _is_python(path: str) -> bool:
    return path.lower().endswith(".py")


def _call_name(function: ast.expr) -> Optional[str]:
    if isinstance(function, ast.Name):
        return function.id
    if (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
    ):
        return f"{function.value.id}.{function.attr}"
    return None


def _changed_rule(
    lines: Sequence[_DiffLine],
    local_line: int,
    category: str,
    severity: str,
    message: str,
    suggestion: str,
    evidence: Optional[str] = None,
) -> Optional[_Rule]:
    if local_line < 1 or local_line > len(lines):
        return None
    diff_line = lines[local_line - 1]
    if not diff_line.added:
        return None
    selected_evidence = (evidence or diff_line.text).strip()
    if not selected_evidence:
        return None
    return _Rule(
        line_number=diff_line.number,
        category=category,
        severity=severity,
        message=message,
        suggestion=suggestion,
        evidence=selected_evidence,
    )


def _ast_rules(lines: Sequence[_DiffLine]) -> List[_Rule]:
    source = "\n".join(line.text for line in lines)
    try:
        tree = ast.parse(textwrap.dedent(source))
    except (IndentationError, SyntaxError, ValueError):
        return []

    rules: List[_Rule] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            rule = _changed_rule(
                lines,
                node.lineno,
                FindingCategory.STATIC_ANALYSIS.value,
                Severity.MEDIUM.value,
                "A bare except catches every exception and can hide programming errors.",
                "Catch the specific exception types this code can handle.",
            )
            if rule is not None:
                rules.append(rule)
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in ("eval", "exec"):
                rule = _changed_rule(
                    lines,
                    node.lineno,
                    FindingCategory.SECURITY.value,
                    Severity.HIGH.value,
                    f"{name}() executes dynamically constructed code and can enable code injection.",
                    "Replace dynamic execution with a safe parser or explicit allowlisted operations.",
                )
            elif name == "os.system":
                rule = _changed_rule(
                    lines,
                    node.lineno,
                    FindingCategory.SECURITY.value,
                    Severity.HIGH.value,
                    "os.system() passes a command through a system shell and may permit command injection.",
                    "Use subprocess with an argument list and avoid shell interpretation.",
                )
            elif (
                name is not None
                and name.startswith("subprocess.")
                and name.split(".", 1)[1] in _SUBPROCESS_FUNCTIONS
                and any(
                    keyword.arg == "shell"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in node.keywords
                )
            ):
                shell_keyword = next(
                    keyword
                    for keyword in node.keywords
                    if keyword.arg == "shell"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                )
                rule = _changed_rule(
                    lines,
                    shell_keyword.lineno,
                    FindingCategory.SECURITY.value,
                    Severity.HIGH.value,
                    "subprocess is invoked with shell=True, so untrusted command data may be interpreted by a shell.",
                    "Pass the executable and arguments as a list and omit shell=True.",
                )
            elif name == "print":
                rule = _changed_rule(
                    lines,
                    node.lineno,
                    FindingCategory.STATIC_ANALYSIS.value,
                    Severity.LOW.value,
                    "print() writes directly to standard output, which is often unsuitable for production code.",
                    "Use the application's configured logging or output mechanism.",
                )
            else:
                rule = None

            if rule is not None:
                rules.append(rule)
    return rules


def _tokens(source: str) -> List[tokenize.TokenInfo]:
    result: List[tokenize.TokenInfo] = []
    generator = tokenize.generate_tokens(io.StringIO(source).readline)
    try:
        while True:
            result.append(next(generator))
    except StopIteration:
        pass
    except (IndentationError, tokenize.TokenError):
        pass
    return [token for token in result if token.type not in _IGNORED_TOKEN_TYPES]


def _lexical_rules(lines: Sequence[_DiffLine]) -> List[_Rule]:
    """Fallback checks for syntactically incomplete hunk fragments."""
    tokens = _tokens("\n".join(line.text for line in lines))
    rules: List[_Rule] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        value = token.string
        line_number = token.start[0]

        if value in ("eval", "exec") and index + 1 < len(tokens):
            if tokens[index + 1].string == "(":
                rule = _changed_rule(
                    lines,
                    line_number,
                    FindingCategory.SECURITY.value,
                    Severity.HIGH.value,
                    f"{value}() executes dynamically constructed code and can enable code injection.",
                    "Replace dynamic execution with a safe parser or explicit allowlisted operations.",
                )
                if rule is not None:
                    rules.append(rule)
        elif (
            value == "os"
            and index + 3 < len(tokens)
            and [item.string for item in tokens[index + 1 : index + 4]]
            == [".", "system", "("]
        ):
            rule = _changed_rule(
                lines,
                line_number,
                FindingCategory.SECURITY.value,
                Severity.HIGH.value,
                "os.system() passes a command through a system shell and may permit command injection.",
                "Use subprocess with an argument list and avoid shell interpretation.",
            )
            if rule is not None:
                rules.append(rule)
        elif (
            value == "subprocess"
            and index + 3 < len(tokens)
            and tokens[index + 1].string == "."
            and tokens[index + 2].string in _SUBPROCESS_FUNCTIONS
            and tokens[index + 3].string == "("
        ):
            depth = 0
            end = index + 3
            while end < len(tokens):
                if tokens[end].string == "(":
                    depth += 1
                elif tokens[end].string == ")":
                    depth -= 1
                    if depth == 0:
                        break
                end += 1
            shell_token = next(
                (
                    tokens[position]
                    for position in range(index + 4, end)
                    if tokens[position].string == "shell"
                    and position + 2 < end
                    and tokens[position + 1].string == "="
                    and tokens[position + 2].string == "True"
                ),
                None,
            )
            if shell_token is not None:
                rule = _changed_rule(
                    lines,
                    shell_token.start[0],
                    FindingCategory.SECURITY.value,
                    Severity.HIGH.value,
                    "subprocess is invoked with shell=True, so untrusted command data may be interpreted by a shell.",
                    "Pass the executable and arguments as a list and omit shell=True.",
                )
                if rule is not None:
                    rules.append(rule)
        elif value == "except" and index + 1 < len(tokens):
            if tokens[index + 1].string == ":":
                rule = _changed_rule(
                    lines,
                    line_number,
                    FindingCategory.STATIC_ANALYSIS.value,
                    Severity.MEDIUM.value,
                    "A bare except catches every exception and can hide programming errors.",
                    "Catch the specific exception types this code can handle.",
                )
                if rule is not None:
                    rules.append(rule)
        elif value == "print" and index + 1 < len(tokens):
            if tokens[index + 1].string == "(":
                rule = _changed_rule(
                    lines,
                    line_number,
                    FindingCategory.STATIC_ANALYSIS.value,
                    Severity.LOW.value,
                    "print() writes directly to standard output, which is often unsuitable for production code.",
                    "Use the application's configured logging or output mechanism.",
                )
                if rule is not None:
                    rules.append(rule)
        index += 1
    return rules


def _make_findings(hunks: Sequence[_DiffHunk]) -> List[AnalysisFinding]:
    findings_by_identity: Dict[Tuple[str, int, str, str], AnalysisFinding] = {}
    for hunk in hunks:
        if not _is_python(hunk.file_path):
            continue
        rules = _ast_rules(hunk.lines)
        if not rules:
            rules = _lexical_rules(hunk.lines)
        for rule in rules:
            identity = (
                hunk.file_path,
                rule.line_number,
                rule.category,
                rule.message,
            )
            findings_by_identity.setdefault(
                identity,
                AnalysisFinding(
                    file_path=hunk.file_path,
                    line_number=rule.line_number,
                    category=rule.category,
                    severity=rule.severity,
                    message=rule.message,
                    evidence=rule.evidence or "",
                    suggestion=rule.suggestion,
                ),
            )

    return sorted(
        findings_by_identity.values(),
        key=lambda item: (
            item.file_path,
            item.line_number if item.line_number is not None else -1,
            item.category,
            item.message,
        ),
    )


def analyze_pr(context: PRContext) -> List[AnalysisFinding]:
    """Analyze added Python lines in a PR diff and return shared findings."""
    if not context.diff:
        return []
    return _make_findings(_diff_hunks(context.diff))
