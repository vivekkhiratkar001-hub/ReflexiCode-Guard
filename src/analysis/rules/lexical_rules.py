"""Lexical token-based fallback rules for syntactically incomplete fragments."""

from typing import List, Optional, Sequence

from src.shared.enums import FindingCategory, Severity
from src.analysis.diff_parser import DiffLine
from src.analysis.rules.base import RuleFinding
from src.analysis.ast_parser import tokenize_source


_SUBPROCESS_FUNCTIONS = {
    "Popen",
    "call",
    "check_call",
    "check_output",
    "run",
}


def _find_line(lines: Sequence[DiffLine], target_line_no: int) -> Optional[DiffLine]:
    for line in lines:
        if line.number == target_line_no:
            return line
    return None


def run_lexical_rules(lines: Sequence[DiffLine]) -> List[RuleFinding]:
    """Fallback rules executed when AST parsing is unavailable for incomplete fragments."""
    tokens = tokenize_source("\n".join(line.text for line in lines))
    findings: List[RuleFinding] = []
    index = 0
    num_tokens = len(tokens)

    while index < num_tokens:
        token = tokens[index]
        value = token.string
        local_line_idx = token.start[0]

        # Map local line index back to DiffLine
        if 1 <= local_line_idx <= len(lines):
            diff_line = lines[local_line_idx - 1]
            if diff_line.is_added and diff_line.number is not None:
                line_no = diff_line.number

                # 1. eval / exec
                if value in ("eval", "exec") and index + 1 < num_tokens:
                    if tokens[index + 1].string == "(":
                        findings.append(
                            RuleFinding(
                                rule_id="SEC001",
                                category=FindingCategory.SECURITY.value,
                                severity=Severity.HIGH.value,
                                message=f"{value}() executes dynamically constructed code and can enable code injection.",
                                suggestion="Replace dynamic execution with a safe parser or explicit allowlisted operations.",
                                line_number=line_no,
                                evidence=diff_line.text.strip(),
                            )
                        )

                # 2. os.system
                elif (
                    value == "os"
                    and index + 3 < num_tokens
                    and [tokens[index + 1].string, tokens[index + 2].string, tokens[index + 3].string]
                    == [".", "system", "("]
                ):
                    findings.append(
                        RuleFinding(
                            rule_id="SEC002",
                            category=FindingCategory.SECURITY.value,
                            severity=Severity.HIGH.value,
                            message="os.system() passes a command through a system shell and may permit command injection.",
                            suggestion="Use subprocess with an argument list and avoid shell interpretation.",
                            line_number=line_no,
                            evidence=diff_line.text.strip(),
                        )
                    )

                # 3. subprocess with shell=True
                elif (
                    value == "subprocess"
                    and index + 3 < num_tokens
                    and tokens[index + 1].string == "."
                    and tokens[index + 2].string in _SUBPROCESS_FUNCTIONS
                    and tokens[index + 3].string == "("
                ):
                    depth = 0
                    end = index + 3
                    while end < num_tokens:
                        if tokens[end].string == "(":
                            depth += 1
                        elif tokens[end].string == ")":
                            depth -= 1
                            if depth == 0:
                                break
                        end += 1

                    shell_token = next(
                        (
                            tokens[pos]
                            for pos in range(index + 4, end)
                            if tokens[pos].string == "shell"
                            and pos + 2 < end
                            and tokens[pos + 1].string == "="
                            and tokens[pos + 2].string == "True"
                        ),
                        None,
                    )
                    if shell_token is not None:
                        shell_local_line = shell_token.start[0]
                        if 1 <= shell_local_line <= len(lines):
                            shell_diff_line = lines[shell_local_line - 1]
                            findings.append(
                                RuleFinding(
                                    rule_id="SEC002",
                                    category=FindingCategory.SECURITY.value,
                                    severity=Severity.HIGH.value,
                                    message="subprocess is invoked with shell=True, so untrusted command data may be interpreted by a shell.",
                                    suggestion="Pass the executable and arguments as a list and omit shell=True.",
                                    line_number=shell_diff_line.number,
                                    evidence=shell_diff_line.text.strip(),
                                )
                            )

                # 4. Bare except
                elif value == "except" and index + 1 < num_tokens:
                    if tokens[index + 1].string == ":":
                        findings.append(
                            RuleFinding(
                                rule_id="DEF001",
                                category=FindingCategory.STATIC_ANALYSIS.value,
                                severity=Severity.MEDIUM.value,
                                message="A bare except catches every exception and can hide programming errors.",
                                suggestion="Catch the specific exception types this code can handle.",
                                line_number=line_no,
                                evidence=diff_line.text.strip(),
                            )
                        )

                # 5. print()
                elif value == "print" and index + 1 < num_tokens:
                    if tokens[index + 1].string == "(":
                        findings.append(
                            RuleFinding(
                                rule_id="DEF008",
                                category=FindingCategory.STATIC_ANALYSIS.value,
                                severity=Severity.LOW.value,
                                message="print() writes directly to standard output, which is often unsuitable for production code.",
                                suggestion="Use the application's configured logging or output mechanism.",
                                line_number=line_no,
                                evidence=diff_line.text.strip(),
                            )
                        )

        index += 1

    return findings
