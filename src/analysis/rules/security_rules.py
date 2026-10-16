"""Security analysis rules for detecting vulnerabilities in Python code."""

import ast
import re
from typing import List, Optional, Sequence, Tuple

from src.shared.enums import FindingCategory, Severity
from src.analysis.diff_parser import DiffLine
from src.analysis.rules.base import RuleFinding
from src.analysis.ast_parser import get_call_name, get_constant_value


_SUBPROCESS_FUNCTIONS = {
    "Popen",
    "call",
    "check_call",
    "check_output",
    "run",
}

# Secret scanning patterns
_AWS_KEY_PATTERN = re.compile(r"\b(AKIA[0-9A-Z]{16})\b")
_PRIVATE_KEY_PATTERN = re.compile(r"-----BEGIN (?:[A-Z0-9_-]+ )?PRIVATE KEY-----")
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"""(?i)\b(api_key|apikey|secret_key|secret|password|passwd|auth_token|access_token|private_token)\b\s*=\s*['"]([^'"]+)['"]"""
)

# Common trivial or placeholder strings
_EXACT_PLACEHOLDERS = {
    "password",
    "password123",
    "123456",
    "12345678",
    "123456789",
    "changeme",
    "default",
    "secret",
    "admin",
    "root",
    "none",
    "null",
}

_PLACEHOLDER_KEYWORDS = (
    "placeholder",
    "example",
    "dummy",
    "fake",
    "sample",
    "your_",
    "my_",
    "todo",
)


def _is_placeholder(value: str) -> bool:
    val_lower = value.strip().lower()
    if len(val_lower) < 8:
        return True
    if val_lower in _EXACT_PLACEHOLDERS:
        return True
    return any(keyword in val_lower for keyword in _PLACEHOLDER_KEYWORDS)


def _mask_secret(text: str) -> str:
    """Mask secret strings in evidence to prevent credential exposure in review logs."""
    def _mask_match(match: re.Match) -> str:
        var_name = match.group(1)
        val = match.group(2)
        if len(val) > 8:
            masked = val[:3] + "..." + val[-3:]
        else:
            masked = "[REDACTED]"
        return f'{var_name}="{masked}"'

    masked_text = _SECRET_ASSIGNMENT_PATTERN.sub(_mask_match, text)
    masked_text = _AWS_KEY_PATTERN.sub("AKIA****************", masked_text)
    if "-----BEGIN" in masked_text:
        masked_text = "[REDACTED PRIVATE KEY]"
    return masked_text


def _find_line(lines: Sequence[DiffLine], target_line_no: int) -> Optional[DiffLine]:
    for line in lines:
        if line.number == target_line_no:
            return line
    return None


def run_security_ast_rules(
    tree: ast.AST,
    lines: Sequence[DiffLine],
    line_offset: int,
) -> List[RuleFinding]:
    """Run AST-based security analysis rules."""
    findings: List[RuleFinding] = []
    added_numbers = {l.number for l in lines if l.is_added and l.number is not None}

    def real_lineno(ast_lineno: int) -> int:
        local_idx = ast_lineno - line_offset
        if 1 <= local_idx <= len(lines):
            return lines[local_idx - 1].number or 0
        return 0

    for node in ast.walk(tree):
        # Function Calls
        if isinstance(node, ast.Call):
            call_name = get_call_name(node.func)
            call_lineno = real_lineno(node.lineno)

            # SEC001: eval / exec
            if call_name in ("eval", "exec"):
                if call_lineno in added_numbers:
                    diff_line = _find_line(lines, call_lineno)
                    findings.append(
                        RuleFinding(
                            rule_id="SEC001",
                            category=FindingCategory.SECURITY.value,
                            severity=Severity.HIGH.value,
                            message=f"{call_name}() executes dynamically constructed code and can enable code injection.",
                            suggestion="Replace dynamic execution with a safe parser or explicit allowlisted operations.",
                            line_number=call_lineno,
                            evidence=(diff_line.text if diff_line else f"{call_name}(...)").strip(),
                        )
                    )

            # SEC002: os.system
            elif call_name == "os.system":
                if call_lineno in added_numbers:
                    diff_line = _find_line(lines, call_lineno)
                    findings.append(
                        RuleFinding(
                            rule_id="SEC002",
                            category=FindingCategory.SECURITY.value,
                            severity=Severity.HIGH.value,
                            message="os.system() passes a command through a system shell and may permit command injection.",
                            suggestion="Use subprocess with an argument list and avoid shell interpretation.",
                            line_number=call_lineno,
                            evidence=(diff_line.text if diff_line else "os.system(...)").strip(),
                        )
                    )

            # SEC002: os.popen
            elif call_name == "os.popen":
                if call_lineno in added_numbers:
                    diff_line = _find_line(lines, call_lineno)
                    findings.append(
                        RuleFinding(
                            rule_id="SEC002",
                            category=FindingCategory.SECURITY.value,
                            severity=Severity.HIGH.value,
                            message="os.popen() executes commands via shell pipeline and risks command injection.",
                            suggestion="Use subprocess.Popen with argument lists and shell=False.",
                            line_number=call_lineno,
                            evidence=(diff_line.text if diff_line else "os.popen(...)").strip(),
                        )
                    )

            # SEC002: subprocess with shell=True
            elif (
                call_name is not None
                and call_name.startswith("subprocess.")
                and call_name.split(".", 1)[1] in _SUBPROCESS_FUNCTIONS
            ):
                shell_kw = next(
                    (
                        kw
                        for kw in node.keywords
                        if kw.arg == "shell"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                    ),
                    None,
                )
                if shell_kw is not None:
                    kw_lineno = real_lineno(shell_kw.lineno)
                    target_lineno = kw_lineno if kw_lineno in added_numbers else call_lineno
                    if target_lineno in added_numbers:
                        diff_line = _find_line(lines, target_lineno)
                        findings.append(
                            RuleFinding(
                                rule_id="SEC002",
                                category=FindingCategory.SECURITY.value,
                                severity=Severity.HIGH.value,
                                message="subprocess is invoked with shell=True, so untrusted command data may be interpreted by a shell.",
                                suggestion="Pass the executable and arguments as a list and omit shell=True.",
                                line_number=target_lineno,
                                evidence=(diff_line.text if diff_line else "shell=True").strip(),
                            )
                        )

            # SEC003: Insecure Deserialization (pickle, marshal, shelve)
            elif call_name in (
                "pickle.loads",
                "pickle.load",
                "_pickle.loads",
                "_pickle.load",
                "marshal.loads",
                "marshal.load",
                "shelve.open",
            ):
                if call_lineno in added_numbers:
                    diff_line = _find_line(lines, call_lineno)
                    findings.append(
                        RuleFinding(
                            rule_id="SEC003",
                            category=FindingCategory.SECURITY.value,
                            severity=Severity.HIGH.value,
                            message=f"Insecure deserialization with {call_name}() can execute arbitrary malicious bytecode.",
                            suggestion="Use safe serialization formats like JSON or protocol buffers instead.",
                            line_number=call_lineno,
                            evidence=(diff_line.text if diff_line else f"{call_name}(...)").strip(),
                        )
                    )

            # SEC004: Insecure tempfile.mktemp
            elif call_name == "tempfile.mktemp":
                if call_lineno in added_numbers:
                    diff_line = _find_line(lines, call_lineno)
                    findings.append(
                        RuleFinding(
                            rule_id="SEC004",
                            category=FindingCategory.SECURITY.value,
                            severity=Severity.HIGH.value,
                            message="tempfile.mktemp() is insecure and subject to race condition symlink attacks.",
                            suggestion="Use tempfile.NamedTemporaryFile() or tempfile.mkstemp() instead.",
                            line_number=call_lineno,
                            evidence=(diff_line.text if diff_line else "tempfile.mktemp(...)").strip(),
                        )
                    )

            # SEC006: SQL Injection in cursor.execute
            elif call_name in ("cursor.execute", "cursor.executemany", "db.execute", "conn.execute"):
                if node.args and call_lineno in added_numbers:
                    first_arg = node.args[0]
                    is_formatted = (
                        isinstance(first_arg, ast.JoinedStr)
                        or (isinstance(first_arg, ast.BinOp) and isinstance(first_arg.op, ast.Mod))
                        or (
                            isinstance(first_arg, ast.Call)
                            and isinstance(first_arg.func, ast.Attribute)
                            and first_arg.func.attr == "format"
                        )
                    )
                    if is_formatted:
                        diff_line = _find_line(lines, call_lineno)
                        findings.append(
                            RuleFinding(
                                rule_id="SEC006",
                                category=FindingCategory.SECURITY.value,
                                severity=Severity.HIGH.value,
                                message="Potential SQL injection: SQL query string is dynamically interpolated instead of parameterized.",
                                suggestion="Pass parameters separately as query arguments (e.g., execute(sql, (param,))).",
                                line_number=call_lineno,
                                evidence=(diff_line.text if diff_line else f"{call_name}(...)").strip(),
                            )
                        )

            # SEC007: Dangerous Archive Extraction (Zip Slip / CVE-2007-4559)
            elif (
                call_name is not None
                and (call_name.endswith(".extractall") or call_name == "tarfile.extractall")
            ):
                if call_lineno in added_numbers:
                    has_filter = any(kw.arg == "filter" for kw in node.keywords)
                    if not has_filter:
                        diff_line = _find_line(lines, call_lineno)
                        findings.append(
                            RuleFinding(
                                rule_id="SEC007",
                                category=FindingCategory.SECURITY.value,
                                severity=Severity.HIGH.value,
                                message="Archive extraction with extractall() without path validation may permit arbitrary file overwrite.",
                                suggestion="Validate file paths of archive members or use a secure filter keyword.",
                                line_number=call_lineno,
                                evidence=(diff_line.text if diff_line else f"{call_name}(...)").strip(),
                            )
                        )

            # SEC008: Weak Hash (md5, sha1)
            elif call_name in ("hashlib.md5", "hashlib.sha1"):
                if call_lineno in added_numbers:
                    used_for_sec = next(
                        (
                            kw.value
                            for kw in node.keywords
                            if kw.arg == "usedforsecurity"
                            and isinstance(kw.value, ast.Constant)
                        ),
                        None,
                    )
                    if not (used_for_sec is not None and getattr(used_for_sec, "value", True) is False):
                        diff_line = _find_line(lines, call_lineno)
                        findings.append(
                            RuleFinding(
                                rule_id="SEC008",
                                category=FindingCategory.SECURITY.value,
                                severity=Severity.MEDIUM.value,
                                message=f"Weak cryptographic hash algorithm '{call_name}()' is vulnerable to collisions.",
                                suggestion="Use stronger cryptographic hashing such as hashlib.sha256().",
                                line_number=call_lineno,
                                evidence=(diff_line.text if diff_line else f"{call_name}(...)").strip(),
                            )
                        )

    # SEC005: Hardcoded credentials & Secrets scanning in added lines
    for line in lines:
        if not line.is_added or line.number is None:
            continue

        raw_text = line.text
        # AWS Key check
        if _AWS_KEY_PATTERN.search(raw_text):
            findings.append(
                RuleFinding(
                    rule_id="SEC005",
                    category=FindingCategory.SECURITY.value,
                    severity=Severity.CRITICAL.value,
                    message="Hardcoded AWS Access Key ID detected in source code.",
                    suggestion="Remove the secret key and supply credentials via environment variables or secret manager.",
                    line_number=line.number,
                    evidence=_mask_secret(raw_text).strip(),
                )
            )

        # Private Key check
        elif _PRIVATE_KEY_PATTERN.search(raw_text):
            findings.append(
                RuleFinding(
                    rule_id="SEC005",
                    category=FindingCategory.SECURITY.value,
                    severity=Severity.CRITICAL.value,
                    message="Hardcoded private key block detected in source code.",
                    suggestion="Do not commit private keys into version control; use secure key storage.",
                    line_number=line.number,
                    evidence="-----BEGIN PRIVATE KEY----- [REDACTED]",
                )
            )

        # Generic secret assignment check
        else:
            match = _SECRET_ASSIGNMENT_PATTERN.search(raw_text)
            if match:
                key_name = match.group(1)
                secret_val = match.group(2)
                if not _is_placeholder(secret_val):
                    findings.append(
                        RuleFinding(
                            rule_id="SEC005",
                            category=FindingCategory.SECURITY.value,
                            severity=Severity.HIGH.value,
                            message=f"Possible hardcoded credential or secret assigned to '{key_name}'.",
                            suggestion="Store sensitive credentials in secure configuration or environment variables.",
                            line_number=line.number,
                            evidence=_mask_secret(raw_text).strip(),
                        )
                    )

    return findings
