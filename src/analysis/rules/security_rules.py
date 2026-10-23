"""Security analysis rules for detecting vulnerabilities in Python code."""

import ast
import base64
import binascii
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
_PRIVATE_KEY_BLOCK_PATTERN = re.compile(
    r"-----BEGIN (?P<label>(?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY)-----"
    r"\s*(?P<body>[A-Za-z0-9+/=\r\n]+?)\s*"
    r"-----END (?P=label)-----",
    re.DOTALL,
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"""(?i)\b(api_key|apikey|secret_key|secret|password|passwd|auth_token|access_token|private_token)\b\s*=\s*['"]([^'"]+)['"]"""
)
_TOKEN_SECRET_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
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

_SYNTHETIC_AWS_KEYS = {
    "AKIA" + "1234567890ABCDEF",
}


def _is_placeholder(value: str) -> bool:
    val_lower = value.strip().lower()
    if len(val_lower) < 8:
        return True
    if val_lower in _EXACT_PLACEHOLDERS:
        return True
    return any(keyword in val_lower for keyword in _PLACEHOLDER_KEYWORDS)


def _is_synthetic_aws_key(value: str) -> bool:
    """Recognize explicitly synthetic AWS examples without exempting test files."""
    return value in _SYNTHETIC_AWS_KEYS


def _read_der_value(data: bytes, offset: int) -> Optional[Tuple[int, bytes, int]]:
    if offset + 2 > len(data):
        return None
    tag = data[offset]
    first_length = data[offset + 1]
    cursor = offset + 2
    if first_length < 0x80:
        content_length = first_length
    else:
        length_bytes = first_length & 0x7F
        if length_bytes == 0 or length_bytes > 4 or cursor + length_bytes > len(data):
            return None
        content_length = int.from_bytes(data[cursor : cursor + length_bytes], "big")
        cursor += length_bytes
    end = cursor + content_length
    if end > len(data):
        return None
    return tag, data[cursor:end], end


def _read_der_sequence(data: bytes) -> Optional[list[Tuple[int, bytes]]]:
    sequence = _read_der_value(data, 0)
    if sequence is None or sequence[0] != 0x30 or sequence[2] != len(data):
        return None
    return _read_der_sequence_contents(sequence[1])


def _read_der_sequence_contents(data: bytes) -> Optional[list[Tuple[int, bytes]]]:
    fields: list[Tuple[int, bytes]] = []
    cursor = 0
    while cursor < len(data):
        value = _read_der_value(data, cursor)
        if value is None:
            return None
        fields.append((value[0], value[1]))
        cursor = value[2]
    return fields


def _is_plausible_private_key(block: str) -> bool:
    match = _PRIVATE_KEY_BLOCK_PATTERN.search(block)
    if match is None:
        return False
    try:
        decoded = base64.b64decode(
            re.sub(r"\s+", "", match.group("body")),
            validate=True,
        )
    except (binascii.Error, ValueError):
        return False
    if match.group("label") == "OPENSSH PRIVATE KEY":
        return decoded.startswith(b"openssh-key-v1\x00") and len(decoded) > 32
    fields = _read_der_sequence(decoded)
    if fields is None or not fields:
        return False
    label = match.group("label")
    if label == "RSA PRIVATE KEY":
        return (
            len(fields) == 9
            and all(tag == 0x02 and value for tag, value in fields)
            and fields[0][1] in (b"\x00", b"\x01")
        )
    if label == "DSA PRIVATE KEY":
        return (
            len(fields) == 6
            and all(tag == 0x02 and value for tag, value in fields)
            and fields[0][1] == b"\x00"
        )
    if label == "EC PRIVATE KEY":
        return (
            len(fields) >= 2
            and fields[0] == (0x02, b"\x01")
            and fields[1][0] == 0x04
            and bool(fields[1][1])
        )
    if label == "ENCRYPTED PRIVATE KEY":
        return (
            len(fields) == 2
            and fields[0][0] == 0x30
            and any(
                tag == 0x06
                for tag, _ in _read_der_sequence_contents(fields[0][1]) or []
            )
            and fields[1][0] == 0x04
            and bool(fields[1][1])
        )
    if label == "PRIVATE KEY":
        if len(fields) != 3 or fields[0][0] != 0x02 or fields[0][1] not in (b"\x00", b"\x01"):
            return False
        algorithm = (
            _read_der_sequence_contents(fields[1][1])
            if fields[1][0] == 0x30
            else None
        )
        return (
            algorithm is not None
            and any(tag == 0x06 and value for tag, value in algorithm)
            and fields[2][0] == 0x04
            and bool(fields[2][1])
        )
    return False


def redact_secret_text(text: str) -> str:
    """Mask recognizable credentials before evidence is included in review comments."""
    def _mask_match(match: re.Match) -> str:
        var_name = match.group(1)
        return f'{var_name}="[REDACTED]"'

    masked_text = _SECRET_ASSIGNMENT_PATTERN.sub(_mask_match, text)
    masked_text = _AWS_KEY_PATTERN.sub("AKIA****************", masked_text)
    for pattern in _TOKEN_SECRET_PATTERNS:
        masked_text = pattern.sub("[REDACTED TOKEN]", masked_text)
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
    added_groups: list[list[DiffLine]] = []
    added_group: list[DiffLine] = []
    previous_added_number: Optional[int] = None
    for diff_line in lines:
        if diff_line.is_added and diff_line.number is not None:
            if (
                added_group
                and previous_added_number is not None
                and diff_line.number != previous_added_number + 1
            ):
                added_groups.append(added_group)
                added_group = []
            added_group.append(diff_line)
            previous_added_number = diff_line.number
        else:
            if added_group:
                added_groups.append(added_group)
                added_group = []
            previous_added_number = None
    if added_group:
        added_groups.append(added_group)

    for group in added_groups:
        added_text = "\n".join(line.text for line in group)
        if not _is_plausible_private_key(added_text):
            continue
        match = _PRIVATE_KEY_BLOCK_PATTERN.search(added_text)
        if match is None:
            continue
        key_line_index = added_text.count("\n", 0, match.start())
        findings.append(
            RuleFinding(
                rule_id="SEC005",
                category=FindingCategory.SECURITY.value,
                severity=Severity.CRITICAL.value,
                message="Hardcoded private key block detected in source code.",
                suggestion="Do not commit private keys into version control; use secure key storage.",
                line_number=group[key_line_index].number,
                evidence="[REDACTED PRIVATE KEY BLOCK]",
            )
        )

    for line in lines:
        if not line.is_added or line.number is None:
            continue

        raw_text = line.text
        # AWS Key check
        aws_match = _AWS_KEY_PATTERN.search(raw_text)
        if aws_match and not _is_synthetic_aws_key(aws_match.group(1)):
            findings.append(
                RuleFinding(
                    rule_id="SEC005",
                    category=FindingCategory.SECURITY.value,
                    severity=Severity.CRITICAL.value,
                    message="Hardcoded AWS Access Key ID detected in source code.",
                    suggestion="Remove the secret key and supply credentials via environment variables or secret manager.",
                    line_number=line.number,
                    evidence=redact_secret_text(raw_text).strip(),
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
                            evidence=redact_secret_text(raw_text).strip(),
                        )
                    )

    return findings
