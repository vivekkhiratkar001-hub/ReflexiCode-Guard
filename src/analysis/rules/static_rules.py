"""Deterministic static analysis rules for Python code."""

import ast
from typing import Any, Callable, List, Optional, Sequence, Set

from src.shared.enums import FindingCategory, Severity
from src.analysis.diff_parser import DiffLine
from src.analysis.rules.base import RuleFinding
from src.analysis.ast_parser import get_call_name, get_constant_value


def _find_line(lines: Sequence[DiffLine], target_line_no: int) -> Optional[DiffLine]:
    """Find a diff line by its target file line number."""
    for line in lines:
        if line.number == target_line_no:
            return line
    return None


def run_static_ast_rules(
    tree: ast.AST,
    lines: Sequence[DiffLine],
    line_offset: int,
) -> List[RuleFinding]:
    """Run AST-based deterministic static analysis rules."""
    findings: List[RuleFinding] = []
    added_numbers = {l.number for l in lines if l.is_added and l.number is not None}

    def real_lineno(ast_lineno: int) -> int:
        local_idx = ast_lineno - line_offset
        if 1 <= local_idx <= len(lines):
            return lines[local_idx - 1].number or 0
        return 0

    for node in ast.walk(tree):
        # 1. DEF001 & DEF002: Exception handlers
        if isinstance(node, ast.ExceptHandler):
            lineno = real_lineno(node.lineno)
            if lineno in added_numbers:
                diff_line = _find_line(lines, lineno)
                if node.type is None:
                    findings.append(
                        RuleFinding(
                            rule_id="DEF001",
                            category=FindingCategory.STATIC_ANALYSIS.value,
                            severity=Severity.MEDIUM.value,
                            message="A bare except catches every exception and can hide programming errors.",
                            suggestion="Catch the specific exception types this code can handle.",
                            line_number=lineno,
                            evidence=(diff_line.text if diff_line else "except:").strip(),
                        )
                    )
                else:
                    is_base_exception = (
                        isinstance(node.type, ast.Name) and node.type.id == "BaseException"
                    )
                    is_empty_pass = (
                        len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
                    )
                    if is_base_exception:
                        findings.append(
                            RuleFinding(
                                rule_id="DEF002",
                                category=FindingCategory.STATIC_ANALYSIS.value,
                                severity=Severity.MEDIUM.value,
                                message="Catching BaseException suppresses critical system exit and interrupt exceptions.",
                                suggestion="Catch Exception or specific derived exception classes instead.",
                                line_number=lineno,
                                evidence=(diff_line.text if diff_line else "").strip(),
                            )
                        )
                    elif is_empty_pass and isinstance(node.type, ast.Name) and node.type.id == "Exception":
                        findings.append(
                            RuleFinding(
                                rule_id="DEF002",
                                category=FindingCategory.STATIC_ANALYSIS.value,
                                severity=Severity.LOW.value,
                                message="Empty exception handler 'except Exception: pass' silently swallows errors.",
                                suggestion="Log the exception or handle expected error scenarios explicitly.",
                                line_number=lineno,
                                evidence=(diff_line.text if diff_line else "").strip(),
                            )
                        )

        # 2. DEF003: Mutable default argument
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for default in node.args.defaults + node.args.kw_defaults:
                if default is not None and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    def_lineno = real_lineno(default.lineno)
                    if def_lineno in added_numbers or real_lineno(node.lineno) in added_numbers:
                        target_no = def_lineno if def_lineno in added_numbers else real_lineno(node.lineno)
                        diff_line = _find_line(lines, target_no)
                        findings.append(
                            RuleFinding(
                                rule_id="DEF003",
                                category=FindingCategory.STATIC_ANALYSIS.value,
                                severity=Severity.MEDIUM.value,
                                message=f"Default argument in function '{node.name}' uses mutable default value.",
                                suggestion="Use None as default and initialize mutable object inside function.",
                                line_number=target_no,
                                evidence=(diff_line.text if diff_line else f"def {node.name}(...)").strip(),
                            )
                        )

        # 3. DEF004: Unreachable code
        if hasattr(node, "body") and isinstance(node.body, list) and len(node.body) > 1:
            terminator_seen = False
            for stmt in node.body:
                if terminator_seen:
                    stmt_lineno = real_lineno(stmt.lineno)
                    if stmt_lineno in added_numbers:
                        diff_line = _find_line(lines, stmt_lineno)
                        findings.append(
                            RuleFinding(
                                rule_id="DEF004",
                                category=FindingCategory.STATIC_ANALYSIS.value,
                                severity=Severity.MEDIUM.value,
                                message="Unreachable code detected following a terminating statement (return/raise/break/continue).",
                                suggestion="Remove the unreachable statements or verify conditional flow.",
                                line_number=stmt_lineno,
                                evidence=(diff_line.text if diff_line else "").strip(),
                            )
                        )
                if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                    terminator_seen = True

        # 4. DEF005: Singleton comparison & DEF007: Literal identity comparison
        if isinstance(node, ast.Compare):
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    val = get_constant_value(comparator)
                    if val is None and isinstance(comparator, ast.Constant):
                        val = None
                    if val in (None, True, False) and isinstance(comparator, ast.Constant):
                        comp_lineno = real_lineno(node.lineno)
                        if comp_lineno in added_numbers:
                            diff_line = _find_line(lines, comp_lineno)
                            target_op = "is" if isinstance(op, ast.Eq) else "is not"
                            findings.append(
                                RuleFinding(
                                    rule_id="DEF005",
                                    category=FindingCategory.STATIC_ANALYSIS.value,
                                    severity=Severity.LOW.value,
                                    message=f"Comparison with singleton {val} should use '{target_op}' identity comparison.",
                                    suggestion=f"Replace equality operator with '{target_op} {val}'.",
                                    line_number=comp_lineno,
                                    evidence=(diff_line.text if diff_line else "").strip(),
                                )
                            )
                elif isinstance(op, (ast.Is, ast.IsNot)):
                    val = get_constant_value(comparator)
                    if isinstance(val, (str, int, float, bytes)):
                        comp_lineno = real_lineno(node.lineno)
                        if comp_lineno in added_numbers:
                            diff_line = _find_line(lines, comp_lineno)
                            findings.append(
                                RuleFinding(
                                    rule_id="DEF007",
                                    category=FindingCategory.STATIC_ANALYSIS.value,
                                    severity=Severity.MEDIUM.value,
                                    message=f"Using 'is' or 'is not' with a literal {type(val).__name__} is undefined.",
                                    suggestion="Use '==' or '!=' equality operators when comparing literals.",
                                    line_number=comp_lineno,
                                    evidence=(diff_line.text if diff_line else "").strip(),
                                )
                            )

        # 5. DEF006: Duplicate dictionary keys
        if isinstance(node, ast.Dict):
            seen_keys: Set[Any] = set()
            for key_node in node.keys:
                if key_node is not None:
                    key_val = get_constant_value(key_node)
                    if key_val is not None:
                        if key_val in seen_keys:
                            key_lineno = real_lineno(key_node.lineno)
                            if key_lineno in added_numbers:
                                diff_line = _find_line(lines, key_lineno)
                                findings.append(
                                    RuleFinding(
                                        rule_id="DEF006",
                                        category=FindingCategory.STATIC_ANALYSIS.value,
                                        severity=Severity.MEDIUM.value,
                                        message=f"Duplicate dictionary key '{key_val}' detected; second value overrides first.",
                                        suggestion=f"Remove or rename duplicate dictionary key '{key_val}'.",
                                        line_number=key_lineno,
                                        evidence=(diff_line.text if diff_line else "").strip(),
                                    )
                                )
                        seen_keys.add(key_val)

        # 6. DEF008: Call to print()
        if isinstance(node, ast.Call):
            call_name = get_call_name(node.func)
            if call_name == "print":
                call_lineno = real_lineno(node.lineno)
                if call_lineno in added_numbers:
                    diff_line = _find_line(lines, call_lineno)
                    findings.append(
                        RuleFinding(
                            rule_id="DEF008",
                            category=FindingCategory.STATIC_ANALYSIS.value,
                            severity=Severity.LOW.value,
                            message="print() writes directly to standard output, which is often unsuitable for production code.",
                            suggestion="Use the application's configured logging or output mechanism.",
                            line_number=call_lineno,
                            evidence=(diff_line.text if diff_line else "").strip(),
                        )
                    )

        # 7. DEF009: Assert tuple
        if isinstance(node, ast.Assert):
            if isinstance(node.test, ast.Tuple) and len(node.test.elts) > 0:
                assert_lineno = real_lineno(node.lineno)
                if assert_lineno in added_numbers:
                    diff_line = _find_line(lines, assert_lineno)
                    findings.append(
                        RuleFinding(
                            rule_id="DEF009",
                            category=FindingCategory.STATIC_ANALYSIS.value,
                            severity=Severity.HIGH.value,
                            message="assert statement called with a non-empty tuple always evaluates to True.",
                            suggestion="Use 'assert condition, message' without wrapping both in a tuple.",
                            line_number=assert_lineno,
                            evidence=(diff_line.text if diff_line else "").strip(),
                        )
                    )

    return findings
