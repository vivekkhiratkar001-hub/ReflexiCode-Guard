"""Complexity analysis rules: McCabe cyclomatic complexity and parameter count."""

import ast
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

from src.shared.enums import FindingCategory, Severity
from src.analysis.diff_parser import DiffLine
from src.analysis.rules.base import RuleFinding


# Documented standard thresholds
DEFAULT_COMPLEXITY_WARNING_THRESHOLD = 10
DEFAULT_COMPLEXITY_HIGH_THRESHOLD = 15
DEFAULT_MAX_PARAMETER_COUNT = 7


@dataclass(frozen=True)
class ComplexityConfig:
    warning_threshold: int = DEFAULT_COMPLEXITY_WARNING_THRESHOLD
    high_threshold: int = DEFAULT_COMPLEXITY_HIGH_THRESHOLD
    max_parameters: int = DEFAULT_MAX_PARAMETER_COUNT


def calculate_cyclomatic_complexity(func_node: ast.AST) -> int:
    """Calculate McCabe cyclomatic complexity for a function AST node.

    Base complexity is 1.
    Each branch/decision point increments complexity:
    - if, elif
    - for, while, async for
    - except handler
    - with, async with
    - assert
    - ternary if (IfExp)
    - comprehension filtering conditions
    - boolean operations (and, or)

    Nested functions are treated as independent units and not double-counted.
    """
    complexity = 1

    for child in ast.iter_child_nodes(func_node):
        # Do not recurse into nested function definitions; they have their own complexity calculation
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        for subnode in ast.walk(child):
            # Do not count nodes inside nested functions
            if isinstance(subnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            if isinstance(subnode, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(subnode, ast.ExceptHandler):
                complexity += 1
            elif isinstance(subnode, (ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(subnode, ast.Assert):
                complexity += 1
            elif isinstance(subnode, ast.IfExp):
                complexity += 1
            elif isinstance(subnode, ast.comprehension):
                complexity += len(subnode.ifs)
            elif isinstance(subnode, ast.BoolOp):
                # e.g., 'a and b' has 2 values -> 1 branch point; 'a and b and c' has 3 values -> 2 branch points
                complexity += max(0, len(subnode.values) - 1)

    return complexity


def count_parameters(func_node: ast.AST) -> int:
    """Count total declared parameters for a function (args, posonlyargs, kwonlyargs, vararg, kwarg)."""
    if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return 0
    args = func_node.args
    total = len(args.args) + len(args.kwonlyargs)
    if hasattr(args, "posonlyargs"):
        total += len(args.posonlyargs)
    if args.vararg:
        total += 1
    if args.kwarg:
        total += 1
    return total


def run_complexity_ast_rules(
    tree: ast.AST,
    lines: Sequence[DiffLine],
    line_offset: int,
    config: Optional[ComplexityConfig] = None,
) -> List[RuleFinding]:
    """Run complexity checks on functions in the AST."""
    if config is None:
        config = ComplexityConfig()

    findings: List[RuleFinding] = []
    added_numbers = {l.number for l in lines if l.is_added and l.number is not None}

    def real_lineno(ast_lineno: int) -> int:
        local_idx = ast_lineno - line_offset
        if 1 <= local_idx <= len(lines):
            return lines[local_idx - 1].number or 0
        return 0

    def find_diff_line(target_no: int) -> Optional[DiffLine]:
        for line in lines:
            if line.number == target_no:
                return line
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            def_lineno = real_lineno(node.lineno)

            # Check if this function has any changed lines in the diff
            # Find all line numbers spanned by this function
            end_lineno = getattr(node, "end_lineno", node.lineno)
            func_line_nums = {
                real_lineno(i)
                for i in range(node.lineno, (end_lineno or node.lineno) + 1)
            }
            has_changes = bool(func_line_nums & added_numbers)
            if not has_changes and def_lineno not in added_numbers:
                continue

            # Target line is function def line if added, or the first added line inside the function
            if def_lineno in added_numbers:
                target_lineno = def_lineno
            else:
                overlapping = sorted(func_line_nums & added_numbers)
                target_lineno = overlapping[0] if overlapping else def_lineno

            diff_line = find_diff_line(target_lineno)
            evidence = (diff_line.text if diff_line else f"def {node.name}(...)").strip()

            # 1. Cyclomatic Complexity
            complexity = calculate_cyclomatic_complexity(node)
            if complexity > config.warning_threshold:
                severity = (
                    Severity.HIGH.value
                    if complexity > config.high_threshold
                    else Severity.MEDIUM.value
                )
                findings.append(
                    RuleFinding(
                        rule_id="CMP001",
                        category=FindingCategory.STATIC_ANALYSIS.value,
                        severity=severity,
                        message=(
                            f"Function '{node.name}' has high cyclomatic complexity of {complexity} "
                            f"(threshold: {config.warning_threshold})."
                        ),
                        suggestion="Refactor complex conditional branches, loops, and handlers into smaller focused helper functions.",
                        line_number=target_lineno,
                        evidence=evidence,
                    )
                )

            # 2. Parameter Count
            param_count = count_parameters(node)
            if param_count > config.max_parameters:
                findings.append(
                    RuleFinding(
                        rule_id="CMP002",
                        category=FindingCategory.STATIC_ANALYSIS.value,
                        severity=Severity.LOW.value,
                        message=(
                            f"Function '{node.name}' defines {param_count} parameters "
                            f"(maximum recommended threshold: {config.max_parameters})."
                        ),
                        suggestion="Group related parameters into a configuration object, dataclass, or dictionary.",
                        line_number=target_lineno,
                        evidence=evidence,
                    )
                )

    return findings
