"""Analysis package: deterministic AST-based static analysis, security checks, and finding localization."""

from .analyzer import analyze_diff, analyze_pr
from .diff_parser import DiffHunk, DiffLine, FileDiff, parse_unified_diff
from .ast_parser import parse_source_or_fragment, tokenize_source
from .normalizer import deduplicate_and_sort_findings, normalize_finding
from .rules import (
    ComplexityConfig,
    RuleFinding,
    calculate_cyclomatic_complexity,
    count_parameters,
    run_complexity_ast_rules,
    run_lexical_rules,
    run_security_ast_rules,
    run_static_ast_rules,
)

__all__ = [
    "analyze_pr",
    "analyze_diff",
    "parse_unified_diff",
    "FileDiff",
    "DiffHunk",
    "DiffLine",
    "parse_source_or_fragment",
    "tokenize_source",
    "deduplicate_and_sort_findings",
    "normalize_finding",
    "ComplexityConfig",
    "RuleFinding",
    "calculate_cyclomatic_complexity",
    "count_parameters",
    "run_complexity_ast_rules",
    "run_lexical_rules",
    "run_security_ast_rules",
    "run_static_ast_rules",
]
