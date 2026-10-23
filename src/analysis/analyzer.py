import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from src.shared.enums import FindingCategory, Severity
from src.shared.models import AnalysisFinding, PRContext
from src.analysis.diff_parser import DiffHunk, DiffLine, FileDiff, parse_unified_diff
from src.analysis.ast_parser import parse_source_or_fragment
from src.analysis.rules.base import RuleFinding
from src.analysis.rules.static_rules import run_static_ast_rules
from src.analysis.rules.security_rules import run_security_ast_rules
from src.analysis.rules.complexity_rules import (
    ComplexityConfig,
    run_complexity_ast_rules,
)
from src.analysis.rules.lexical_rules import run_lexical_rules
from src.analysis.normalizer import deduplicate_and_sort_findings

logger = logging.getLogger(__name__)


def _is_python(path: str) -> bool:
    """Determine if a file path points to a Python file."""
    return path.lower().endswith(".py")


def _report_print_findings(path: str) -> bool:
    """Keep print diagnostics focused on application code, not output surfaces."""
    normalized = path.replace("\\", "/").casefold()
    parts = normalized.split("/")
    return not (
        parts[0] in {"tests", "scripts"}
        or parts[-1].startswith("test_")
        or parts[-1] == "test_ollama.py"
        or normalized == "src/github/cli.py"
    )


def analyze_diff(
    diff: str,
    changed_files: Optional[Sequence[str]] = None,
    complexity_config: Optional[ComplexityConfig] = None,
) -> List[AnalysisFinding]:
    """Analyze changed lines in a unified diff and return normalized findings."""
    if not diff or not diff.strip():
        return []

    file_diffs = parse_unified_diff(diff)
    all_findings: List[AnalysisFinding] = []

    for file_diff in file_diffs:
        target_path = file_diff.target_path
        if not _is_python(target_path):
            continue
        if file_diff.is_binary or file_diff.change_type == "deleted":
            continue

        file_raw_findings: List[RuleFinding] = []

        for hunk in file_diff.hunks:
            if not hunk.lines:
                continue

            target_lines = [l for l in hunk.lines if not l.is_deleted]
            if not target_lines:
                continue

            # Extract line texts for AST fragment parsing from the new state of the code
            lines_text = [l.text for l in target_lines]
            tree, line_offset, _ = parse_source_or_fragment(lines_text)

            hunk_rules: List[RuleFinding] = []
            if tree is not None:
                try:
                    hunk_rules.extend(
                        run_static_ast_rules(
                            tree,
                            target_lines,
                            line_offset,
                            report_print=_report_print_findings(target_path),
                        )
                    )
                except Exception as exc:
                    logger.warning("Static AST rules failed for %s: %s", target_path, exc)

                try:
                    hunk_rules.extend(run_security_ast_rules(tree, target_lines, line_offset))
                except Exception as exc:
                    logger.warning("Security AST rules failed for %s: %s", target_path, exc)

                try:
                    hunk_rules.extend(
                        run_complexity_ast_rules(tree, target_lines, line_offset, complexity_config)
                    )
                except Exception as exc:
                    logger.warning("Complexity AST rules failed for %s: %s", target_path, exc)

            # If AST rules produced no findings (e.g. fragment or malformed code), use lexical fallback
            if not hunk_rules:
                try:
                    hunk_rules.extend(
                        run_lexical_rules(
                            target_lines,
                            report_print=_report_print_findings(target_path),
                        )
                    )
                except Exception as exc:
                    logger.warning("Lexical rules failed for %s: %s", target_path, exc)
            elif tree is not None:
                # Also ensure pattern-based secret checks ran for added lines in the hunk
                try:
                    secret_findings = [
                        f for f in run_security_ast_rules(tree, target_lines, line_offset)
                        if f.rule_id == "SEC005"
                    ]
                    for sf in secret_findings:
                        if sf not in hunk_rules:
                            hunk_rules.append(sf)
                except Exception as exc:
                    logger.warning("Secret checks failed for %s: %s", target_path, exc)

            file_raw_findings.extend(hunk_rules)

        # Normalize, deduplicate, and sort findings for this file
        normalized = deduplicate_and_sort_findings(target_path, file_raw_findings)
        all_findings.extend(normalized)

    # Deterministic final sort across all files
    return sorted(
        all_findings,
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
    return analyze_diff(context.diff, changed_files=context.changed_files)
