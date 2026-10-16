"""Reusable prompt construction for evidence-based PR review."""

import json
from dataclasses import dataclass
from typing import Any

from .context import ReviewContext


MAX_REVIEW_PROMPT_CHARS = 10000
DEFAULT_REVIEW_CODE_CHARS = 4000
DEFAULT_REVIEW_ANALYSIS_CHARS = 2500
DEFAULT_REVIEW_RULES_CHARS = 600
DEFAULT_REVIEW_MAX_FILES = 5
MAX_SEMANTIC_FINDINGS = 5
_MAX_LINE_CHARS = 500


@dataclass(frozen=True)
class PromptBuildResult:
    prompt: str
    limitations: tuple[str, ...] = ()
    included_files: tuple[str, ...] = ()
    included_changed_lines: int = 0
    included_findings: int = 0
    available_files: int = 0
    available_changed_lines: int = 0
    available_findings: int = 0
    supplied_findings: int = 0


class ReviewPromptBuilder:
    """Render a prioritized, bounded subset of review context."""

    def __init__(
        self,
        *,
        max_prompt_chars: int = MAX_REVIEW_PROMPT_CHARS,
        code_chars: int = DEFAULT_REVIEW_CODE_CHARS,
        analysis_chars: int = DEFAULT_REVIEW_ANALYSIS_CHARS,
        rules_chars: int = DEFAULT_REVIEW_RULES_CHARS,
        max_files: int = DEFAULT_REVIEW_MAX_FILES,
    ) -> None:
        for name, value in (
            ("max_prompt_chars", max_prompt_chars),
            ("code_chars", code_chars),
            ("analysis_chars", analysis_chars),
            ("rules_chars", rules_chars),
            ("max_files", max_files),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if code_chars + analysis_chars + rules_chars + 1500 >= max_prompt_chars:
            raise ValueError(
                "code_chars, analysis_chars, and rules_chars must leave room "
                "for review instructions and PR metadata."
            )
        self.max_prompt_chars = max_prompt_chars
        self.code_chars = code_chars
        self.analysis_chars = analysis_chars
        self.rules_chars = rules_chars
        self.max_files = max_files

    def build(self, context: ReviewContext) -> str:
        """Build a bounded prompt for callers that do not need coverage metadata."""
        return self.build_with_metadata(context).prompt

    def build_with_metadata(self, context: ReviewContext) -> PromptBuildResult:
        """Prioritize production changes and credible evidence within fixed budgets."""
        limitations: list[str] = []
        title = self._bounded(context.pr_title, 150, "PR title", limitations)
        description = self._bounded(
            context.pr_description, 350, "PR description", limitations
        )
        issue_context = self._bounded(
            context.issue_context, 200, "issue context", limitations
        )
        rules = self._render_rules(context.repository_rules, limitations)
        code, included_files, included_changed_lines = self._render_changed_code(
            context, limitations
        )
        included_file_keys = {
            path.replace("\\", "/").casefold() for path in included_files
        }
        supplied_findings = [
            finding
            for finding in context.analysis_findings
            if str(finding.get("file", "")).replace("\\", "/").casefold()
            in included_file_keys
        ]
        excluded_findings = len(context.analysis_findings) - len(supplied_findings)
        if excluded_findings:
            limitations.append(
                f"Excluded {excluded_findings} analysis findings for files outside "
                "the selected semantic-review scope."
            )
        findings, included_findings = self._render_findings(
            supplied_findings, limitations
        )

        limitation_text = ""
        if limitations:
            shown = [item[:120] for item in limitations[:4]]
            limitation_text = (
                "\nContext limitations:\n"
                + "\n".join(f"- {item}" for item in shown)
                + (
                    f"\n- {len(limitations) - len(shown)} additional omissions."
                    if len(limitations) > len(shown)
                    else ""
                )
                + "\nTreat the included material as a prioritized subset, not complete coverage.\n"
            )
        prompt = f"""You are a careful semantic code reviewer for a pull request.

Selected changed files: {len(included_files)}
PR / Issue Intent
- PR title: {title or 'not provided'}
- PR description: {description or 'not provided'}
- Issue context: {issue_context or 'not provided'}

Prioritized changed code (Changed Code; file paths and source line numbers are authoritative)
{code}

Repository Rules
{rules}

AST / Static Analysis Findings (prioritized subset)
{findings}

Review only the supplied files and changed lines; omitted files were not reviewed.
Focus on the changed code. Changed-line information and source paths are authoritative.
Use PR/issue context to understand intent.
Use repository rules as project-specific constraints.
Use AST/static findings as supporting evidence.
Do not report findings on unchanged lines unless the issue is directly caused by this change.
Return at most {MAX_SEMANTIC_FINDINGS} important, evidence-supported findings.
Do not invent files, lines, rules, or evidence. Do not report unsupported issues.
Prefer security and correctness defects over style or speculative concerns.
Return ONLY valid JSON, without markdown fences or surrounding prose, in exactly
this shape:
{{"findings":[{{"file":"path/to/file","line":1,"category":"Bug",
"severity":"High","message":"Concise actionable explanation",
"evidence":"Exact supporting code or supplied analysis evidence",
"suggestion":"Practical correction or null"}}]}}

Severity must be one of Low, Medium, High, or Critical. When no supported findings
exist, return {{"findings":[]}}. Use exact supplied file paths and source line
numbers. Keep each finding concise.
{limitation_text}"""
        if len(prompt) > self.max_prompt_chars:
            raise ValueError(
                "Review prompt exceeded its configured hard limit after applying "
                "section budgets; reduce a section budget."
            )
        return PromptBuildResult(
            prompt=prompt,
            limitations=tuple(limitations),
            included_files=tuple(included_files),
            included_changed_lines=included_changed_lines,
            included_findings=included_findings,
            available_files=len(context.changed_code),
            available_changed_lines=sum(
                len(set(lines)) for lines in (context.changed_lines or {}).values()
            ),
            available_findings=len(context.analysis_findings),
            supplied_findings=len(supplied_findings),
        )

    @staticmethod
    def _bounded(
        value: str,
        limit: int,
        label: str,
        limitations: list[str],
    ) -> str:
        if len(value) <= limit:
            return value
        limitations.append(f"{label} truncated from {len(value)} to {limit} characters.")
        return value[: limit - 3].rstrip() + "..."

    def _render_rules(
        self,
        rules: list[dict[str, Any]],
        limitations: list[str],
    ) -> str:
        if not rules:
            return "No repository rules were supplied."
        lines: list[str] = []
        used = 0
        for rule in rules:
            rendered = json.dumps(rule, ensure_ascii=False, separators=(",", ":"))
            remaining = self.rules_chars - used
            if remaining <= 0:
                break
            if len(rendered) > remaining:
                if remaining > 3:
                    lines.append(rendered[: remaining - 3] + "...")
                used = self.rules_chars
                break
            lines.append(rendered)
            used += len(rendered) + 1
        if len(lines) < len(rules) or used >= self.rules_chars:
            limitations.append(f"Repository rules limited to {self.rules_chars} characters.")
        return "\n".join(lines) or "Repository rules exceeded their context budget."

    def _render_findings(
        self,
        findings: list[dict[str, Any]],
        limitations: list[str],
    ) -> tuple[str, int]:
        if not findings:
            return "No AST/static-analysis findings were supplied.", 0
        ranked = sorted(findings, key=self._finding_priority, reverse=True)
        lines: list[str] = []
        used = 0
        for finding in ranked:
            if len(lines) >= MAX_SEMANTIC_FINDINGS:
                break
            compact = {
                key: self._bounded_value(finding.get(key), limit)
                for key, limit in (
                    ("file", 180),
                    ("line", 20),
                    ("category", 40),
                    ("severity", 20),
                    ("message", 180),
                    ("evidence", 160),
                )
                if finding.get(key) is not None
            }
            rendered = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
            if used + len(rendered) + 1 > self.analysis_chars:
                break
            lines.append(rendered)
            used += len(rendered) + 1
        if len(lines) < len(findings):
            limitations.append(
                f"Prioritized {self.analysis_chars}-character analysis budget included "
                f"{len(lines)} of {len(findings)} findings (maximum "
                f"{MAX_SEMANTIC_FINDINGS})."
            )
        return "\n".join(lines) or "Analysis evidence exceeded its context budget.", len(lines)

    @staticmethod
    def _finding_priority(finding: dict[str, Any]) -> tuple[int, int, int]:
        severity = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
        }.get(str(finding.get("severity", "")).casefold(), 0)
        security = int(str(finding.get("category", "")).casefold() == "security")
        confidence = int(bool(finding.get("evidence")))
        return severity + security, confidence, len(str(finding.get("message", "")))

    @staticmethod
    def _bounded_value(value: Any, limit: int) -> Any:
        if not isinstance(value, str) or len(value) <= limit:
            return value
        return value[: limit - 3].rstrip() + "..."

    def _render_changed_code(
        self,
        context: ReviewContext,
        limitations: list[str],
    ) -> tuple[str, list[str], int]:
        if not context.changed_code:
            return "No changed code was supplied.", [], 0

        findings_by_file: dict[str, list[dict[str, Any]]] = {}
        for finding in context.analysis_findings:
            path = str(finding.get("file", "")).replace("\\", "/").casefold()
            findings_by_file.setdefault(path, []).append(finding)

        code_by_path = {
            path.replace("\\", "/").casefold(): (path, code)
            for path, code in context.changed_code.items()
        }
        line_content_by_path = {
            path.replace("\\", "/").casefold(): content
            for path, content in context.changed_line_content.items()
        }
        changed_by_path = {
            path.replace("\\", "/").casefold(): set(numbers)
            for path, numbers in (context.changed_lines or {}).items()
        }

        def priority(item: tuple[str, tuple[str, str]]) -> tuple[int, int, int, int, str]:
            path_key, (path, _) = item
            normalized = path.replace("\\", "/").casefold()
            matching = findings_by_file.get(path_key, [])
            severity = max(
                (self._finding_priority(finding)[0] for finding in matching),
                default=0,
            )
            parts = normalized.split("/")
            production = int(
                parts[0] not in {"tests", "scripts", "docs", "examples"}
                and not parts[-1].startswith("test_")
            )
            changed_count = len(changed_by_path.get(path_key, set()))
            manageable = int(0 < changed_count <= 40)
            return (
                severity,
                production,
                manageable,
                len(matching),
                normalized,
            )

        ranked_files = sorted(code_by_path.items(), key=priority, reverse=True)
        selected_files = ranked_files[: self.max_files]
        if len(selected_files) < len(ranked_files):
            limitations.append(
                f"Selected {len(selected_files)} of {len(ranked_files)} changed files "
                f"(maximum {self.max_files})."
            )

        entries: list[str] = []
        included_paths: list[str] = []
        included_changed_lines = 0
        used = 0
        for position, (path_key, (path, source)) in enumerate(selected_files):
            remaining_slots = len(selected_files) - position
            file_budget = min(
                1400,
                (self.code_chars - used) // max(1, remaining_slots),
            )
            if file_budget < 1:
                continue

            line_map = line_content_by_path.get(path_key)
            changed = changed_by_path.get(path_key, set())
            if not line_map:
                raw_lines = source.splitlines()
                line_map = {
                    number: raw_lines[number - 1]
                    for number in changed
                    if 1 <= number <= len(raw_lines)
                }
            if not line_map or not changed:
                limitations.append(f"{path}: no changed-line content was available.")
                continue

            finding_lines = {
                finding.get("line")
                for finding in findings_by_file.get(path_key, [])
                if finding.get("line") is not None
            }

            sorted_lines = sorted(changed)
            file_entries: list[str] = []

            for line_num in sorted_lines:
                if line_num not in line_map:
                    continue

                content = line_map[line_num]
                # Prioritize findings on these lines
                if line_num in finding_lines:
                    entry = f"Line {line_num}: {content}"
                else:
                    entry = f"Line {line_num}: {content}"

                if used + len(entry) + 1 > self.code_chars:
                    break

                file_entries.append(entry)
                used += len(entry) + 1

            if file_entries:
                entries.append(f"File: {path}\n" + "\n".join(file_entries))
                included_paths.append(path)
                included_changed_lines += sum(
                    len(line_map[ln]) for ln in sorted_lines if ln in line_map
                )

        return "\n\n".join(entries), included_paths, included_changed_lines
