"""Reusable prompt construction for evidence-based PR review."""

import json
from dataclasses import dataclass
from typing import Any

from .context import ReviewContext


MAX_REVIEW_PROMPT_CHARS = 24000
_MAX_CODE_CHARS = 9000
_MAX_ANALYSIS_CHARS = 6500
_MAX_RULES_CHARS = 1500


@dataclass(frozen=True)
class PromptBuildResult:
    prompt: str
    limitations: tuple[str, ...] = ()


class ReviewPromptBuilder:
    """Render a complete review context into a structured-output prompt."""

    def build(self, context: ReviewContext) -> str:
        """Build a bounded prompt for callers that do not need coverage metadata."""
        return self.build_with_metadata(context).prompt

    def build_with_metadata(self, context: ReviewContext) -> PromptBuildResult:
        """Prioritize changed lines and findings while keeping the request bounded."""
        limitations: list[str] = []
        title = self._bounded(context.pr_title, 400, "PR title", limitations)
        description = self._bounded(
            context.pr_description, 1000, "PR description", limitations
        )
        issue_context = self._bounded(
            context.issue_context, 600, "issue context", limitations
        )
        rules = self._render_rules(context.repository_rules, limitations)
        findings = self._render_findings(context.analysis_findings, limitations)
        code = self._render_changed_code(context, limitations)

        limitation_text = (
            "\nContext limitations:\n"
            + "\n".join(f"- {item}" for item in limitations)
            + "\nTreat the included material as a prioritized subset, not complete coverage.\n"
            if limitations
            else ""
        )
        prompt = f"""You are a careful semantic code reviewer for a pull request.

PR / Issue Intent
- PR title: {title or 'not provided'}
- PR description: {description or 'not provided'}
- Issue context: {issue_context or 'not provided'}

Prioritized changed code (Changed Code; Changed-line information uses authoritative source line numbers)
{code}

Repository Rules
{rules}

AST / Static Analysis Findings
{findings}

Focus on the changed code.
Use PR/issue context to understand intent.
Use repository rules as project-specific constraints.
Use AST/static findings as supporting evidence.
Do not invent files, lines, rules, or evidence.
Do not report unsupported issues.
Do not report findings on unchanged lines unless the issue is directly caused by this change.
Prefer evidence-supported findings.
Return ONLY valid JSON, without markdown fences or surrounding prose, in exactly
this shape:
{{"findings":[{{"file":"path/to/file","line":1,"category":"Bug",
"severity":"High","message":"Concise actionable explanation",
"evidence":"Exact supporting code or supplied analysis evidence",
"suggestion":"Practical correction or null"}}]}}

Severity must be one of Low, Medium, High, or Critical. When no supported findings
exist, return {{"findings":[]}}. Include exact file and line information when
available. Explain why the issue matters and give an actionable suggestion when
appropriate.

{limitation_text}Return findings only for the supplied changed files and lines.
"""
        if len(prompt) > MAX_REVIEW_PROMPT_CHARS:
            marker = "\n\n[Context truncated to prompt hard limit.]"
            prompt = (
                prompt[: MAX_REVIEW_PROMPT_CHARS - len(marker)].rstrip() + marker
            )
            limitations.append("Prompt reached the hard character limit.")
        return PromptBuildResult(prompt=prompt, limitations=tuple(limitations))

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
            remaining = _MAX_RULES_CHARS - used
            if remaining <= 0:
                break
            if len(rendered) > remaining:
                lines.append(rendered[: max(0, remaining - 3)] + "...")
                used = _MAX_RULES_CHARS
                break
            lines.append(rendered)
            used += len(rendered) + 1
        if len(lines) < len(rules) or used >= _MAX_RULES_CHARS:
            limitations.append(
                f"Repository rules limited to {_MAX_RULES_CHARS} characters."
            )
        return "\n".join(lines)

    def _render_findings(
        self,
        findings: list[dict[str, Any]],
        limitations: list[str],
    ) -> str:
        if not findings:
            return "No AST/static-analysis findings were supplied."
        ranked = sorted(findings, key=self._finding_priority, reverse=True)
        lines: list[str] = []
        used = 0
        for finding in ranked:
            compact = {
                key: self._bounded_value(finding.get(key), limit)
                for key, limit in (
                    ("file", 180),
                    ("line", 20),
                    ("category", 40),
                    ("severity", 20),
                    ("message", 200),
                    ("evidence", 180),
                )
                if finding.get(key) is not None
            }
            rendered = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
            if used + len(rendered) + 1 > _MAX_ANALYSIS_CHARS:
                break
            lines.append(rendered)
            used += len(rendered) + 1
        if len(lines) < len(findings):
            limitations.append(
                f"Prioritized {_MAX_ANALYSIS_CHARS}-character analysis evidence budget "
                f"included {len(lines)} of {len(findings)} findings."
            )
        return "\n".join(lines) or "Analysis evidence exceeded its context budget."

    @staticmethod
    def _finding_priority(finding: dict[str, Any]) -> tuple[int, int]:
        severity = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
        }.get(str(finding.get("severity", "")).casefold(), 0)
        security = int(str(finding.get("category", "")).casefold() == "security")
        return severity + security, len(str(finding.get("message", "")))

    @staticmethod
    def _bounded_value(value: Any, limit: int) -> Any:
        if not isinstance(value, str) or len(value) <= limit:
            return value
        return value[: limit - 3].rstrip() + "..."

    def _render_changed_code(
        self,
        context: ReviewContext,
        limitations: list[str],
    ) -> str:
        if not context.changed_code:
            return "No changed code was supplied."

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
            path.replace("\\", "/").casefold(): numbers
            for path, numbers in (context.changed_lines or {}).items()
        }

        def priority(item: tuple[str, str]) -> tuple[int, int, int]:
            path, _ = item
            matching_findings = findings_by_file.get(path.replace("\\", "/").casefold(), [])
            severity = max(
                (
                    self._finding_priority(finding)[0]
                    for finding in matching_findings
                ),
                default=0,
            )
            production = int(
                not path.replace("\\", "/").casefold().startswith(
                    ("tests/", "scripts/", "docs/")
                )
            )
            return severity, len(matching_findings), production

        ranked_files = sorted(code_by_path.items(), key=priority, reverse=True)
        entries: list[str] = []
        used = 0
        included = 0
        for path_key, (path, source) in ranked_files:
            if included >= 16 or used >= _MAX_CODE_CHARS:
                break
            line_map = line_content_by_path.get(path_key)
            changed = changed_by_path.get(path_key, [])
            file_findings = findings_by_file.get(path_key, [])
            if line_map:
                selected = {
                    int(finding["line"])
                    for finding in file_findings
                    if isinstance(finding.get("line"), int)
                }
                if not selected:
                    selected.update(changed[:80])
                else:
                    for line_no in tuple(selected):
                        selected.update(
                            neighbor
                            for neighbor in (line_no - 1, line_no + 1)
                            if neighbor in line_map
                        )
                if len(selected) < 40:
                    selected.update(changed[: 40 - len(selected)])
                omitted_changed = len(set(changed) - selected)
                if omitted_changed:
                    limitations.append(
                        f"{path}: prioritized {len(set(changed) & selected)} of "
                        f"{len(set(changed))} changed lines."
                    )
                source_lines = [
                    f"L{line_no}: {line_map[line_no]}"
                    for line_no in sorted(selected)
                    if line_no in line_map
                ]
            else:
                lines = source.splitlines()
                selected_numbers = changed[:80] if changed else list(range(1, min(len(lines), 80) + 1))
                source_lines = [
                    f"L{line_no}: {lines[line_no - 1]}"
                    for line_no in selected_numbers
                    if 1 <= line_no <= len(lines)
                ]

            snippet = "\n".join(source_lines)
            if not snippet:
                continue
            remaining = min(1800, _MAX_CODE_CHARS - used)
            header = f"\n{path}\n"
            content_budget = remaining - len(header)
            if content_budget <= 0:
                break
            if len(snippet) > content_budget:
                snippet = snippet[: max(0, content_budget - 3)].rstrip() + "..."
                limitations.append(f"{path}: source excerpt truncated to fit the context budget.")
            entry = header + snippet
            entries.append(entry)
            used += len(entry) + 1
            included += 1

        if included < len(context.changed_code):
            limitations.append(
                f"Prioritized changed code for {included} of "
                f"{len(context.changed_code)} files within the {_MAX_CODE_CHARS}-character "
                "code budget (maximum 16 files)."
            )
        return "\n".join(entries) if entries else "No changed lines were available."