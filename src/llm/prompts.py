"""Reusable prompt construction for evidence-based PR review."""

import json

from .context import ReviewContext


class ReviewPromptBuilder:
    """Render a complete review context into a structured-output prompt."""

    @staticmethod
    def _render_changed_code(changed_code: dict[str, str]) -> str:
        if not changed_code:
            return "No changed code was supplied."
        entries = [
            f"- {path}:\n```python\n{code}\n```\n"
            for path, code in changed_code.items()
        ]
        return "\n".join(entries)

    @staticmethod
    def _render_rules(repository_rules: list[dict[str, object]]) -> str:
        if not repository_rules:
            return "No repository rules were supplied."
        return "\n".join(
            json.dumps(rule, ensure_ascii=False, indent=2) for rule in repository_rules
        )

    @staticmethod
    def _render_findings(analysis_findings: list[dict[str, object]]) -> str:
        if not analysis_findings:
            return "No AST/static-analysis findings were supplied."
        return "\n".join(
            json.dumps(finding, ensure_ascii=False, indent=2)
            for finding in analysis_findings
        )

    @staticmethod
    def _render_changed_lines(changed_lines: dict[str, list[int]] | None) -> str:
        if not changed_lines:
            return "No changed-line information was supplied."
        return "\n".join(
            f"- {path}: {', '.join(str(line) for line in lines)}"
            for path, lines in changed_lines.items()
        )

    def build(self, context: ReviewContext) -> str:
        """Build the instructions and serialized context sent to the LLM."""
        context_json = context.model_dump_json(indent=2)
        return f"""You are a careful semantic code reviewer for a pull request.

PR / Issue Intent
- PR title: {context.pr_title or 'not provided'}
- PR description: {context.pr_description or 'not provided'}
- Issue context: {context.issue_context or 'not provided'}

Changed Code
{self._render_changed_code(context.changed_code)}

Repository Rules
{self._render_rules(context.repository_rules)}

AST / Static Analysis Findings
{self._render_findings(context.analysis_findings)}

Changed-line information
{self._render_changed_lines(context.changed_lines)}

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

The supplied context follows as JSON:

{context_json}
"""