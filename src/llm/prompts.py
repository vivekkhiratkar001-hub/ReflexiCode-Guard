"""Reusable prompt construction for evidence-based PR review."""

from .context import ReviewContext


class ReviewPromptBuilder:
    """Render a complete review context into a structured-output prompt."""

    def build(self, context: ReviewContext) -> str:
        """Build the instructions and serialized context sent to the LLM."""
        context_json = context.model_dump_json(indent=2)
        return f"""You are a careful semantic code reviewer for a pull request.

Review the changed code. Use the PR title, PR description, and issue context to
understand the intended change. Consider the provided repository rules and
AST/static-analysis findings as supporting evidence. Focus on actual defects and
meaningful issues; do not report style-only concerns unless a supplied repository
rule is violated.

Never invent files, line numbers, repository rules, requirements, or evidence. Report
only evidence-supported issues. Identify the exact changed line when possible. Do
not report findings on unchanged lines unless the issue is directly caused by this
change. Explain why each issue matters, give a practical correction when possible,
and avoid duplicate findings.

Return ONLY valid JSON, without markdown fences or surrounding prose, in exactly
this shape:
{{"findings":[{{"file":"path/to/file","line":1,"category":"Bug",
"severity":"High","message":"Concise actionable explanation",
"evidence":"Exact supporting code or supplied analysis evidence",
"suggestion":"Practical correction or null"}}]}}

Severity must be one of Low, Medium, High, or Critical. When no supported findings
exist, return {{"findings":[]}}. The supplied context follows as JSON:

{context_json}
"""