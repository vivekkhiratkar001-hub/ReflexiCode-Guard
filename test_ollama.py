"""Standalone smoke test for the local Ollama review provider."""

from src.llm import OllamaProvider, ReviewContextBuilder, ReviewPipeline


def main() -> None:
    context = ReviewContextBuilder().build(
        pr_title="Fix user addition function",
        pr_description="Correct the implementation of the addition function.",
        issue_context="The function should add two numbers.",
        changed_code={"src/example.py": "def add(a, b):\n    return a - b"},
        repository_rules=[],
        analysis_findings=[],
        changed_lines={"src/example.py": [1, 2]},
    )

    pipeline = ReviewPipeline(OllamaProvider())
    findings = pipeline.review(context)
    print({"findings": [finding.model_dump() for finding in findings]})


if __name__ == "__main__":
    main()