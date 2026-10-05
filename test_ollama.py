"""Standalone smoke test for the local Ollama review provider."""

from src.llm import OllamaProvider


def main() -> None:
    context = {
        "pr_title": "Implement the add function",
        "pr_description": "Add a helper that returns the sum of two numbers.",
        "issue_context": "The function is expected to add two numbers.",
        "repository_rules": "Keep the implementation simple and correct.",
        "changed_code": "def add(a, b):\n    return a - b",
        "analysis_findings": [],
    }

    provider = OllamaProvider()
    print(provider.review(context))


if __name__ == "__main__":
    main()