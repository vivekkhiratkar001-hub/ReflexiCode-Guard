"""Standalone smoke test for the local Ollama review provider."""

import logging
from time import monotonic

from src.llm import OllamaProvider, ReviewContextBuilder, ReviewPipeline


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    context = ReviewContextBuilder().build(
        pr_title="Fix user addition function",
        pr_description="Correct the implementation of the addition function.",
        issue_context="The function should add two numbers. The incorrect statement is on line 2: return a - b.",
        changed_code={"src/example.py": "def add(a, b):\n    return a - b"},
        repository_rules=[],
        analysis_findings=[],
        changed_lines={"src/example.py": [2]},
        changed_line_content={"src/example.py": {2: "    return a - b"}},
    )

    pipeline = ReviewPipeline(OllamaProvider())
    started = monotonic()
    findings = pipeline.review(context)
    elapsed = monotonic() - started
    if pipeline.last_review_failures:
        raise RuntimeError("Semantic smoke test failed: " + " ".join(pipeline.last_review_failures))
    grounded = [
        finding
        for finding in findings
        if finding.file == "src/example.py"
        and finding.line == 2
        and "return a - b" in finding.evidence
    ]
    if not grounded:
        raise RuntimeError(
            "Ollama completed without a validated finding for the known addition defect."
        )
    print(
        {
            "semantic_status": "success_with_findings",
            "elapsed_seconds": round(elapsed, 2),
            "findings": [finding.model_dump() for finding in findings],
        }
    )


if __name__ == "__main__":
    main()