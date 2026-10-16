# Member 3 — LLM / Semantic Review Module

## 1. Responsibility

Member 3 is responsible for the LLM-focused review pipeline used by ReflexiCode-Guard.

It handles:

- LLM integration
- local Ollama communication
- Qwen2.5-Coder model use
- review prompt construction
- PR and issue context preparation
- response parsing
- finding validation and deduplication
- end-to-end review orchestration
- evaluation of predicted findings against ground truth

This module does not own repository- or GitHub-specific publishing logic. It focuses on producing a validated review result from the available context.

## 2. Architecture

The actual pipeline is:

PR / Issue Context
        +
Changed Code
        +
Changed Lines
        +
Repository Rules
        +
AST / Static Analysis Findings
        ↓
ReviewContext
        ↓
Prompt Builder
        ↓
Ollama / Qwen2.5-Coder
        ↓
JSON Response
        ↓
Parser
        ↓
Validation
        ↓
ReviewFinding[]
        ↓
Evaluation

Repository rules and static-analysis findings are treated as supporting context. They are used to help the model understand constraints and evidence, but they are not automatically treated as violations by the LLM or by the parser.

## 3. File Structure

The actual files under src/llm are:

- provider.py
  → common provider interface used by review models

- ollama_provider.py
  → local Ollama implementation and HTTP handling

- mock_provider.py
  → deterministic offline provider for tests and local development

- schemas.py
  → Pydantic model definitions for review findings and envelopes

- context.py
  → ReviewContext and ReviewContextBuilder for normalizing PR and code context

- prompts.py
  → prompt builder that assembles PR intent, changed code, rules, and analysis evidence

- parser.py
  → JSON extraction, schema validation, file/line/evidence checks, and duplicate filtering

- pipeline.py
  → end-to-end orchestration from context to final findings

- evaluation.py
  → deterministic evaluator for TP / FP / FN / Precision / Recall / F1

- __init__.py
  → public API exports for the LLM package

## 4. LLM Integration

The active project implementation uses a local Ollama model:

- model: qwen2.5-coder:7b
- API endpoint: http://localhost:11434/api/generate
- request format: JSON with model, prompt, stream, and format fields
- timeout: configured in OllamaProvider

The provider follows the pattern:

1. Build the review prompt.
2. Send it to the local Ollama endpoint.
3. Validate the HTTP response.
4. Read the result JSON.
5. Extract the response text.
6. Return raw JSON text for downstream parsing.

The code also includes MockProvider for tests and offline execution. This keeps the review pipeline valid without requiring a running Ollama server during normal unit tests.

## 5. Review Context

The review context is built in ReviewContextBuilder and contains the information that the LLM should use when forming a review:

- PR title
- PR description
- issue context
- changed code
- changed lines
- repository rules
- AST/static-analysis findings

This is stored in ReviewContext, which is then used by the prompt builder and pipeline.

These inputs are useful because the model needs both the intent of the PR and the concrete code being changed. The repository rules and static-analysis findings provide guardrails and supporting evidence, while the changed code tells the model what is relevant.

## 6. Prompt Engineering

The prompt builder creates a structured review prompt that tells the model to:

- focus on the changed code
- use PR/issue context to understand intent
- use repository rules as project-specific constraints
- use AST/static findings as supporting evidence
- avoid unsupported claims
- prefer evidence-based findings
- include exact file and line information when available
- provide actionable suggestions
- return only valid JSON

The implementation is intentionally conservative. It does not ask the model to invent files, rules, or evidence.

## 7. ReviewFinding Schema

The actual schema is defined in schemas.py.

ReviewFinding includes:

- file: string
- line: positive integer
- category: string
- severity: one of Low, Medium, High, or Critical
- message: string
- evidence: string
- suggestion: optional string

The JSON envelope is a top-level object with a findings list.

## 8. Validation and Quality Checks

The parser performs deterministic validation to reduce unsupported or noisy output.

Implemented checks include:

- structural validation of the JSON envelope
- non-empty field validation
- file validation against changed_code
- changed-line validation when changed_lines is available
- evidence validation against the supplied changed code
- support for static-analysis evidence as a valid justification path
- duplicate detection for obvious repeated issues
- malformed or unsupported output rejection

This does not guarantee that an LLM can never hallucinate, but malformed or unsupported findings cause the whole response to be rejected rather than partially filtering out the bad item. The current parser is intentionally all-or-nothing: a single invalid finding can fail the entire response.

## 9. End-to-End Pipeline

The actual orchestration is in ReviewPipeline.

The flow is:

ReviewContext
→ Prompt
→ Provider
→ JSON
→ Parser
→ Validation/Deduplication
→ ReviewFinding[]

This is the main high-level contract for the LLM review module.

## 10. Evaluation

Module 5 adds a separate evaluation system in evaluation.py.

It supports:

- ground-truth findings
- predicted findings
- deterministic matching
- configurable line tolerance
- TP / FP / FN
- Precision
- Recall
- F1

The matching is intentionally simple and deterministic. It compares canonicalized file, category, and line number using the configured tolerance window. Severity is not currently used as a matching criterion; matching is based on canonicalized file, category, and line/tolerance. It does not rely on embeddings or other complex matching methods.

The current evaluation dataset is manually constructed and small. It is designed to validate the evaluation framework, not to establish general model performance.

The evaluation matching logic is deterministic and uses file/category constraints plus the configured line tolerance; exact zero tolerance means exact line matching. For the runtime review pipeline, `changed_lines` is treated as the source of truth when present, and the parser rejects findings whose line is not in that map. If no `changed_lines` information is available, the validation step skips line membership checks rather than rejecting every finding.

Naming is intentionally not fully flattened across the project: `ReviewContext.analysis_findings` is the runtime review input, while `EvaluationCase.static_analysis_findings` is the evaluation-fixture field used in the separate scoring dataset.

## 11. Testing

The repository’s actual project test commands are:

- python -m pytest -q
- python test_ollama.py

The project test suite includes unit tests for:

- prompt and context behavior
- parser validation
- provider behavior
- end-to-end pipeline behavior
- evaluation logic

The latest verified session result showed the project test suite passing before this documentation update, and the Ollama smoke test also returned structured findings successfully.

## 12. Integration Contract for Other Team Members

To use Member 3 correctly, the rest of the team should provide the following context:

- PR title
- PR description
- issue context
- changed code
- changed lines if available
- repository rules
- AST/static-analysis findings

The final output contract for Member 3 is:

- ReviewFinding[]

Member 3 does not need GitHub-specific code inside the LLM module. It consumes structured review context and returns structured review findings.

## 13. How Another Developer Can Use It

The actual public API is available through src/llm/__init__.py.

Example:

```python
from src.llm import ReviewContextBuilder, ReviewPipeline, MockProvider

context = ReviewContextBuilder().build(
    pr_title="Fix direct database access in user service",
    pr_description="Use the repository layer instead of direct SQL.",
    issue_context="All database access must go through repository classes.",
    changed_code={
        "src/services/user_service.py": (
            "def get_user(user_id):\n"
            "    query = f\"SELECT * FROM users WHERE id = {user_id}\"\n"
            "    return db.execute(query)\n"
        )
    },
    repository_rules=[{"rule": "Database access must use repository layer."}],
    analysis_findings=[{"file": "src/services/user_service.py", "line": 2, "message": "Potential SQL injection."}],
    changed_lines={"src/services/user_service.py": [1, 2, 3]},
)

pipeline = ReviewPipeline(MockProvider('{"findings": []}'))
findings = pipeline.review(context)
```

This uses the real public API currently exported by the project.

## 14. Known Limitations

The real limitations of the current implementation are:

- Ollama must be available for real LLM execution.
- Evaluation is based on a small manually constructed dataset.
- LLM output is not guaranteed to be free from hallucination.
- Any real Member 1/2 production integration still depends on the actual schemas and modules when they become available in the workspace.

## 15. Handoff Checklist

- [x] LLM provider interface completed
- [x] Ollama provider completed
- [x] Prompt builder completed
- [x] Review context completed
- [x] Response parser completed
- [x] Finding validation completed
- [x] End-to-end pipeline completed
- [x] Evaluation completed
- [x] Tests passing
- [x] Documentation completed

## Summary

The Member 3 LLM / semantic review module is a focused, testable review pipeline that turns PR context and code changes into structured review findings, validates the output, and supports deterministic evaluation. It is ready for team handoff as a clean integration boundary for the rest of the project.
