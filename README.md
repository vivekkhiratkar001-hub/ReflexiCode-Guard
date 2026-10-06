# Automatic PR Review System

First-week local prototype for the team project. It establishes the shared PR
contracts, independent module boundaries, and an offline end-to-end demo.

## Structure

- `src/shared/`: the single source of truth for `PRContext`, `AnalysisFinding`,
  `ReviewResult`, and common enums.
- `src/context/`: loads optional plain-text repository guidance from
  `.reviewrules`.
- `src/analysis/`: a deliberately small static check for added `eval()` calls.
- `src/llm/`: optional local Ollama reviewer, defaulting to
  `qwen2.5-coder:7b`.
- `src/github/`: joins the modules into a `PRContext` to `ReviewResult`
  pipeline. It does not call GitHub yet.
- `scripts/demo.py`: constructs a sample PR locally and prints JSON output.

## Run

Requires Python 3.9+ and no third-party packages.

```powershell
python -m unittest discover -s tests
python -m scripts.demo
```

To include semantic review, start Ollama and make the selected model available,
then run:

```powershell
python -m scripts.demo --ollama
```

The offline demo uses the same pipeline and contracts. If the optional Ollama
reviewer is unavailable, static findings are retained and the result is marked
`partial`.

## Module handoff

All modules accept and return the shared models from `src/shared/models.py`;
do not define module-specific copies.

| Module | Entry point | Input | Output |
| --- | --- | --- | --- |
| Context | `load_repository_rules(root)` | Repository root | Rule text or empty string |
| Analysis | `analyze_pr(context)` | `PRContext` | `list[AnalysisFinding]` |
| LLM | `OllamaReviewer.review(context, findings, rules)` | Shared context/findings and rules | Combined findings |
| Integration | `review_pr(context, root, reviewer)` | `PRContext` and optional reviewer | `ReviewResult` |

Dependencies: Python standard library only. Tests: `python -m unittest
discover -s tests`. Limitations: the static analyzer implements only one
example rule; GitHub API/Actions integration and richer repository context are
not implemented in this first-week prototype. Ollama output is model-generated
and should be reviewed before it is treated as authoritative.
