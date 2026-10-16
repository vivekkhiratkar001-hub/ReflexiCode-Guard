# Automatic PR Review System

First-week local prototype for the team project. It establishes the shared PR
contracts, independent module boundaries, and an offline end-to-end demo.

## Structure

- `src/shared/`: the single source of truth for `PRContext`, `AnalysisFinding`,
  `ReviewResult`, and common enums.
- `src/context/`: loads optional plain-text repository guidance from
  `.reviewrules`.
- `src/analysis/`: AST parsing, changed-line localization, deterministic static rules, security vulnerability checks, complexity analysis, finding normalization, and evaluation benchmarking.
- `src/llm/`: optional local Ollama reviewer, defaulting to
  `qwen2.5-coder:7b`.
- `src/github/`: joins the modules into a `PRContext` to `ReviewResult`
  pipeline. It does not call GitHub yet.
- `scripts/demo.py`: constructs a sample PR locally and prints JSON output.
- `scripts/run_evaluation.py`: executes benchmark evaluation suite and generates machine-readable metrics.

## Run

Requires Python 3.9+ and standard library.

```powershell
python -m unittest discover -s tests
python -m scripts.run_evaluation
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

### Member 2: Analysis Engine Details

The analysis engine evaluates unified PR diffs and isolates findings specifically to added and modified lines:

- **AST & Diff Parsing**: Parses Git unified diff hunks, tracks line numbering accurately across added/deleted/renamed files, and parses AST fragments safely with fallback token scanning.
- **Deterministic Static Rules**: Bare except handlers (`DEF001`), broad exceptions / empty handlers (`DEF002`), mutable default arguments (`DEF003`), unreachable code detection (`DEF004`), singleton comparisons (`DEF005`), duplicate dictionary keys (`DEF006`), literal identity checks (`DEF007`), production print statements (`DEF008`), and tuple assertions (`DEF009`).
- **Security Checks**: Dynamic code execution (`eval`/`exec`, `SEC001`), shell injection (`os.system`/`os.popen`/`subprocess` with `shell=True`, `SEC002`), insecure deserialization (`pickle`/`marshal`/`shelve`, `SEC003`), insecure temporary files (`tempfile.mktemp`, `SEC004`), hardcoded credentials/AWS keys/tokens with secret masking in logs (`SEC005`), SQL injection in query calls (`SEC006`), directory traversal in archive extraction (`SEC007`), and weak hashing (`SEC008`).
- **Complexity Analysis**: McCabe cyclomatic complexity calculation (`CMP001`) and parameter count limits (`CMP002`).
- **Finding Normalization & Deduplication**: Canonical formatting into `AnalysisFinding` with deterministic sorting and stable deduplication.
- **Evaluation & Reproducibility**: `src/analysis/evaluation.py` calculates precision, recall, F1 score, and line localization accuracy against ground-truth benchmarks.

