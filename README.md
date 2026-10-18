# ReflexiCode Guard

Automatic pull-request review with deterministic analysis, optional semantic
review, and GitHub inline-comment publishing.

## Components

- `src/shared/`: shared `PRContext`, `AnalysisFinding`, `ReviewResult`, and
  severity/category/status enums.
- `src/context/`: reads optional plain-text repository guidance from
  `.reviewrules`.
- `src/analysis/`: parses unified diffs and checks changed Python lines with
  deterministic static, security, and complexity rules.
- `src/llm/`: constructs review context and prompts, validates Ollama or mock
  responses, and evaluates semantic findings.
- `src/github/pipeline.py`: local pipeline returning the shared `ReviewResult`.
- `src/github/`: GitHub collection, comment mapping, duplicate filtering,
  dry-run output, and optional review publishing.

The GitHub integration uses transport-specific PR/comment models at its API
boundary. Findings and analysis are normalized to the shared contracts before
they are passed to the review pipeline and publisher.

Branch selection and the Member 4 integration boundary are documented in
[docs/integration-notes.md](docs/integration-notes.md).

## Setup and tests

Python 3.11+ is required. Install dependencies and run the full suite:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m unittest discover -s tests
```

The suite includes offline mock-provider and mocked-GitHub tests; it does not
require an Ollama service or GitHub credentials.

Run the local demo and analyzer evaluation with:

```powershell
python -m scripts.demo
python -m scripts.run_evaluation
```

## GitHub CLI

The default is a read-only dry run. Supply a token only when reading private
repositories or explicitly publishing comments.

```powershell
python -m src.github --owner OWNER --repo REPOSITORY --pr-number NUMBER --dry-run --use-mock-llm
```

### Local Ollama semantic review

Install and start Ollama, then install the model you intend to use (the default
is `qwen2.5-coder:7b`):

```powershell
ollama pull qwen2.5-coder:7b
ollama list
```

The GitHub CLI uses Ollama's local `/api/generate` endpoint with JSON output.
Select an already-installed model with `--ollama-model` or
`REFLEXICODE_OLLAMA_MODEL`; the application does not download or silently switch
models. The default timeout is 120 seconds (configurable from 1 to 300), output
is limited to 512 tokens, and Ollama `num_ctx` defaults to 4096. The default
10,000-character prompt cap estimates to about 2,500 input tokens, leaving
room for output and template overhead. Use
`--ollama-timeout`, `--ollama-num-predict`, and `--ollama-num-ctx`, or
`REFLEXICODE_OLLAMA_TIMEOUT`, `REFLEXICODE_OLLAMA_NUM_PREDICT`, and
`REFLEXICODE_OLLAMA_NUM_CTX`.

Each semantic review is one bounded request. Defaults are 10,000 prompt
characters overall, 4,000 characters of changed code with nearby hunk context,
2,500 characters of supporting analysis, at most five selected files, and at
most three semantic findings. Character-to-token conversion is approximate
(logged as an estimate); hardware, model loading, and generated output affect
runtime. Configure limits with `--review-prompt-chars`, `--review-code-chars`,
`--review-analysis-chars`, `--review-max-files`, `--review-max-findings`, or
their `REFLEXICODE_REVIEW_*` environment variables.

The CLI reports semantic status separately from total findings:
`success_with_findings`, `success_no_findings`, `partial`, `failed`,
`partial_failure`, or `not_run`. Partial scope lists omitted files; successful
empty output is distinct from failed or invalid model output. Deterministic
static/security findings are produced locally and retained on semantic failure;
LLM findings are separately validated against selected files, changed lines,
supported categories/severities, and source evidence. Neither approach proves
that every defect was found. Logs contain request metadata and timings, not
prompts or source text. Dry runs do not publish comments.

The offline unit tests do not need Ollama or GitHub credentials. For an
explicit live local-model smoke test, run `python test_ollama.py`; this exercises
the semantic provider, parser, and pipeline but does not prove that a GitHub PR
integration will complete. A real PR dry run also requires network access to
GitHub (and a token for private repositories). Do not describe live integration
as verified unless that run completed successfully.

Publishing requires a token with pull-request write permission and an explicit
`--publish` flag:

```powershell
$env:GITHUB_TOKEN = "..."
python -m src.github --owner OWNER --repo REPOSITORY --pr-number NUMBER --publish
```

GitHub Actions runs the test suite on the pull-request merge commit without
credentials. The API-backed review is intentionally not run before merge; run
the documented local dry run from trusted code for a non-draft, same-repository
pull request. CI never publishes comments. See
[docs/member1_github.md](docs/member1_github.md) for configuration and
[docs/member3_llm.md](docs/member3_llm.md) for semantic-review details.

## Analysis and evaluation

The deterministic analyzer localizes findings to added or modified lines and
normalizes/deduplicates findings using the shared `AnalysisFinding` schema.
Its rules cover common Python defects, security risks, and complexity. Run
`python -m scripts.run_evaluation` to calculate benchmark precision, recall,
F1, and localization results.
