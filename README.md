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

Publishing requires a token with pull-request write permission and an explicit
`--publish` flag:

```powershell
$env:GITHUB_TOKEN = "..."
python -m src.github --owner OWNER --repo REPOSITORY --pr-number NUMBER --publish
```

GitHub Actions runs tests and the review in dry-run mode; it does not publish
comments automatically. See [docs/member1_github.md](docs/member1_github.md) for
configuration and [docs/member3_llm.md](docs/member3_llm.md) for semantic-review
details.

## Analysis and evaluation

The deterministic analyzer localizes findings to added or modified lines and
normalizes/deduplicates findings using the shared `AnalysisFinding` schema.
Its rules cover common Python defects, security risks, and complexity. Run
`python -m scripts.run_evaluation` to calculate benchmark precision, recall,
F1, and localization results.
