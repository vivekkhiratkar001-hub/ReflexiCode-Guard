# ReflexiCode-Guard

AI-based pull request review system.

This repository currently contains:

- **Member 1** — GitHub integration and PR review automation (this document, plus [docs/member1_github.md](docs/member1_github.md))
- **Member 3** — LLM / semantic review pipeline ([docs/member3_llm.md](docs/member3_llm.md))

Member 2 (repository rules and static analysis) and Member 4 are not in this tree yet. Member 1 uses empty or file-based adapters until those modules exist.

## Member 1 — GitHub Integration

Member 1 collects pull-request metadata and diffs from GitHub, builds a `ReviewContext` for Member 3, maps validated findings onto GitHub diff lines, and can publish inline review comments.

Default mode is **dry-run**: the pipeline runs and the comments are printed, but GitHub is not updated.

### Main files

- `src/github/` — GitHub client, collector, diff parser, mapper, publisher, CLI
- `.github/workflows/pr-review.yml` — run tests and the review entry point on pull requests
- `docs/member1_github.md` — full Member 1 guide

### Environment

Python 3.11+ is recommended. Dependencies are listed in `requirements.txt` (`pydantic`, `requests`, `pytest`). No extra GitHub SDK is required.

Required for reading a private PR or publishing comments:

- `GITHUB_TOKEN`

See [docs/member1_github.md](docs/member1_github.md) for the full variable list and token permissions.

### Tests (Windows PowerShell)

```powershell
cd C:\Users\DELL\ReflexiCode-Guard
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

### Local dry run (Windows PowerShell)

```powershell
cd C:\Users\DELL\ReflexiCode-Guard
$env:GITHUB_TOKEN = "ghp_your_token_here"
.\.venv\Scripts\python.exe -m src.github --owner vivekkhiratkar001-hub --repo ReflexiCode-Guard --pr-number 1 --dry-run --use-mock-llm
```

`--use-mock-llm` avoids calling Ollama. Remove it to use Member 3's `OllamaProvider` when Ollama is running.

### Publish comments (only after a successful dry run)

```powershell
$env:GITHUB_TOKEN = "ghp_your_token_here"
.\.venv\Scripts\python.exe -m src.github --owner vivekkhiratkar001-hub --repo ReflexiCode-Guard --pr-number 1 --publish
```

GitHub Actions always runs read-only in dry-run mode. Publishing is intentionally not enabled in the workflow; run `--publish` locally only after reviewing the dry-run output and providing a narrowly scoped token.

## Member 3 — LLM Review Module

The LLM module is documented in [docs/member3_llm.md](docs/member3_llm.md).

It provides:

- ReviewContext construction from PR and code inputs
- Ollama + Qwen2.5-Coder integration
- prompt construction for structured review output
- parser and validation logic for review findings
- end-to-end pipeline orchestration
- deterministic evaluation helpers for TP / FP / FN / Precision / Recall / F1

For usage and integration details, see [docs/member3_llm.md](docs/member3_llm.md).
