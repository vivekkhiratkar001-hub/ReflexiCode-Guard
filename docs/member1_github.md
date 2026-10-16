# Member 1 — GitHub Integration and PR Review Automation

## 1. Responsibility

Member 1 connects GitHub pull requests to the existing Member 3 LLM review pipeline and (optionally) publishes validated findings as inline review comments.

Member 1 does **not** implement a second LLM, parser, or finding schema. It reuses:

- `ReviewContextBuilder`
- `ReviewPipeline`
- `ReviewFinding`
- `MockProvider` / `OllamaProvider`

## 2. Architecture

```
GitHub pull_request event
        ↓
GitHubClient (HTTP only)
        ↓
PullRequestCollector (metadata + changed files)
        ↓
Diff parser (added / modified / renamed / deleted / missing patch)
        ↓
Member 2 adapters (rules + static analysis, or empty stand-ins)
        ↓
ReviewContextBuilder  →  ReviewPipeline (Member 3)
        ↓
FindingCommentMapper (only valid diff lines)
        ↓
ReviewPublisher (dry-run or GitHub review comments)
        ↓
ReviewSummary
```

## 3. Files

| File | Purpose |
|------|---------|
| `src/github/client.py` | GitHub REST calls, pagination, rate-limit retries |
| `src/github/collector.py` | PR metadata and changed-file collection |
| `src/github/diff.py` | Unified-diff parsing and line maps |
| `src/github/mapper.py` | Finding → GitHub inline comment mapping |
| `src/github/comments.py` | Publish comments and skip duplicates |
| `src/github/orchestrator.py` | End-to-end flow |
| `src/github/cli.py` | `python -m src.github` entry point |
| `src/github/adapters.py` | Temporary Member 2 file/null providers |
| `src/github/config.py` | Environment / CLI / Actions configuration |
| `.github/workflows/pr-review.yml` | Tests + review job |

## 4. Execution flow

1. Identify `owner`, `repo`, and pull request number.
2. Fetch PR title, body, branches, and head SHA.
3. Fetch changed files (paginated). Skip binary files and files without a patch.
4. Parse unified diffs. Build `changed_code` and `changed_lines` for Member 3 from **added** lines (or deleted lines for removed files).
5. Load repository rules and static-analysis findings if JSON files are provided. Otherwise pass empty lists.
6. Call `ReviewPipeline.review(context)` and keep Member 3's validation unchanged.
7. Map each `ReviewFinding` to a GitHub `RIGHT` or `LEFT` diff line. Unmapped findings go into the summary only.
8. Skip comments whose fingerprint already exists on the pull request.
9. In dry-run mode, print the comments and do **not** call the GitHub write API.
10. In publish mode, create one GitHub pull-request review with the new inline comments.

## 5. Environment variables

| Name | Required | Meaning |
|------|----------|---------|
| `GITHUB_TOKEN` | For private repos and for `--publish` | GitHub token. In Actions this is `secrets.GITHUB_TOKEN`. |
| `GITHUB_REPOSITORY` | Optional | `owner/repo` (set automatically in Actions) |
| `GITHUB_PR_NUMBER` | Optional | Pull request number for local runs |
| `GITHUB_EVENT_PATH` | Optional | Actions event payload path |
| `GITHUB_API_URL` | Optional | Defaults to `https://api.github.com` |
| `REFLEXICODE_USE_MOCK_LLM` | Optional | Use `MockProvider` (needed on GitHub-hosted runners) |
| `REFLEXICODE_MOCK_LLM_RESPONSE` | Optional | JSON returned by the mock provider |
| `REFLEXICODE_RULES_FILE` | Optional | JSON list of repository rules |
| `REFLEXICODE_ANALYSIS_FILE` | Optional | JSON list of static-analysis findings |
| `GITHUB_STEP_SUMMARY` | Actions | Appended markdown job summary |

Never put tokens in source files. `.env` is gitignored.

## 6. GitHub token permissions

For **dry-run / read**:

- `contents: read`
- Ability to read pull requests (public repos work with a fine-grained or classic token that can read metadata)

For **publishing comments in a deliberate local run**:

- `pull-requests: write`
- Use a narrowly scoped token and run `--publish` only after checking dry-run output.

The included GitHub Actions workflow is read-only (`contents: read`, `pull-requests: read`) and always dry-runs. It checks out trusted base-branch code for the review step, scopes `GITHUB_TOKEN` to that step, and disables persisted checkout credentials. PR test code has no token. It does not publish comments.

Administrator settings that are **not** changed by this code:

1. Ensure Actions are enabled for the repository.
2. Do not grant this workflow write permissions or configure it to publish.
3. Do **not** add a `pull_request_target` workflow.

## 7. How to run tests (Windows PowerShell)

From the repository root, with the project virtual environment:

```powershell
cd C:\Users\DELL\ReflexiCode-Guard
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest tests\test_github.py tests\test_github_orchestrator.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_llm.py tests\test_end_to_end.py tests\test_context_integration.py tests\test_evaluation.py -q
```

Tests mock GitHub HTTP responses. They do not need a real token and they never post comments.

## 8. Local dry run (Windows PowerShell)

Dry-run still **reads** the pull request from GitHub, then runs the pipeline, then prints comments without writing them.

```powershell
cd C:\Users\DELL\ReflexiCode-Guard
$env:GITHUB_TOKEN = "ghp_your_token_here"
.\.venv\Scripts\python.exe -m src.github --owner vivekkhiratkar001-hub --repo ReflexiCode-Guard --pr-number 1 --dry-run --use-mock-llm
```

With Member 2 stand-in JSON files:

```powershell
.\.venv\Scripts\python.exe -m src.github --owner vivekkhiratkar001-hub --repo ReflexiCode-Guard --pr-number 1 --dry-run --use-mock-llm --rules-file tests\fixtures\sample_repository_rules.json --analysis-file tests\fixtures\sample_analysis_findings.json
```

To use the real Ollama Member 3 provider locally (Ollama must be running):

```powershell
Remove-Item Env:REFLEXICODE_USE_MOCK_LLM -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m src.github --owner vivekkhiratkar001-hub --repo ReflexiCode-Guard --pr-number 1 --dry-run
```

## 9. Enable actual comment publication safely

1. Confirm dry-run output looks correct.
2. Use a token with `pull-requests: write`.
3. Run with `--publish` (this is the only mode that calls the write API):

```powershell
$env:GITHUB_TOKEN = "ghp_your_token_here"
.\.venv\Scripts\python.exe -m src.github --owner vivekkhiratkar001-hub --repo ReflexiCode-Guard --pr-number 1 --publish --use-mock-llm
```

`--use-mock-llm` still avoids calling Ollama. Remove it only when Ollama is available.

GitHub Actions publishing is not supported by this workflow. Keep automated runs read-only; use the local command above for deliberate publishing after checking the proposed comments.

## 10. Dependencies on other members

- **Member 3 (available):** `ReviewContextBuilder`, `ReviewPipeline`, `ReviewFinding`, providers. Required for a real review.
- **Member 2 (missing in this repository):** repository rules and static analysis. Member 1 passes `[]` or optional JSON files. It does not invent a second analyzer.
- **Member 4 (missing):** no dashboard/orchestration module is present. Member 1 writes a `ReviewSummary` that another module could consume later.

## 11. Duplicate prevention

Each comment includes an HTML fingerprint:

`<!-- reflexicode-guard:fp=... -->`

On rerun, existing PR review comments are loaded and matching fingerprints are skipped. In publish mode, failure to load existing comments aborts publication rather than risking duplicates.

## 12. Troubleshooting

| Symptom | What to check |
|---------|----------------|
| `Configuration error: Missing pull-request identity` | Pass `--owner`, `--repo`, `--pr-number` or set `GITHUB_REPOSITORY` and `GITHUB_PR_NUMBER`. |
| HTTP 401 | Token missing or invalid. |
| HTTP 404 | Wrong owner/repo/PR number, or the token cannot see a private repo. |
| Rate limit error | Wait; authenticated requests have a higher limit. |
| `No reviewable text diffs` | The PR only has binary files or GitHub omitted patches. |
| Member 3 rejected the model response | The model returned JSON that failed schema/line/evidence checks. Use `--use-mock-llm` to debug mapping without Ollama. |
| Comments not appearing | You are in dry-run, the finding line is not in the diff, or the fingerprint already exists. |

## 13. Known limitations

- GitHub-hosted Actions runners do not include Ollama, so the workflow uses `MockProvider` unless you add a self-hosted runner with Ollama.
- Member 2 production analyzers are not wired because those modules are not in the tree.
- Inline comments are only posted on lines that appear in the GitHub diff. Findings that Member 3 accepts without a mappable diff line are listed as unmapped.
- `pull_request_target` is intentionally not used.
