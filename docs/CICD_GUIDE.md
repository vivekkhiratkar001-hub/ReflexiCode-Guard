# GitHub / CI-CD Documentation (Member 4)

## 1. What runs, and when
```
PR opened / pushed / reopened / ready_for_review  (base: main, master, develop)
        |
        v
 job `regression`  : unit+integration tests -> offline E2E -> benchmark vs frozen baseline -> upload results/
        |  (must pass)
        v
 job `review`      : checkout (full history) -> python -m reflexicode_guard
                       events.should_review -> github_client.build_context -> analyzers (reflexicode.json)
                       -> pipeline gates -> publisher.publish -> findings.json artifact
```
Drafts are skipped. Fork PRs are forced into dry-run (GitHub gives forks a read-only token).
Manual run: **Actions -> ReflexiCode-Guard -> Run workflow**, enter a PR number (dry-run by default).

## 2. Files
| File | Responsibility |
|---|---|
| `events.py` | Parse `pull_request` payload; branch-strategy gate; linked-issue extraction (`Fixes #3`) |
| `github_client.py` | REST calls, retries (5xx / 429 / rate limit), pagination, `PRContext` builder |
| `diff_utils.py` | Unified-diff parsing: changed lines, commentable lines, hunks |
| `pipeline.py` | Run analyzers, then gates: schema, diff-line, LLM-evidence verification, dedupe, rank, cap |
| `publisher.py` | One review anchored to the PR head SHA; hidden fingerprint marker prevents re-posting on re-runs; 422 fallback to a single PR comment |
| `config.py`, `reflexicode.json` | Which analyzers run (`module:function`), cap, protected branches |
| `e2e_runner.py`, `variants.py`, `metrics.py` | Reproducible evaluation and ablation |
| `contract_check.py` | Verifies a teammate's analyzer obeys the `Finding` contract |

## 3. Why each design choice exists
| Choice | Reason |
|---|---|
| Review anchored to `head_sha` | A stale review on an old commit points at lines that no longer exist |
| Only lines inside the diff are commented | GitHub answers HTTP 422 for any other line, which would fail the whole review |
| Fingerprint marker in each comment | `synchronize` events re-run on every push; without it every push re-posts everything |
| 422 -> single issue comment | A force-push can shift the diff mid-run; degrade instead of failing silently |
| `permissions: contents: read, pull-requests: write` | Least privilege for `GITHUB_TOKEN` |
| `concurrency` + `cancel-in-progress` | A new push cancels the stale review of the previous commit |
| Plain `pull_request` trigger | The alternative `_target` trigger runs untrusted fork code with a write token |
| One broken analyzer cannot kill the run | Exceptions are caught per analyzer and counted in `stats.analyzer_errors` |
| LLM findings must quote the line they cite | Deterministic evidence validates semantic output (core claim of the architecture slide) |

## 4. Configuration
`reflexicode.json` (override per run with env vars):
| Setting | Env var | Meaning |
|---|---|---|
| `analyzers` | `RFG_ANALYZERS` | Comma list. Built-ins: `demo_static`, `mock_llm`. Teammates: `package.module:function` |
| `max_findings` | `RFG_MAX_FINDINGS` | Cap per review (lowest severity dropped first) |
| `llm_provider` | `LLM_PROVIDER` | `mock` (CI default) or `ollama` (needs a self-hosted runner) |
| `protected_bases` | - | Branches that trigger reviews |

## 5. Commands
```bash
make test                      # 64 tests, stdlib only
make regression                # what CI runs
make e2e                       # benchmark + comparison.md in results/latest
python -m reflexicode_guard --offline tests/fixtures      # local demo, no network
python -m reflexicode_guard --pr 7 --repo owner/name --dry-run   # real PR, nothing posted
python -m reflexicode_guard.contract_check m2_ast.analyzer:ast_analyzer
```

## 6. Troubleshooting
| Symptom | Cause | Fix |
|---|---|---|
| `GitHub API 401` | `GITHUB_TOKEN` missing | Map the secret in the job `env:` |
| `GitHub API 403` on posting | Workflow permissions read-only | Repo Settings -> Actions -> General -> Workflow permissions -> Read and write; or keep the `permissions:` block |
| `GitHub API 422` | Comment line not in diff | Pipeline already filters; if it appears the publisher falls back to a PR comment |
| No comments on a fork PR | Read-only token | Expected; see `findings.json` artifact |
| Nothing happens on a draft PR | Gate | Mark ready for review |
| Regression job fails on `baseline` | Behaviour changed | If intentional: `make freeze-baseline` and explain why |
