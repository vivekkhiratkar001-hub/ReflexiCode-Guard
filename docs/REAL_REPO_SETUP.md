# Running it on real GitHub (what could NOT be done in the authoring sandbox)

Everything below was tested only against a local fake GitHub server. Do this once, early (the risks sheet rates
"GitHub API/Actions authentication" as High).

## A. Create the controlled test repo (10 min)
1. New **private** repo `reflexicode-guard-testbed` (never your real project repo).
2. Copy this project in; push to `main`.
3. Settings -> Actions -> General -> Workflow permissions: **Read and write** (or leave read-only and rely on the `permissions:` block).
4. Settings -> Branches: protect `main` (require PR + the `regression` check).

## B. Smoke test (the 8 Oct checklist)
| # | Check | How | Expected |
|---|---|---|---|
| 1 | Trigger | Branch `test/controlled-1`, change `src/auth.py` adding `API_KEY = "sk-live-12345678"`, open a PR to `main` | Workflow starts |
| 2 | Checkout + deps | Open the run log | Both jobs reach "Run regression gate" |
| 3 | Regression job | Log | "REGRESSION PASSED" |
| 4 | Publishing | PR "Files changed" tab | One review with an inline comment on the secret line |
| 5 | Head SHA | Push another commit | New run; old comment **not** duplicated |
| 6 | Draft | Convert to draft, push | `review` job skipped |
| 7 | Manual | Actions -> Run workflow, enter PR number | `findings.json` artifact, nothing posted (dry run) |
| 8 | Failure path | Revoke `permissions:` temporarily | A clear 403 in the log, not a silent pass |

## C. Capture evidence (needed for 24 Oct)
Screenshots: workflow run summary, the inline review comment, the artifacts list. Save to `docs/screenshots/`.

## D. Report back
Anything that differs from "Expected" is a real defect: fix it before the 23 Oct freeze.
