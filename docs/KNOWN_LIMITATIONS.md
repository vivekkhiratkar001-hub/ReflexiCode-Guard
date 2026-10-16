# Known limitations (triage list for 29 Oct; also the "limitations" slide)

| # | Limitation | Impact | Mitigation |
|---|---|---|---|
| 1 | GitHub omits `patch` for very large or binary files | Those files get no line comments | Logged as unreviewable; not silently passed |
| 2 | Comments only on lines inside the diff; removed lines cannot be commented | Deleted-code issues are invisible | By design (GitHub API) |
| 3 | Fork PRs get a read-only token | No comments, dry-run only | `findings.json` artifact still produced |
| 4 | Fingerprint includes the line number | If a push shifts lines, the same issue may be re-posted once | Acceptable; avoids stale-line comments |
| 5 | Review capped at `max_findings` (25) | Low-severity items dropped on huge PRs | Summary states how many were omitted |
| 6 | Ollama/Qwen cannot run on GitHub-hosted runners | CI uses the mock provider | Self-hosted runner, or run the LLM stage locally for the demo |
| 7 | Stand-in analyzers are regex-based | Weak recall on logic bugs | Replaced by M1/M2/M3 modules |
| 8 | Dockerfile not build-tested in the authoring sandbox | Possible small fix on first build | Build it before 30 Oct |
| 9 | Real-GitHub run not yet performed | Auth/permission surprises possible | `REAL_REPO_SETUP.md` |
