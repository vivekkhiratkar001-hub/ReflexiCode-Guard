# Branch Strategy (M4 – 30 Sep deliverable)

| Branch | Purpose | Rules |
|---|---|---|
| `main` | Stable, demo-ready | Protected; PR + passing CI required; reviewed by ReflexiCode-Guard |
| `develop` | Integration branch (Week 2+) | Members merge here first; CI runs on every PR |
| `feature/m1-rules`, `feature/m2-ast`, `feature/m3-llm`, `feature/m4-github` | One per member | Short-lived, rebase on `develop` daily |
| `fix/<topic>` | Bug fixes | Only type allowed after feature freeze (18 Oct) |
| `test/controlled-<n>` | Controlled test PRs for evaluation | Never merged; used to verify line localization |

**Review gate** (`events.should_review`): reviews run only for actions `opened | synchronize | reopened | ready_for_review`, non-draft PRs, with base `main/master/develop`. Fork PRs run in dry-run (read-only token).

**Commits:** `<member>: <module> – <what>` e.g. `m4: publisher – dedupe by fingerprint`.
**Freeze dates:** features freeze 18 Oct, demo environment freeze 31 Oct (see timetable).
