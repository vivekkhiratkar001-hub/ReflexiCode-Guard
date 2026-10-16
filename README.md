# ReflexiCode-Guard: Member 4 (GitHub / CI-CD / Integration)

Reviews pull requests inside GitHub Actions and posts line-level comments. Deterministic evidence constrains and
validates semantic (LLM) findings before anything is published.

```bash
make test         # 64 tests, standard library only
make regression   # what CI runs: tests + offline E2E + benchmark vs frozen baseline
make e2e          # 6-variant ablation over 13 controlled PRs -> results/latest/comparison.md
make demo         # offline review of a sample PR (no network)
python -m reflexicode_guard --pr 7 --repo owner/name --dry-run      # a real PR, nothing posted
python -m reflexicode_guard.contract_check <module:function>        # check a teammate's analyzer
```

Read first: `docs/M4_DAY_LOG.md` (every timetable task with honest status), `docs/CICD_GUIDE.md`,
`docs/REAL_REPO_SETUP.md` (the one thing not yet done: a real GitHub run), `docs/EVALUATION_NOTES.md` (limits of the numbers).

`demo_static` and `mock_llm` are stand-ins for the other members' modules (see `docs/INTEGRATION_GUIDE.md`).
