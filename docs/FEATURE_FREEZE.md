# Feature freeze: CI / GitHub workflow (effective Fri 23 Oct)

Allowed after freeze: fixes that make the demo or the regression gate pass, documentation, screenshots.
Not allowed: new triggers, new gates, new config keys, new analyzers, dependency changes, refactors.

Process for a critical fix: reproduce with a failing test -> fix -> `make regression` green -> one-line note in `KNOWN_LIMITATIONS.md` if behaviour changed.
`benchmarks/baseline_summary.json` is the frozen reference. Changing it needs a written reason in the commit message.
Demo environment freeze: Fri 30 Oct (see `REHEARSAL_AND_BACKUP.md`).
