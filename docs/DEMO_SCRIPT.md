# Demo script (about 6 minutes)

**Setup before the audience arrives:** testbed repo open, PR #1 *closed*, a second branch ready with the vulnerable change,
terminal in the project folder, `make regression` already green, a screenshot of each step as backup.

| Min | Say | Do |
|---|---|---|
| 0:00 | "ReflexiCode-Guard reviews pull requests. Rule: deterministic code evidence validates the LLM before anything is posted." | Show architecture slide |
| 0:45 | "Here is a PR that adds a hard-coded key, an `eval`, and a TODO." | Open PR diff |
| 1:30 | "The pull_request event triggers the workflow." | Actions tab -> running workflow |
| 2:15 | "First job is the regression gate: tests plus the benchmark against a frozen baseline." | Show `REGRESSION PASSED` |
| 3:00 | "Second job reviews the PR." | Open review step log (stats line) |
| 3:45 | "Comments appear on exact lines, with evidence, anchored to the latest commit." | Files changed -> 3 inline comments |
| 4:30 | "Pushing again does not duplicate them." | Push a trivial commit; show no new comments |
| 5:00 | "Ablation: unchecked LLM output would produce 15 comments GitHub rejects; the gates remove them." | Open `comparison.md` (state that the LLM here is simulated) |
| 5:45 | "Limits and next steps." | Show limitations slide |

## Backup ladder (use the first rung that works)
1. **Live PR on GitHub** (above).
2. **Manual dispatch, dry-run** with a PR number: `findings.json` shows the same findings.
3. **Local replay:** `make demo` prints the exact review payload with no network.
4. **Screenshots / screen recording** captured on 24 Oct.
5. **Docker:** `docker run --rm reflexicode-guard` runs the regression gate offline.

## Questions to be ready for
- *Why not hosted Flask?* GitHub Actions already delivers events and hosts execution; no server to run or secure.
- *Why local LLM?* Privacy and cost; the validation gate controls hallucination.
- *What does the gate not catch?* Plausible wrong claims on a real line (see EVALUATION_NOTES, limit 3).
- *Are these numbers the project result?* No: synthetic benchmark; real dataset numbers come from the final evaluation.
