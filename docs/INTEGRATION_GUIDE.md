# Integration guide: how the small M4 tasks combine into the full Member 4 task

Member 4 owns: **GitHub API, PR events, GitHub Actions, comment publishing, E2E integration.**

```
PR event -> events.should_review -> github_client.build_context -> PRContext
                                                                     |
              pipeline.run_pipeline  <-------------------------------+
                 analyzers from reflexicode.json (callables PRContext -> [Finding])
                    M1 rule_analyzer     M2 ast_analyzer     M3 llm_analyzer
                 gates: schema -> diff line -> LLM-evidence check -> dedupe -> rank -> cap
                                  |
                        publisher.publish -> GitHub review (line comments) + findings.json
```

## Task chain (timetable -> final system)
| Step | Timetable task | Becomes |
|---|---|---|
| 1 | PR event / metadata / branch strategy (30 Sep) | `events.py`, `github_client.py` (input side) |
| 2 | Actions skeleton + entry point (1-3 Oct) | workflow + `review_entry.py` (trigger/runner) |
| 3 | Local pipeline + publisher (4-7 Oct) | `pipeline.py`, `publisher.py`, `config.py` (output side) |
| 4 | Test trigger/checkout/deps/publishing (8 Oct) | `tests/test_workflow.py`, `REAL_REPO_SETUP.md` |
| 5 | Reproducible E2E runner (9-11 Oct) | `e2e_runner.py`, benchmark, artifacts |
| 6 | Connect final pipeline to publisher (12-13 Oct) | plug-in loader + `contract_check` |
| 7 | Integration tests, measurement, refinement (14-20 Oct) | metrics, variants, evidence verification |
| 8 | Docs, regression, freeze, demo (21-31 Oct) | `docs/`, `scripts/regression.sh`, baseline |

## Contract for teammates
Each module exposes one function: `def analyzer(ctx: PRContext) -> list[Finding]`.
| Member | Function (example) | `source` | Notes |
|---|---|---|---|
| M1 | `m1_rules.analyzer:rule_analyzer` | `rule` | Evidence = the rule text + the code line |
| M2 | `m2_ast.analyzer:ast_analyzer` | `ast` / `static` / `security` | Line must be a changed line in the diff |
| M3 | `m3_llm.analyzer:llm_analyzer` | `llm` | `evidence` must be copied from the cited line, or the finding is dropped |

Hand-off in two commands:
```bash
python -m reflexicode_guard.contract_check m2_ast.analyzer:ast_analyzer     # PASS required
# then add the spec to "analyzers" in reflexicode.json and run:  make regression
```
`demo_static` and `mock_llm` are stand-ins: remove them from `reflexicode.json` when the real modules land,
and re-freeze the baseline with a written reason (`make freeze-baseline`).
