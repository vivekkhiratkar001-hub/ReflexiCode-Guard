# Team Branch Integration Notes

The integration branch is based on `origin/main` and preserves the existing
shared `PRContext`, `AnalysisFinding`, and `ReviewResult` contracts.

## Included work

- Member 2's modular deterministic analyzer and repository context reader are
  the shared analysis foundation.
- Member 3's semantic context, Ollama/mock providers, response parser,
  validation, and evaluation pipeline are retained.
- Member 1's GitHub client, PR collector, inline-comment mapper/publisher,
  dry-run CLI, and read-only Actions workflow are integrated. GitHub-collected
  diffs are adapted to the shared contracts, then deterministic findings and
  validated semantic findings flow through `ReviewResult` before publishing.
- Member 4's exact changed-line evidence gate is represented by optional
  `changed_line_content` in `ReviewContext`; the GitHub adapter supplies this
  map and tests that evidence cannot be borrowed from another changed line.
  Its draft-PR skip is also carried into the Member 1 Actions review-job gate.

## Intentionally not copied

Member 4's branch contains a separate `reflexicode_guard` application with its
own PR/finding contracts, pipeline, GitHub client, publisher, workflow, and
benchmark runner. Copying that application alongside Member 1 would create two
competing implementations and bypass the shared models. Its line-specific
evidence behavior was carried into the canonical pipeline; its duplicate
publisher and standalone runner were not.
The branch-specific protected-base filter and analyzer-ablation runner were
also left out: the existing workflow retains its configured PR actions and
fork restriction, skips drafts, and the integrated unit/evaluation suite
exercises the canonical shared pipeline without introducing a second runner.

`origin/master` has no merge base with `origin/main`. It contains the earlier
prototype and an older, monolithic version of Member 2's analyzer. The newer
`member2-analysis` branch supersedes those relevant analyzer changes with the
modular implementation, so no unique `master` changes were added.

Pull request #1, "Member2 analysis week 1", was inspected and remains open on
`member2-analysis`; this integration work does not merge or modify it.
