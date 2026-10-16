# Evaluation notes: read before quoting any number

## What was measured
`make e2e` runs 6 variants x 5 repeats over 13 controlled PRs (13 ground-truth defects, 3 clean PRs).
Unit of comparison: unique `(file, line)` location. Output: `results/latest/comparison.md`.

| Variant | Question it answers |
|---|---|
| A hunk baseline | How much does exact line localization matter vs hunk-level? |
| B static only | What do deterministic rules alone find? |
| C static + mock LLM | What does a semantic layer add? |
| D noisy LLM, no gates | What happens if LLM output is published unchecked? |
| E1 + schema/diff gates | What do evidence-required schema and diff-line gates remove? |
| E2 + evidence verification | What does deterministic verification of LLM evidence add? |

## Results of the last run (5 repeats, all reproducible, identical in a fresh interpreter)
| Variant | Precision | Recall | F1 | FP | Would be rejected by GitHub (422) |
|---|---|---|---|---|---|
| A | 0.00 | 0.00 | 0.00 | 8 | 0 |
| B | 0.88 | 0.54 | 0.67 | 1 | 0 |
| C | 0.92 | 0.85 | 0.88 | 1 | 0 |
| D | 0.20 | 0.85 | 0.32 | 45 | **15** |
| E1 | 0.37 | 0.85 | 0.51 | 19 | 0 |
| E2 | 0.41 | 0.85 | 0.55 | 16 | 0 |

Hunk-level credit lifts A from F1 0.00 to 0.60: a tool that only points at the right hunk is not useful for line comments.

## Limits (these are important)
1. **The benchmark is synthetic**: 13 small PRs written to exercise the harness. It is not the project's real ground truth (valid PRs + human review comments). Do not present these numbers as project results.
2. **The "LLM" is a pattern-matching stand-in.** Variants D/E use *simulated* hallucinations (`noisy_llm_analyzer`) of four kinds. They show what each gate is capable of catching; they are not measurements of Qwen2.5-Coder.
3. **Gates cannot remove everything.** The 16 false positives left in E2 are 15 "plausible but wrong claims" that quote a real line, plus 1 regex false positive (a comment mentioning `eval(`). Only a stronger semantic check can catch those; this is an honest residual.
4. **Both semantic defects (off-by-one, division by zero) are missed by every variant.** That gap is what the real LLM has to close.
5. **Timing covers only M4's pipeline** (about 0.05 ms per PR). Real model latency is M3's measurement.
6. Metrics are micro-averaged over locations; FPR is over lines inside the diff.

## To turn this into real results
Drop the real dataset into a folder with the same layout (`<case>/pr_event.json`, `sample.diff`, `expected.json`), then
`python -m reflexicode_guard.e2e_runner --bench <folder>`, register M1/M2/M3 in `reflexicode.json`, and re-freeze the baseline.
