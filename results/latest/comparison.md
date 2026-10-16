# Comparison (generated 2026-10-16T05:18:20Z)

Benchmark: **13 controlled PRs**, 13 ground-truth defects, 3 clean PRs. Repeats per variant: 5. Synthetic benchmark + simulated hallucinations: see docs/EVALUATION_NOTES.md before quoting numbers.

## Line-level (exact line must match)

FPR = flagged innocent lines inside the diff / innocent lines inside the diff. Would-422 = flags outside the diff (GitHub rejects them).

| Variant | TP | FP | FN | Precision | Recall | F1 | FPR | Would-422 | Reproducible |
|---|---|---|---|---|---|---|---|---|---|
| A_hunk_baseline | 0 | 8 | 13 | 0.00 | 0.00 | 0.00 | 0.140 | 0 | yes |
| B_static_only | 7 | 1 | 6 | 0.88 | 0.54 | 0.67 | 0.018 | 0 | yes |
| C_static+mockLLM | 11 | 1 | 2 | 0.92 | 0.85 | 0.88 | 0.018 | 0 | yes |
| D_noisyLLM_no_gate | 11 | 45 | 2 | 0.20 | 0.85 | 0.32 | 0.526 | 15 | yes |
| E1_noisyLLM_schema+diff | 11 | 19 | 2 | 0.37 | 0.85 | 0.51 | 0.333 | 0 | yes |
| E2_noisyLLM_full_gate | 11 | 16 | 2 | 0.41 | 0.85 | 0.55 | 0.281 | 0 | yes |

## Hunk-level (credit for being in the right hunk)

| Variant | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| A_hunk_baseline | 6 | 2 | 6 | 0.75 | 0.50 | 0.60 |
| B_static_only | 7 | 1 | 6 | 0.88 | 0.54 | 0.67 |
| C_static+mockLLM | 11 | 1 | 2 | 0.92 | 0.85 | 0.88 |
| D_noisyLLM_no_gate | 32 | 24 | 0 | 0.57 | 1.00 | 0.73 |
| E1_noisyLLM_schema+diff | 24 | 6 | 0 | 0.80 | 1.00 | 0.89 |
| E2_noisyLLM_full_gate | 23 | 4 | 0 | 0.85 | 1.00 | 0.92 |

## What each gate removed (first run)

| Variant | invalid schema | off-diff line | evidence mismatch | duplicates |
|---|---|---|---|---|
| A_hunk_baseline | 0 | 0 | 0 | 0 |
| B_static_only | 0 | 0 | 0 | 0 |
| C_static+mockLLM | 0 | 0 | 0 | 0 |
| D_noisyLLM_no_gate | 0 | 0 | 0 | 0 |
| E1_noisyLLM_schema+diff | 15 | 15 | 0 | 0 |
| E2_noisyLLM_full_gate | 15 | 15 | 15 | 0 |

## Processing time

| Variant | whole benchmark median (ms) | per PR median (ms) | per PR p95 (ms) |
|---|---|---|---|
| A_hunk_baseline | 0.944 | 0.033 | 0.068 |
| B_static_only | 0.684 | 0.024 | 0.046 |
| C_static+mockLLM | 0.833 | 0.034 | 0.067 |
| D_noisyLLM_no_gate | 1.569 | 0.059 | 0.126 |
| E1_noisyLLM_schema+diff | 1.338 | 0.056 | 0.117 |
| E2_noisyLLM_full_gate | 1.279 | 0.056 | 0.112 |

Timing here covers only the M4 pipeline on mock analyzers. Real Ollama/Qwen latency must be measured separately (M3).
