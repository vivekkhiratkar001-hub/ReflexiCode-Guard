# ReflexiCode-Guard

This project contains the Member 3 LLM / semantic review pipeline.

## Member 3 — LLM Review Module

The LLM module is documented in [docs/member3_llm.md](docs/member3_llm.md).

It provides:

- ReviewContext construction from PR and code inputs
- Ollama + Qwen2.5-Coder integration
- prompt construction for structured review output
- parser and validation logic for review findings
- end-to-end pipeline orchestration
- deterministic evaluation helpers for TP / FP / FN / Precision / Recall / F1

For usage and integration details, see [docs/member3_llm.md](docs/member3_llm.md).
