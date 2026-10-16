"""Ablation variants. Each isolates the contribution of one pipeline component."""
from __future__ import annotations

from dataclasses import dataclass, field

from .pipeline import PipelineOptions, demo_static_analyzer
from .stubs import hunk_baseline_analyzer, mock_llm_analyzer, noisy_llm_analyzer


@dataclass
class Variant:
    name: str
    description: str
    analyzers: list = field(default_factory=list)
    options: PipelineOptions = field(default_factory=PipelineOptions)


NO_GATE = PipelineOptions(validate_schema=False, filter_to_diff=False, dedupe=True, verify_evidence=False)
SCHEMA_DIFF_ONLY = PipelineOptions(verify_evidence=False)
LLM = [demo_static_analyzer, mock_llm_analyzer, noisy_llm_analyzer]


def all_variants() -> list[Variant]:
    return [
        Variant("A_hunk_baseline", "Naive baseline: scans whole hunk, reports hunk start (hunk-level localization)",
                [hunk_baseline_analyzer]),
        Variant("B_static_only", "Line-level deterministic static rules only", [demo_static_analyzer]),
        Variant("C_static+mockLLM", "Static + clean mock LLM (no hallucination), full gate",
                [demo_static_analyzer, mock_llm_analyzer]),
        Variant("D_noisyLLM_no_gate", "Static + mock LLM + SIMULATED hallucinations, validation gates OFF", LLM, NO_GATE),
        Variant("E1_noisyLLM_schema+diff", "Same, with evidence-schema + diff-line gates only", LLM, SCHEMA_DIFF_ONLY),
        Variant("E2_noisyLLM_full_gate", "Same, with all gates incl. deterministic evidence verification (proposed)", LLM),
    ]
