"""STAND-INS ONLY. Replace with the real modules from M1/M2/M3.

mock_llm_analyzer mimics M3's contract (evidence-required findings, source="llm") deterministically,
so the CI workflow and the E2E runner work before Ollama/Qwen is ready, and on runners without a GPU.
The risks sheet calls for exactly this: "Keep Mock LLM/provider; cache test outputs".
"""
from __future__ import annotations

import re

from .diff_utils import parse_patch
from .models import Finding, PRContext

_PATTERNS = [
    (re.compile(r"\bexcept\s*:\s*$|\bexcept\s+Exception\s*:\s*pass\b"), "medium", "logic",
     "Bare/broad except hides errors; catch specific exceptions."),
    (re.compile(r"\bshell\s*=\s*True\b"), "high", "security",
     "subprocess with shell=True enables command injection."),
    (re.compile(r"""execute\(\s*f?["'].*(%s|\{).*["']"""), "high", "security",
     "SQL built via string formatting; use parameterised queries."),
]


def mock_llm_analyzer(ctx: PRContext) -> list[Finding]:
    out = []
    for f in ctx.files:
        for ln, text in parse_patch(f.patch)["added"].items():
            for rx, sev, cat, msg in _PATTERNS:
                if rx.search(text):
                    out.append(Finding(f.path, ln, sev, cat, msg, text.strip(), "llm", "MOCK-LLM", 0.5))
    return out


def noisy_llm_analyzer(ctx: PRContext) -> list[Finding]:
    """SIMULATION of LLM hallucination, used ONLY by the ablation study (deterministic, no randomness).
    Per changed file it emits four kinds of bad finding:
      1. line outside the diff            -> removed by the diff/line-localization gate
      2. empty evidence                   -> removed by the evidence-required schema gate
      3. evidence not on the cited line   -> removed by deterministic evidence verification
      4. wrong claim, real unchanged line+text -> NOT removable by any deterministic gate (residual false positive)
    Real numbers must come from real Qwen output (M3); this only shows what each gate is able to catch."""
    out = []
    for f in ctx.files:
        p = parse_patch(f.patch)
        if not p["added"]:
            continue
        last = max(p["added"])
        # Anchor on an UNCHANGED (context) line so a simulated hallucination can never hit a real defect
        # by luck; it is also a realistic failure ("commenting on code the PR did not touch").
        # Files with no context line (pure additions) get only the off-diff hallucination.
        out.append(Finding(f.path, last + 500, "high", "logic", "Hallucinated: line not in diff", "x = 1", "llm", "NOISE-1", 0.3))
        cl = sorted(l for l, t in p["context"].items() if t.strip())     # non-blank unchanged lines
        if cl:
            # spread over different unchanged lines when the hunk allows it, so each gate's effect is visible
            a2, a3, a4 = cl[0], cl[-1], cl[len(cl) // 2] if len(cl) >= 3 else cl[-1]
            out.append(Finding(f.path, a2, "medium", "logic", "Hallucinated: no evidence given", "", "llm", "NOISE-2", 0.3))
            out.append(Finding(f.path, a3, "medium", "logic", "Hallucinated: evidence not on line", "does_not_exist(42)", "llm", "NOISE-3", 0.3))
            out.append(Finding(f.path, a4, "low", "logic", "Plausible but wrong claim", p["context"][a4].strip(), "llm", "NOISE-4", 0.3))
    return out


def hunk_baseline_analyzer(ctx: PRContext) -> list[Finding]:
    """BASELINE for line-vs-hunk localization: scans the whole hunk (context lines too, like a tool that
    ignores which lines changed) and reports at the hunk's first line instead of the offending line."""
    import re as _re
    from .diff_utils import hunk_ranges
    from .pipeline import _DEMO_RULES
    out = []
    for f in ctx.files:
        p = parse_patch(f.patch)
        lines = {**p["context"], **p["added"]}
        for start, end in hunk_ranges(f.patch):
            for ln in range(start, end + 1):
                text = lines.get(ln, "")
                for rid, sev, cat, rx, msg in _DEMO_RULES:
                    if rx.search(text):
                        out.append(Finding(f.path, start, sev, cat, msg, text.strip(), "static", rid))
                        break
    return out
