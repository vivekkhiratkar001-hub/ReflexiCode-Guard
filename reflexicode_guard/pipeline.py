"""Day 5-6 (4-5 Oct): local pipeline that every member's module plugs into.

Each analyzer is any callable `(PRContext) -> list[Finding]`. M1/M2/M3 register theirs
in `default_analyzers()`; until then the built-in DemoStaticAnalyzer stands in so the
local end-to-end simulation (Week-1 milestone) already works.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Iterable

from .diff_utils import commentable_lines, parse_patch
from .models import Finding, PRContext

log = logging.getLogger("reflexicode_guard")
Analyzer = Callable[[PRContext], list[Finding]]
SEV_ORDER = {s: i for i, s in enumerate(("info", "low", "medium", "high", "critical"))}

_DEMO_RULES = [
    ("SEC001", "critical", "security", re.compile(r"""(?i)(password|secret|api[_-]?key|token)\s*=\s*['"][^'"]{4,}['"]"""),
     "Hard-coded credential. Load it from an environment variable or secret store."),
    ("SEC002", "high", "security", re.compile(r"\b(eval|exec)\s*\("),
     "Use of eval/exec on dynamic input is dangerous."),
    ("STY001", "low", "style", re.compile(r"(?i)\b(TODO|FIXME)\b"),
     "Unresolved TODO/FIXME in changed code."),
]


def demo_static_analyzer(ctx: PRContext) -> list[Finding]:
    """Placeholder for M2's real AST/static module: regex over ADDED lines only."""
    out = []
    for f in ctx.files:
        for ln, text in parse_patch(f.patch)["added"].items():
            for rid, sev, cat, rx, msg in _DEMO_RULES:
                if rx.search(text):
                    out.append(Finding(f.path, ln, sev, cat, msg, text.strip(), "static", rid))
    return out


def default_analyzers() -> list[Analyzer]:
    """Analyzers come from config (reflexicode.json / RFG_ANALYZERS). M1/M2/M3 register there."""
    from .config import load_analyzers, load_config
    return load_analyzers(load_config())


@dataclass
class PipelineOptions:
    """Switches used by the ablation study. Production keeps all True."""
    validate_schema: bool = True     # evidence-required schema gate
    filter_to_diff: bool = True      # line-level localization gate (avoids HTTP 422)
    dedupe: bool = True
    max_findings: int = 25
    verify_evidence: bool = True     # LLM findings must quote text that really is on the target line


def _norm(t: str) -> str:
    return " ".join(t.split())


def _evidence_on_line(fd: Finding, line_text: dict) -> bool:
    """Deterministic check: the quoted evidence must actually appear on the line the LLM points at.
    Only applied to source="llm"; rule/AST/static findings are produced from the code itself."""
    text = line_text.get(fd.file, {}).get(fd.line)
    return text is not None and _norm(fd.evidence) in _norm(text)


def run_pipeline(ctx: PRContext, analyzers: Iterable[Analyzer] | None = None,
                 max_findings: int | None = None,
                 options: PipelineOptions | None = None) -> tuple[list[Finding], dict]:
    """Run analyzers -> validate schema -> keep only commentable lines -> dedupe -> rank."""
    opt = options or PipelineOptions()
    if max_findings is not None:
        opt = PipelineOptions(opt.validate_schema, opt.filter_to_diff, opt.dedupe, max_findings, opt.verify_evidence)
    stats = {"raw": 0, "invalid_schema": 0, "off_diff": 0, "evidence_mismatch": 0, "duplicates": 0, "analyzer_errors": 0, "unreviewable_files": 0}
    raw: list[Finding] = []
    for an in (analyzers if analyzers is not None else default_analyzers()):
        try:
            raw += an(ctx)
        except Exception:                      # one broken module must not kill the review
            stats["analyzer_errors"] += 1
            log.exception("analyzer %s failed", getattr(an, "__name__", an))
    stats["raw"] = len(raw)
    # GitHub omits `patch` for very large / binary files: say so instead of silently passing them.
    unreviewable = [f.path for f in ctx.files if not f.patch and f.status in ("added", "modified", "changed")]
    stats["unreviewable_files"] = len(unreviewable)
    if unreviewable:
        log.warning("no diff available for %d file(s), not reviewed: %s", len(unreviewable), ", ".join(unreviewable[:5]))

    targets = {f.path: commentable_lines(f.patch) for f in ctx.files}
    line_text = {}
    for f in ctx.files:
        p = parse_patch(f.patch)
        line_text[f.path] = {**p["context"], **p["added"]}
    seen, kept = set(), []
    for fd in raw:
        if opt.validate_schema and fd.validate():
            stats["invalid_schema"] += 1
        elif opt.filter_to_diff and fd.line not in targets.get(fd.file, set()):
            stats["off_diff"] += 1
        elif opt.verify_evidence and fd.source == "llm" and not _evidence_on_line(fd, line_text):
            stats["evidence_mismatch"] += 1
        elif opt.dedupe and fd.fingerprint in seen:
            stats["duplicates"] += 1
        else:
            seen.add(fd.fingerprint)
            kept.append(fd)
    kept.sort(key=lambda x: (-SEV_ORDER.get(x.severity, 0), x.file, x.line))
    stats["published_candidates"] = min(len(kept), opt.max_findings)
    stats["truncated"] = max(0, len(kept) - opt.max_findings)
    return kept[:opt.max_findings], stats
