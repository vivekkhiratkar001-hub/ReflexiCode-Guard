"""Integration contract checker (12-13 Oct): run a teammate's analyzer against the benchmark and report
whether its output is safe to plug into the pipeline.

    python -m reflexicode_guard.contract_check m2_ast.analyzer:ast_analyzer
    python -m reflexicode_guard.contract_check demo_static --bench benchmarks/controlled_prs

Checks per finding: is a Finding; schema valid (evidence non-empty, severity/source allowed, line >= 1);
file is in the PR; line is a commentable line (else GitHub returns 422); for source="llm" evidence is on the line.
Also: analyzer must not raise, must return a list, and must be deterministic (same output twice).
Exit code 1 if anything fails.
"""
from __future__ import annotations

import argparse
import sys

from .config import resolve_analyzer
from .diff_utils import commentable_lines, parse_patch
from .e2e_runner import load_benchmark
from .models import Finding


def check_analyzer(fn, cases) -> dict:
    problems, total = [], 0
    for c in cases:
        try:
            out = fn(c.ctx)
            again = fn(c.ctx)
        except Exception as e:                                   # noqa: BLE001
            problems.append(f"{c.name}: raised {type(e).__name__}: {e}")
            continue
        if not isinstance(out, list):
            problems.append(f"{c.name}: returned {type(out).__name__}, expected list[Finding]")
            continue
        if [getattr(f, "fingerprint", None) for f in out] != [getattr(f, "fingerprint", None) for f in again]:
            problems.append(f"{c.name}: non-deterministic output (two runs differ)")
        ok_lines = {f.path: commentable_lines(f.patch) for f in c.ctx.files}
        text = {f.path: {**parse_patch(f.patch)["context"], **parse_patch(f.patch)["added"]} for f in c.ctx.files}
        for f in out:
            total += 1
            if not isinstance(f, Finding):
                problems.append(f"{c.name}: item is {type(f).__name__}, not Finding"); continue
            for e in f.validate():
                problems.append(f"{c.name} {f.file}:{f.line}: {e}")
            if f.file not in ok_lines:
                problems.append(f"{c.name}: file '{f.file}' is not part of the PR")
            elif f.line not in ok_lines[f.file]:
                problems.append(f"{c.name} {f.file}:{f.line}: line is outside the diff (GitHub would reject with 422)")
            elif f.source == "llm" and " ".join(f.evidence.split()) not in " ".join(text[f.file].get(f.line, "").split()):
                problems.append(f"{c.name} {f.file}:{f.line}: LLM evidence does not appear on the cited line")
    return {"findings": total, "problems": problems, "ok": not problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check an analyzer against the Finding contract")
    ap.add_argument("spec", help="'module:function' or built-in name (demo_static, mock_llm)")
    ap.add_argument("--bench", default="benchmarks/controlled_prs")
    a = ap.parse_args(argv)
    r = check_analyzer(resolve_analyzer(a.spec), load_benchmark(a.bench))
    for p in r["problems"][:30]:
        print("  -", p)
    print(f"{'PASS' if r['ok'] else 'FAIL'}: {a.spec} -> {r['findings']} findings, {len(r['problems'])} problem(s)")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
