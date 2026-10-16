"""Reproducible E2E runner + result artifacts (9-12 Oct), timing/reproducibility (16-18 Oct),
repeated-run comparison automation (20 Oct), regression baseline check (22 Oct onward).

    python -m reflexicode_guard.e2e_runner --bench benchmarks/controlled_prs --out results/latest --repeats 5
    python -m reflexicode_guard.e2e_runner ... --baseline benchmarks/baseline_summary.json        # regression gate
    python -m reflexicode_guard.e2e_runner ... --update-baseline benchmarks/baseline_summary.json  # freeze

Outputs in --out: manifest.json, summary.json, per_case.csv, comparison.md, findings_<variant>.json
Exit code 1 if results are not reproducible or differ from --baseline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass

from . import __version__
from .diff_utils import commentable_lines
from .metrics import aggregate, score_case
from .models import PRContext
from .pipeline import run_pipeline
from .review_entry import context_from_fixture
from .variants import Variant, all_variants

PKG_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Case:
    name: str
    ctx: PRContext
    truth: set
    kinds: dict          # (file,line) -> "detectable" | "semantic"


def load_benchmark(path: str) -> list[Case]:
    cases = []
    for name in sorted(os.listdir(path)):
        d = os.path.join(path, name)
        if not os.path.exists(os.path.join(d, "expected.json")):
            continue
        exp = json.load(open(os.path.join(d, "expected.json"), encoding="utf-8"))
        truth = {(x["file"], x["line"]) for x in exp["defects"]}
        kinds = {(x["file"], x["line"]): x.get("kind", "") for x in exp["defects"]}
        cases.append(Case(name, context_from_fixture(d), truth, kinds))
    if not cases:
        raise SystemExit(f"No cases (folders with expected.json) found in {path}")
    return cases


def _sha(*chunks: bytes) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()[:16]


def tree_hash(root: str, exts=(".py", ".json", ".diff")) -> str:
    h = hashlib.sha256()
    for dp, dn, fn in sorted(os.walk(root)):
        dn[:] = sorted(d for d in dn if d != "__pycache__")
        for f in sorted(fn):
            if f.endswith(exts):
                h.update(os.path.relpath(os.path.join(dp, f), root).encode())
                h.update(open(os.path.join(dp, f), "rb").read())
    return h.hexdigest()[:16]


def run_variant_once(v: Variant, cases: list[Case]) -> dict:
    per_case, case_ms = [], []
    t_all = time.perf_counter()
    for c in cases:
        t0 = time.perf_counter()
        findings, stats = run_pipeline(c.ctx, v.analyzers, options=v.options)
        case_ms.append((time.perf_counter() - t0) * 1000)
        preds = {(f.file, f.line) for f in findings}
        valid = {fl.path: commentable_lines(fl.patch) for fl in c.ctx.files}
        would_422 = sum(1 for f in findings if f.line not in valid.get(f.file, set()))
        sc = score_case(c.ctx, preds, c.truth)
        missed = sorted(c.truth - preds)
        per_case.append({"case": c.name, "score": sc, "stats": stats, "would_422": would_422,
                         "missed": [{"file": f, "line": l, "kind": c.kinds[(f, l)]} for f, l in missed],
                         "false_positives": [{"file": f, "line": l} for f, l in sorted(preds - c.truth)],
                         "findings": [f.to_dict() for f in findings]})
    return {"per_case": per_case, "total_ms": (time.perf_counter() - t_all) * 1000, "case_ms": case_ms}


def canonical_hash(run: dict) -> str:
    """Hash of everything except timings."""
    return _sha(json.dumps(run["per_case"], sort_keys=True).encode())


def run_all(bench: str, repeats: int, variants: list[Variant]) -> dict:
    cases = load_benchmark(bench)
    result = {"variants": {}}
    for v in variants:
        runs = [run_variant_once(v, cases) for _ in range(repeats)]
        hashes = [canonical_hash(r) for r in runs]
        first = runs[0]
        agg = aggregate([pc["score"] for pc in first["per_case"]])
        totals = [r["total_ms"] for r in runs]
        flat_case_ms = [ms for r in runs for ms in r["case_ms"]]
        result["variants"][v.name] = {
            "description": v.description,
            "metrics": agg,
            "would_422_total": sum(pc["would_422"] for pc in first["per_case"]),
            "gate_drops": {k: sum(pc["stats"][k] for pc in first["per_case"])
                           for k in ("invalid_schema", "off_diff", "evidence_mismatch", "duplicates")},
            "reproducible": len(set(hashes)) == 1,
            "result_hash": hashes[0],
            "timing_ms": {"repeats": repeats, "pass_median": round(statistics.median(totals), 3),
                          "pass_min": round(min(totals), 3), "pass_max": round(max(totals), 3),
                          "per_pr_median": round(statistics.median(flat_case_ms), 3),
                          "per_pr_p95": round(sorted(flat_case_ms)[int(0.95 * (len(flat_case_ms) - 1))], 3)},
            "_per_case": first["per_case"],
        }
    result["n_cases"] = len(cases)
    result["n_truth_defects"] = sum(len(c.truth) for c in cases)
    result["n_clean_prs"] = sum(1 for c in cases if not c.truth)
    return result


def fresh_process_hashes(bench: str, repeats: int) -> dict:
    """Re-run in a brand-new interpreter (different hash seed) and return variant -> hash."""
    env = {**os.environ, "PYTHONHASHSEED": "random"}
    p = subprocess.run([sys.executable, "-m", "reflexicode_guard.e2e_runner", "--bench", bench,
                        "--repeats", "1", "--print-hashes"], capture_output=True, text=True, env=env,
                       cwd=os.path.dirname(PKG_DIR))
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-500:])
    return json.loads(p.stdout)


# ---- artifacts -------------------------------------------------------------------------------------
def write_artifacts(res: dict, out: str, bench: str, repeats: int, fresh: dict | None):
    os.makedirs(out, exist_ok=True)
    manifest = {"tool_version": __version__, "python": sys.version.split()[0], "platform": platform.platform(),
                "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "repeats": repeats, "benchmark_dir": bench, "benchmark_hash": tree_hash(bench),
                "code_hash": tree_hash(PKG_DIR), "n_cases": res["n_cases"],
                "n_truth_defects": res["n_truth_defects"], "n_clean_prs": res["n_clean_prs"],
                "variants": list(res["variants"]),
                "reproduce_with": f"python -m reflexicode_guard.e2e_runner --bench {bench} --out {out} --repeats {repeats}"}
    summary = {}
    for name, v in res["variants"].items():
        v = dict(v)
        per_case = v.pop("_per_case")
        if fresh is not None:
            v["fresh_process_match"] = fresh.get(name) == v["result_hash"]
        summary[name] = v
        json.dump({"variant": name, "cases": per_case}, open(os.path.join(out, f"findings_{name}.json"), "w"), indent=2)
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=2)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)

    with open(os.path.join(out, "per_case.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "case", "truth", "tp", "fp", "fn", "hunk_tp", "hunk_fp", "hunk_fn", "would_422", "missed", "false_positives"])
        for name, v in res["variants"].items():
            for pc in v["_per_case"]:
                s = pc["score"]
                w.writerow([name, pc["case"], s["truth_lines"], s["line"]["tp"], s["line"]["fp"], s["line"]["fn"],
                            s["hunk"]["tp"], s["hunk"]["fp"], s["hunk"]["fn"], pc["would_422"],
                            ";".join(f"{m['file']}:{m['line']}" for m in pc["missed"]),
                            ";".join(f"{m['file']}:{m['line']}" for m in pc["false_positives"])])

    L = [f"# Comparison (generated {manifest['generated_at_utc']})", "",
         f"Benchmark: **{res['n_cases']} controlled PRs**, {res['n_truth_defects']} ground-truth defects, "
         f"{res['n_clean_prs']} clean PRs. Repeats per variant: {repeats}. "
         "Synthetic benchmark + simulated hallucinations: see docs/EVALUATION_NOTES.md before quoting numbers.", "",
         "## Line-level (exact line must match)", "",
         "FPR = flagged innocent lines inside the diff / innocent lines inside the diff. Would-422 = flags outside the diff (GitHub rejects them).", "",
         "| Variant | TP | FP | FN | Precision | Recall | F1 | FPR | Would-422 | Reproducible |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in summary.items():
        m = v["metrics"]["line"]
        L.append(f"| {name} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | "
                 f"{v['metrics']['fpr']:.3f} | {v['would_422_total']} | {'yes' if v['reproducible'] else '**NO**'} |")
    L += ["", "## Hunk-level (credit for being in the right hunk)", "",
          "| Variant | TP | FP | FN | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|"]
    for name, v in summary.items():
        m = v["metrics"]["hunk"]
        L.append(f"| {name} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} |")
    L += ["", "## What each gate removed (first run)", "",
          "| Variant | invalid schema | off-diff line | evidence mismatch | duplicates |", "|---|---|---|---|---|"]
    for name, v in summary.items():
        d = v["gate_drops"]
        L.append(f"| {name} | {d['invalid_schema']} | {d['off_diff']} | {d['evidence_mismatch']} | {d['duplicates']} |")
    L += ["", "## Processing time", "", "| Variant | whole benchmark median (ms) | per PR median (ms) | per PR p95 (ms) |", "|---|---|---|---|"]
    for name, v in summary.items():
        t = v["timing_ms"]
        L.append(f"| {name} | {t['pass_median']} | {t['per_pr_median']} | {t['per_pr_p95']} |")
    L += ["", "Timing here covers only the M4 pipeline on mock analyzers. Real Ollama/Qwen latency must be measured separately (M3)."]
    open(os.path.join(out, "comparison.md"), "w").write("\n".join(L) + "\n")
    return summary


def metric_view(summary: dict) -> dict:
    return {n: {"metrics": v["metrics"], "would_422_total": v["would_422_total"], "gate_drops": v["gate_drops"],
                "result_hash": v["result_hash"]} for n, v in summary.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ReflexiCode-Guard reproducible E2E / ablation runner")
    ap.add_argument("--bench", default="benchmarks/controlled_prs")
    ap.add_argument("--out", default="results/latest")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--baseline", help="frozen summary to compare against (regression gate)")
    ap.add_argument("--update-baseline", metavar="FILE", help="write current metrics as the new frozen baseline")
    ap.add_argument("--no-fresh-process", action="store_true", help="skip the new-interpreter reproducibility check")
    ap.add_argument("--print-hashes", action="store_true", help="internal: print variant->hash JSON and exit")
    args = ap.parse_args(argv)

    res = run_all(args.bench, args.repeats, all_variants())
    if args.print_hashes:
        print(json.dumps({n: v["result_hash"] for n, v in res["variants"].items()}))
        return 0
    fresh = None if args.no_fresh_process else fresh_process_hashes(args.bench, 1)
    summary = write_artifacts(res, args.out, args.bench, args.repeats, fresh)

    rc = 0
    for name, v in summary.items():
        flag = "ok " if v["reproducible"] and v.get("fresh_process_match", True) else "FAIL"
        m = v["metrics"]["line"]
        print(f"[{flag}] {name:28s} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} "
              f"FP={m['fp']} 422s={v['would_422_total']} pr_median={v['timing_ms']['per_pr_median']}ms")
        if flag == "FAIL":
            rc = 1
    if args.update_baseline:
        json.dump(metric_view(summary), open(args.update_baseline, "w"), indent=2)
        print(f"baseline written: {args.update_baseline}")
    if args.baseline:
        base = json.load(open(args.baseline))
        cur = metric_view(summary)
        if base != cur:
            diffs = [n for n in set(base) | set(cur) if base.get(n) != cur.get(n)]
            print(f"REGRESSION: results differ from frozen baseline for: {sorted(diffs)}")
            rc = 1
        else:
            print("baseline match: results identical to frozen baseline")
    print(f"artifacts: {args.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
