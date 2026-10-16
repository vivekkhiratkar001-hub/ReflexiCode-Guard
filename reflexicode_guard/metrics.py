"""Evaluation metrics. Unit of comparison = unique (file, line) location.

line-level : a prediction is a TP only if its exact (file, line) is a ground-truth defect.
hunk-level : a prediction is a TP if its hunk contains a ground-truth defect; a defect is found if its
             hunk contains any prediction. (Shows how much credit a tool gets for being merely 'near'.)
FPR = FP_on_commentable_lines / (commentable lines - ground-truth lines): the share of innocent lines inside the
diff that were flagged. Flags OUTSIDE the diff are not part of that universe; they are reported separately as
'would_422' (GitHub would reject them).
"""
from __future__ import annotations

from .diff_utils import commentable_lines, hunk_ranges
from .models import PRContext


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def _hunk_of(ctx: PRContext) -> dict:
    return {f.path: hunk_ranges(f.patch) for f in ctx.files}


def _hunk_id(hunks: dict, file: str, line: int):
    for i, (a, b) in enumerate(hunks.get(file, [])):
        if a <= line <= b:
            return (file, i)
    return None


def score_case(ctx: PRContext, preds: set, truth: set) -> dict:
    tp, fp, fn = len(preds & truth), len(preds - truth), len(truth - preds)
    hunks = _hunk_of(ctx)
    t_h = {_hunk_id(hunks, f, l) for f, l in truth} - {None}
    p_h = {_hunk_id(hunks, f, l) for f, l in preds}
    p_h_noscope = {h for h in p_h if h is not None}
    htp_pred = sum(1 for p in preds if _hunk_id(hunks, *p) in t_h)
    hfp = len(preds) - htp_pred
    hfn = len(t_h - p_h_noscope)
    universe = {f.path: commentable_lines(f.patch) for f in ctx.files}
    fp_in_universe = sum(1 for (f, l) in preds - truth if l in universe.get(f, set()))
    return {"line": {"tp": tp, "fp": fp, "fn": fn},
            "hunk": {"tp": htp_pred, "fp": hfp, "fn": hfn},
            "universe_lines": sum(len(v) for v in universe.values()),
            "fp_in_universe": fp_in_universe, "truth_lines": len(truth)}


def aggregate(case_scores: list[dict]) -> dict:
    out = {}
    for level in ("line", "hunk"):
        out[level] = prf(sum(c[level]["tp"] for c in case_scores),
                         sum(c[level]["fp"] for c in case_scores),
                         sum(c[level]["fn"] for c in case_scores))
    neg = sum(c["universe_lines"] - c["truth_lines"] for c in case_scores)
    out["fpr"] = round(sum(c["fp_in_universe"] for c in case_scores) / neg, 4) if neg else 0.0
    out["negatives"] = neg
    return out
