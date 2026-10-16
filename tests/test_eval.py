"""Tests for metrics, benchmark integrity, evidence verification, runner reproducibility, regression gate."""
import json, os, shutil, tempfile, unittest

from reflexicode_guard.diff_utils import changed_lines, commentable_lines, hunk_ranges
from reflexicode_guard.e2e_runner import load_benchmark, main as runner_main, run_all, tree_hash
from reflexicode_guard.metrics import aggregate, prf, score_case
from reflexicode_guard.models import ChangedFile, Finding, PRContext
from reflexicode_guard.pipeline import run_pipeline
from reflexicode_guard.variants import all_variants

ROOT = os.path.join(os.path.dirname(__file__), "..")
BENCH = os.path.join(ROOT, "benchmarks", "controlled_prs")


def ctx_with(patch, path="a.py"):
    return PRContext("o/r", 1, "t", "", "a", "main", "f", "b", "h", files=[ChangedFile(path, "modified", patch)])


class TestMetrics(unittest.TestCase):
    def test_prf_basic_and_zero_division(self):
        self.assertEqual(prf(2, 2, 2)["f1"], 0.5)
        z = prf(0, 0, 0); self.assertEqual((z["precision"], z["recall"], z["f1"]), (0.0, 0.0, 0.0))
        self.assertEqual(prf(3, 0, 0)["f1"], 1.0)
    def test_hunk_ranges(self):
        self.assertEqual(hunk_ranges("@@ -1,2 +1,5 @@\n x\n+a\n+b\n+c\n y\n@@ -20 +30,2 @@\n+q\n+r"), [(1, 5), (30, 31)])
    def test_line_vs_hunk_credit(self):
        patch = "@@ -1,2 +1,5 @@\n x\n+a\n+b\n+c\n y"
        c = ctx_with(patch)
        s = score_case(c, {("a.py", 1)}, {("a.py", 3)})           # right hunk, wrong line
        self.assertEqual((s["line"]["tp"], s["line"]["fp"], s["line"]["fn"]), (0, 1, 1))
        self.assertEqual((s["hunk"]["tp"], s["hunk"]["fp"], s["hunk"]["fn"]), (1, 0, 0))
    def test_fpr_ignores_off_diff_flags(self):
        c = ctx_with("@@ -0,0 +1,2 @@\n+a\n+b")
        s = score_case(c, {("a.py", 999)}, set())
        self.assertEqual(s["fp_in_universe"], 0)                    # off-diff is would_422, not FPR
        self.assertEqual(aggregate([s])["fpr"], 0.0)


class TestBenchmarkIntegrity(unittest.TestCase):
    def test_every_truth_line_is_an_added_line_in_its_diff(self):
        for c in load_benchmark(BENCH):
            added = {f.path: changed_lines(f.patch) for f in c.ctx.files}
            for f, l in c.truth:
                self.assertIn(l, added[f], f"{c.name}: truth {f}:{l} is not an added line")
    def test_benchmark_shape(self):
        cs = load_benchmark(BENCH)
        self.assertEqual(len(cs), 13)
        self.assertEqual(sum(len(c.truth) for c in cs), 13)
        self.assertEqual(sum(1 for c in cs if not c.truth), 3)
        self.assertEqual(len({c.ctx.head_sha for c in cs}), 13)       # distinct commits
    def test_traps_are_present(self):
        d = {c.name: c for c in load_benchmark(BENCH)}
        ctx = d["09_preexisting_secret_in_context"].ctx
        self.assertIn("old-prod-key", ctx.files[0].patch)              # present in the diff as CONTEXT
        self.assertFalse(any("old-prod-key" in t for t in
                             [l[1:] for l in ctx.files[0].patch.splitlines() if l.startswith("+")]))


class TestEvidenceVerification(unittest.TestCase):
    P = "@@ -0,0 +1,2 @@\n+x = eval(a)\n+y = 2"
    def run_one(self, f):
        return run_pipeline(ctx_with(self.P), [lambda c: [f]])
    def test_matching_llm_evidence_kept(self):
        fs, _ = self.run_one(Finding("a.py", 1, "high", "sec", "m", "eval(a)", "llm"))
        self.assertEqual(len(fs), 1)
    def test_whitespace_differences_tolerated(self):
        fs, _ = self.run_one(Finding("a.py", 1, "high", "sec", "m", "x   =   eval(a)", "llm"))
        self.assertEqual(len(fs), 1)
    def test_wrong_llm_evidence_dropped(self):
        fs, st = self.run_one(Finding("a.py", 2, "high", "sec", "m", "eval(a)", "llm"))   # evidence is on line 1, not 2
        self.assertEqual((len(fs), st["evidence_mismatch"]), (0, 1))
    def test_deterministic_sources_not_verified(self):
        fs, _ = self.run_one(Finding("a.py", 2, "high", "sec", "m", "rule text, not code", "rule"))
        self.assertEqual(len(fs), 1)


class TestRunner(unittest.TestCase):
    def test_all_variants_reproducible_and_ordered(self):
        res = run_all(BENCH, 3, all_variants())
        v = res["variants"]
        self.assertTrue(all(x["reproducible"] for x in v.values()))
        f1 = {n: x["metrics"]["line"]["f1"] for n, x in v.items()}
        p = {n: x["metrics"]["line"]["precision"] for n, x in v.items()}
        self.assertLess(p["D_noisyLLM_no_gate"], p["E1_noisyLLM_schema+diff"])
        self.assertLessEqual(p["E1_noisyLLM_schema+diff"], p["E2_noisyLLM_full_gate"])
        self.assertGreater(f1["C_static+mockLLM"], f1["B_static_only"])             # LLM adds recall
        self.assertGreater(v["D_noisyLLM_no_gate"]["would_422_total"], 0)           # ungated -> GitHub would reject
        self.assertEqual(v["E2_noisyLLM_full_gate"]["would_422_total"], 0)
        # gating must never reduce recall
        self.assertEqual(v["D_noisyLLM_no_gate"]["metrics"]["line"]["recall"], v["E2_noisyLLM_full_gate"]["metrics"]["line"]["recall"])
    def test_only_semantic_defects_are_missed_by_llm_variants(self):
        """Pattern-based stand-ins cannot find logic bugs; that gap is what the real LLM (M3) must close."""
        res = run_all(BENCH, 1, all_variants())
        for name in ("C_static+mockLLM", "D_noisyLLM_no_gate", "E1_noisyLLM_schema+diff", "E2_noisyLLM_full_gate"):
            kinds = {m["kind"] for pc in res["variants"][name]["_per_case"] for m in pc["missed"]}
            self.assertEqual(kinds, {"semantic"}, name)
    def test_artifacts_and_baseline_gate(self):
        with tempfile.TemporaryDirectory() as d:
            out, base = os.path.join(d, "out"), os.path.join(d, "base.json")
            self.assertEqual(runner_main(["--bench", BENCH, "--out", out, "--repeats", "2", "--update-baseline", base]), 0)
            for f in ("manifest.json", "summary.json", "per_case.csv", "comparison.md", "findings_E2_noisyLLM_full_gate.json"):
                self.assertTrue(os.path.exists(os.path.join(out, f)), f)
            m = json.load(open(os.path.join(out, "manifest.json")))
            self.assertEqual((m["n_cases"], m["repeats"]), (13, 2))
            s = json.load(open(os.path.join(out, "summary.json")))
            self.assertTrue(all(v["fresh_process_match"] for v in s.values()))     # new interpreter, same hashes
            self.assertEqual(runner_main(["--bench", BENCH, "--out", out, "--repeats", "1", "--no-fresh-process", "--baseline", base]), 0)
            b = json.load(open(base)); b["B_static_only"]["metrics"]["line"]["tp"] += 1; json.dump(b, open(base, "w"))
            self.assertEqual(runner_main(["--bench", BENCH, "--out", out, "--repeats", "1", "--no-fresh-process", "--baseline", base]), 1)
    def test_tree_hash_changes_with_content(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "a.json"), "w").write("1"); h1 = tree_hash(d)
            open(os.path.join(d, "a.json"), "w").write("2"); self.assertNotEqual(h1, tree_hash(d))


if __name__ == "__main__":
    unittest.main()


class TestContractCheck(unittest.TestCase):
    def setUp(self):
        from reflexicode_guard.contract_check import check_analyzer
        self.check, self.cases = check_analyzer, load_benchmark(BENCH)
    def test_builtin_analyzers_pass(self):
        from reflexicode_guard.config import resolve_analyzer
        for s in ("demo_static", "mock_llm"):
            self.assertTrue(self.check(resolve_analyzer(s), self.cases)["ok"], s)
    def test_catches_each_kind_of_violation(self):
        f = lambda **k: Finding(**{"file": "src/config.py", "line": 7, "severity": "high", "category": "x",
                                   "message": "m", "evidence": "e", **k})
        bad = {"off-diff": lambda c: [f(line=900)], "no evidence": lambda c: [f(evidence="")],
               "wrong type": lambda c: ["oops"], "raises": lambda c: 1 / 0, "not a list": lambda c: None,
               "llm wrong evidence": lambda c: [f(source="llm", evidence="nothing like this")]}
        for name, fn in bad.items():
            self.assertFalse(self.check(fn, self.cases[:1])["ok"], name)
    def test_nondeterminism_detected(self):
        import itertools
        n = itertools.count()
        flaky = lambda c: [Finding("src/config.py", 7, "low", "x", f"msg{next(n)}", "e")]
        self.assertTrue(any("non-deterministic" in p for p in self.check(flaky, self.cases[:1])["problems"]))


class TestUnreviewableFiles(unittest.TestCase):
    def test_file_without_patch_is_reported_not_silently_passed(self):
        from reflexicode_guard.publisher import format_summary
        ctx = PRContext("o/r", 1, "t", "", "a", "main", "f", "b", "h",
                        files=[ChangedFile("big.sql", "modified", None), ChangedFile("gone.py", "removed", None),
                               ChangedFile("a.py", "modified", "@@ -0,0 +1 @@\n+x = 1")])
        fs, st = run_pipeline(ctx, [])
        self.assertEqual(st["unreviewable_files"], 1)                        # removed files do not count
        self.assertIn("1 file(s) could not be reviewed", format_summary(fs, st))
