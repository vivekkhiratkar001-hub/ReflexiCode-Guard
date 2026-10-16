import json, os, tempfile, unittest
from reflexicode_guard.diff_utils import parse_patch, changed_lines, commentable_lines, split_diff_by_file
from reflexicode_guard.events import parse_pr_event, should_review, extract_linked_issues, load_event
from reflexicode_guard.github_client import GitHubClient, GitHubError
from reflexicode_guard.models import Finding
from reflexicode_guard.pipeline import run_pipeline
from reflexicode_guard.publisher import publish, build_review_payload, MARK_RE
from reflexicode_guard.review_entry import context_from_fixture, main

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


class FakeGH:
    """Records calls; stands in for GitHubClient in publisher tests."""
    def __init__(self, existing=None, fail_review=None):
        self.existing, self.fail_review, self.reviews, self.issue_comments = existing or [], fail_review, [], []
    def list_review_comments(self, repo, n): return self.existing
    def create_review(self, repo, n, sha, body, comments, event="COMMENT"):
        if self.fail_review: raise self.fail_review
        self.reviews.append((sha, body, comments))
    def create_issue_comment(self, repo, n, body): self.issue_comments.append(body)


class TestEvents(unittest.TestCase):
    def setUp(self): self.ev = parse_pr_event(load_event(os.path.join(FIX, "pr_event.json")))
    def test_parse(self):
        self.assertEqual((self.ev.repo, self.ev.number, self.ev.base_ref), ("demo-org/demo-repo", 7, "main"))
        self.assertFalse(self.ev.from_fork)
    def test_gate_ok(self): self.assertTrue(should_review(self.ev)[0])
    def test_gate_draft(self):
        self.ev.draft = True; self.assertFalse(should_review(self.ev)[0])
    def test_gate_branch(self):
        self.ev.base_ref = "feature/x"; self.assertFalse(should_review(self.ev)[0])
    def test_gate_action(self):
        self.ev.action = "closed"; self.assertFalse(should_review(self.ev)[0])
    def test_non_pr_payload(self):
        with self.assertRaises(ValueError): parse_pr_event({"action": "push"})
    def test_linked_issues(self): self.assertEqual(extract_linked_issues("Fixes #3, closes #12 and fix #3"), [3, 12])


class TestDiff(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(FIX, "sample.diff")) as f: self.patch = split_diff_by_file(f.read())["src/auth.py"]
    def test_added_lines(self): self.assertEqual(changed_lines(self.patch), {3, 4, 6, 7, 8})
    def test_commentable_includes_context(self): self.assertTrue({1, 2, 3, 5, 9} <= commentable_lines(self.patch))
    def test_removed_tracked(self): self.assertEqual(parse_patch(self.patch)["removed"], [4])
    def test_empty_patch(self): self.assertEqual(changed_lines(None), set())
    def test_no_newline_marker(self):
        p = "@@ -1,1 +1,1 @@\n-a\n+b\n\\ No newline at end of file"
        self.assertEqual(changed_lines(p), {1})


class TestPipeline(unittest.TestCase):
    def setUp(self): self.ctx = context_from_fixture(FIX)
    def test_demo_findings(self):
        fs, st = run_pipeline(self.ctx)
        self.assertEqual({f.rule_id for f in fs}, {"SEC001", "SEC002", "STY001"})
        self.assertEqual(fs[0].severity, "critical")           # ranked by severity
    def test_off_diff_and_invalid_filtered(self):
        bad = [Finding("src/auth.py", 999, "high", "x", "m", "e"),    # off-diff
               Finding("src/auth.py", 3, "high", "x", "m", ""),       # no evidence
               Finding("src/auth.py", 3, "high", "x", "m", "e")]
        _, st = run_pipeline(self.ctx, [lambda c: bad])
        self.assertEqual((st["off_diff"], st["invalid_schema"], st["published_candidates"]), (1, 1, 1))
    def test_dedupe(self):
        f = Finding("src/auth.py", 3, "low", "x", "m", "e")
        fs, st = run_pipeline(self.ctx, [lambda c: [f], lambda c: [f]])
        self.assertEqual((len(fs), st["duplicates"]), (1, 1))
    def test_crashing_analyzer_isolated(self):
        def boom(c): raise RuntimeError("x")
        fs, st = run_pipeline(self.ctx, [boom, lambda c: [Finding("src/auth.py", 3, "low", "x", "m", "e")]])
        self.assertEqual((len(fs), st["analyzer_errors"]), (1, 1))
    def test_max_findings_truncates(self):
        fs, st = run_pipeline(self.ctx, max_findings=1)
        self.assertEqual((len(fs), st["truncated"]), (1, 2))


class TestPublisher(unittest.TestCase):
    def setUp(self):
        self.ctx = context_from_fixture(FIX); self.fs, self.st = run_pipeline(self.ctx)
    def test_payload_shape(self):
        p = build_review_payload(self.fs, self.st)
        self.assertEqual(len(p["comments"]), 3)
        self.assertTrue(all(c["side"] == "RIGHT" and c["line"] > 0 for c in p["comments"]))
        self.assertIn("reflexicode-guard:", p["comments"][0]["body"])
    def test_publish_anchors_to_head_sha(self):
        gh = FakeGH(); r = publish(gh, self.ctx, self.fs, self.st)
        self.assertEqual((r["posted"], gh.reviews[0][0]), (3, self.ctx.head_sha))
    def test_rerun_does_not_repost(self):
        first = FakeGH(); publish(first, self.ctx, self.fs, self.st)
        existing = [{"body": c["body"]} for c in first.reviews[0][2]]
        gh = FakeGH(existing=existing); r = publish(gh, self.ctx, self.fs, self.st)
        self.assertEqual((r["posted"], r["skipped_existing"], len(gh.reviews)), (0, 3, 0))
    def test_422_falls_back_to_issue_comment(self):
        gh = FakeGH(fail_review=GitHubError(422, "bad line")); r = publish(gh, self.ctx, self.fs, self.st)
        self.assertTrue(r["fallback"]); self.assertEqual(len(gh.issue_comments), 1)
    def test_other_errors_propagate(self):
        with self.assertRaises(GitHubError): publish(FakeGH(fail_review=GitHubError(401, "no")), self.ctx, self.fs, self.st)
    def test_dry_run_posts_nothing(self):
        gh = FakeGH(); r = publish(gh, self.ctx, self.fs, self.st, dry_run=True)
        self.assertEqual((r["posted"], len(gh.reviews)), (3, 0))


class TestClient(unittest.TestCase):
    def test_build_context_with_fake_transport(self):
        pr = json.load(open(os.path.join(FIX, "pr_event.json")))["pull_request"]
        pr["labels"] = [{"name": "bug"}]
        def transport(method, url, headers, body):
            if url.endswith("/pulls/7"): return 200, json.dumps(pr)
            if "/files" in url: return 200, json.dumps([{"filename": "a.py", "status": "added", "patch": "@@ -0,0 +1 @@\n+x", "additions": 1}])
            return 404, "nope"
        ctx = GitHubClient("t", transport=transport).build_context("demo-org/demo-repo", 7)
        self.assertEqual((ctx.head_sha[:7], ctx.labels, ctx.linked_issues, ctx.files[0].path), ("abc1234", ["bug"], [3], "a.py"))
    def test_retry_on_5xx_then_success(self):
        calls = []
        def transport(m, u, h, b):
            calls.append(1); return (502, "bad") if len(calls) < 3 else (200, "{}")
        GitHubClient("t", transport=transport, sleep=lambda s: None).get_pull("o/r", 1)
        self.assertEqual(len(calls), 3)
    def test_4xx_raises(self):
        with self.assertRaises(GitHubError):
            GitHubClient("t", transport=lambda *a: (404, "x")).get_pull("o/r", 1)


class TestEndToEndLocal(unittest.TestCase):
    """The Week-1 milestone: first local E2E simulation, no network."""
    def test_offline_run_writes_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "findings.json")
            self.assertEqual(main(["--offline", FIX, "--output", out]), 0)
            data = json.load(open(out))
            self.assertEqual(len(data["findings"]), 3)
            self.assertTrue(data["publish"]["dry_run"])
            self.assertEqual(data["head_sha"], "abc1234def5678abc1234def5678abc1234def56")


if __name__ == "__main__":
    unittest.main()
