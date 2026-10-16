"""PR event -> review -> comment integration over real HTTP (8 Oct + 14-15 Oct tasks)."""
import json, os, shutil, tempfile, unittest
from unittest import mock

from reflexicode_guard import review_entry
from reflexicode_guard.config import Config, load_config, resolve_analyzer, load_analyzers
from reflexicode_guard.diff_utils import split_diff_by_file
from reflexicode_guard.models import Finding
from tests.fake_github_server import FakeGitHub

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def make_server(**kw) -> FakeGitHub:
    event = json.load(open(os.path.join(FIX, "pr_event.json")))
    pr = event["pull_request"]
    patches = split_diff_by_file(open(os.path.join(FIX, "sample.diff")).read())
    files = [{"filename": p, "status": "modified", "patch": patch, "additions": 5} for p, patch in patches.items()]
    return FakeGitHub(pr, files, **kw)


class WorkflowCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.out = os.path.join(self.tmp, "findings.json")
        self.cwd = os.getcwd(); os.chdir(self.tmp)       # so no stray reflexicode.json is picked up
        self.fake = make_server(); self.url = self.fake.start()
        self.event = os.path.join(FIX, "pr_event.json")
    def tearDown(self):
        os.chdir(self.cwd); self.fake.stop(); shutil.rmtree(self.tmp, ignore_errors=True)
    def run_entry(self, token="t0ken", extra=(), env=None):
        e = {"GITHUB_TOKEN": token, "GITHUB_API_URL": self.url, **(env or {})}
        if not token: e.pop("GITHUB_TOKEN")
        with mock.patch.dict(os.environ, e, clear=False):
            if not token: os.environ.pop("GITHUB_TOKEN", None)
            return review_entry.main(["--event", self.event, "--output", self.out, *extra])


class TestPRTrigger(WorkflowCase):
    def test_event_to_published_review(self):
        self.assertEqual(self.run_entry(), 0)
        self.assertEqual(len(self.fake.reviews), 1)
        rv = self.fake.reviews[0]
        self.assertEqual(rv["commit_id"], "abc1234def5678abc1234def5678abc1234def56")   # anchored to head SHA
        self.assertEqual(len(rv["comments"]), 3)
        self.assertEqual(json.load(open(self.out))["publish"]["posted"], 3)

    def test_rerun_on_synchronize_is_quiet(self):
        self.run_entry(); self.run_entry()
        self.assertEqual(len(self.fake.reviews), 1)                         # second run posted nothing new
        self.assertEqual(json.load(open(self.out))["publish"]["skipped_existing"], 3)

    def test_draft_pr_skipped(self):
        ev = json.load(open(self.event)); ev["pull_request"]["draft"] = True
        self.event = os.path.join(self.tmp, "draft.json"); json.dump(ev, open(self.event, "w"))
        self.assertEqual(self.run_entry(), 0)
        self.assertEqual((self.fake.reviews, self.fake.requests), ([], []))   # no API traffic at all

    def test_non_protected_base_skipped(self):
        ev = json.load(open(self.event)); ev["pull_request"]["base"]["ref"] = "feature/x"
        self.event = os.path.join(self.tmp, "b.json"); json.dump(ev, open(self.event, "w"))
        self.run_entry(); self.assertEqual(self.fake.reviews, [])

    def test_fork_pr_forced_dry_run(self):
        ev = json.load(open(self.event)); ev["pull_request"]["head"]["repo"]["full_name"] = "someone/fork"
        self.event = os.path.join(self.tmp, "f.json"); json.dump(ev, open(self.event, "w"))
        self.run_entry()
        self.assertEqual(self.fake.reviews, [])
        self.assertTrue(json.load(open(self.out))["publish"]["dry_run"])


class TestAuthAndFailures(WorkflowCase):
    def test_missing_token_fails_loudly(self):
        from reflexicode_guard.github_client import GitHubError
        with self.assertRaises(GitHubError) as cm: self.run_entry(token=None)
        self.assertEqual(cm.exception.status, 401)

    def test_transient_502_is_retried(self):
        self.fake.flaky_left = 2
        with mock.patch("time.sleep"): self.assertEqual(self.run_entry(), 0)
        self.assertEqual(len(self.fake.reviews), 1)

    def test_422_degrades_to_issue_comment(self):
        # An analyzer reports a line the fake server considers off-diff, but our own filter
        # would normally stop it. Disable the filter to prove the publisher fallback works.
        from reflexicode_guard.github_client import GitHubClient
        from reflexicode_guard.pipeline import run_pipeline, PipelineOptions
        from reflexicode_guard.publisher import publish
        client = GitHubClient("t", api_url=self.url)
        ctx = client.build_context("demo-org/demo-repo", 7)
        bad = lambda c: [Finding("src/auth.py", 500, "high", "x", "off-diff finding", "evidence")]
        fs, st = run_pipeline(ctx, [bad], options=PipelineOptions(filter_to_diff=False))
        r = publish(client, ctx, fs, st)
        self.assertTrue(r["fallback"]); self.assertEqual(len(self.fake.issue_comments), 1)
        self.assertEqual(self.fake.reviews, [])


class TestChecklistItems(WorkflowCase):
    """One test per 8 Oct item: trigger, checkout/deps (static), publishing."""
    def test_workflow_yaml_has_required_pieces(self):
        y = open(os.path.join(os.path.dirname(__file__), "..", ".github", "workflows", "reflexicode-guard.yml")).read()
        for needle in ("pull_request:", "actions/checkout@v4", "pull-requests: write", "contents: read",
                       "concurrency:", "upload-artifact", "GITHUB_TOKEN"):
            self.assertIn(needle, y, f"workflow missing {needle}")
        self.assertNotIn("pull_request_target", y)       # unsafe with untrusted fork code
    def test_requirements_file_exists(self):
        self.assertTrue(os.path.exists(os.path.join(os.path.dirname(__file__), "..", "requirements.txt")))


class TestConfigPlugins(unittest.TestCase):
    def test_default_and_env_override(self):
        self.assertEqual(load_config("nonexistent.json", env={}).analyzers, ["demo_static"])
        c = load_config("nonexistent.json", env={"RFG_ANALYZERS": "demo_static,mock_llm", "LLM_PROVIDER": "ollama"})
        self.assertEqual((c.analyzers, c.llm_provider), (["demo_static", "mock_llm"], "ollama"))
    def test_resolve_builtin_and_dotted(self):
        self.assertTrue(callable(resolve_analyzer("mock_llm")))
        self.assertTrue(callable(resolve_analyzer("reflexicode_guard.pipeline:demo_static_analyzer")))
    def test_bad_spec_errors_clearly(self):
        for bad in ("nocolon", "no.such.module:fn", "reflexicode_guard.pipeline:nope"):
            with self.assertRaises(ValueError): resolve_analyzer(bad)
    def test_unknown_config_key_rejected(self):
        p = tempfile.mktemp(suffix=".json"); json.dump({"analyzrs": []}, open(p, "w"))
        with self.assertRaises(ValueError): load_config(p, env={})
    def test_mock_llm_plugs_in_next_to_static(self):
        from reflexicode_guard.models import PRContext, ChangedFile
        from reflexicode_guard.pipeline import run_pipeline
        ctx = PRContext("o/r", 1, "t", "", "a", "main", "f", "b", "h",
                        files=[ChangedFile("a.py", "modified", "@@ -0,0 +1,2 @@\n+subprocess.run(x, shell=True)\n+api_key = 'abcdefgh'")])
        fs, _ = run_pipeline(ctx, load_analyzers(Config(analyzers=["demo_static", "mock_llm"])))
        self.assertEqual({f.source for f in fs}, {"static", "llm"})


if __name__ == "__main__":
    unittest.main()


class TestManualDispatch(WorkflowCase):
    """workflow_dispatch path: `--pr N` fetches the PR from the API instead of reading an event file."""
    def test_pr_number_mode_posts_review(self):
        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "t", "GITHUB_API_URL": self.url, "GITHUB_REPOSITORY": "demo-org/demo-repo"}):
            rc = review_entry.main(["--pr", "7", "--output", self.out])
        self.assertEqual(rc, 0); self.assertEqual(len(self.fake.reviews), 1)
    def test_pr_number_mode_dry_run_posts_nothing(self):
        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "t", "GITHUB_API_URL": self.url, "GITHUB_REPOSITORY": "demo-org/demo-repo"}):
            review_entry.main(["--pr", "7", "--dry-run", "--output", self.out])
        self.assertEqual(self.fake.reviews, [])
        with open(self.out) as fh: self.assertTrue(json.load(fh)["publish"]["dry_run"])
    def test_workflow_has_manual_and_regression_jobs(self):
        y = open(os.path.join(os.path.dirname(__file__), "..", ".github", "workflows", "reflexicode-guard.yml")).read()
        for needle in ("workflow_dispatch:", "pr_number", "regression:", "needs: regression", "scripts/regression.sh"):
            self.assertIn(needle, y)
