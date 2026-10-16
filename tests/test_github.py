"""Unit tests for Member 1 GitHub collection, diffs, mapping, and HTTP handling."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from src.github.client import GitHubClient
from src.github.adapters import normalize_analysis_findings
from src.github.collector import PullRequestCollector, prepare_review_inputs
from src.github.config import build_config
from src.github.diff import parse_patch
from src.github.exceptions import (
    GitHubAuthError,
    GitHubAPIError,
    GitHubConfigError,
    GitHubNotFoundError,
    GitHubRateLimitError,
)
from src.github.mapper import FindingCommentMapper, extract_fingerprints, format_comment_body
from src.llm.schemas import ReviewFinding


ADDED_PATCH = """@@ -0,0 +1,3 @@
+def add(a, b):
+    return a - b
+
"""

MODIFIED_PATCH = """@@ -10,4 +10,5 @@ def login(user):
     token = user.token
-    query = f"SELECT * FROM users WHERE id = {user.id}"
+    query = "SELECT * FROM users WHERE id = %s"
+    return db.execute(query, user.id)
     return db.execute(query)
"""

RENAMED_PATCH = """@@ -1,2 +1,2 @@
 def helper():
-    return 1
+    return 2
"""

DELETED_PATCH = """@@ -1,3 +0,0 @@
-def unused():
-    pass
-
"""


class FakeResponse:
    def __init__(self, status_code, json_data=None, text="", headers=None, reason="Error"):
        self.status_code = status_code
        self._json = json_data
        self.text = text if text else (json.dumps(json_data) if json_data is not None else "")
        self.headers = headers or {}
        self.reason = reason

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


def _finding(**overrides: object) -> ReviewFinding:
    data: dict[str, object] = {
        "file": "src/math.py",
        "line": 2,
        "category": "Bug",
        "severity": "High",
        "message": "The function subtracts instead of adding.",
        "evidence": "return a - b",
        "suggestion": "Return a + b instead.",
    }
    data.update(overrides)
    return ReviewFinding.model_validate(data)


def test_parse_added_modified_renamed_and_deleted_patches() -> None:
    added = parse_patch(ADDED_PATCH)
    assert added.added_line_map()[2] == "    return a - b"
    assert 1 in added.right_commentable_lines()

    modified = parse_patch(MODIFIED_PATCH)
    assert any("SELECT * FROM users WHERE id = %s" in text for text in modified.added_line_map().values())
    assert modified.deleted_line_map()

    renamed = parse_patch(RENAMED_PATCH)
    assert 2 in renamed.right_commentable_lines()

    deleted = parse_patch(DELETED_PATCH)
    assert 1 in deleted.left_commentable_lines()
    assert deleted.added_line_map() == {}


def test_parse_patch_handles_missing_patch() -> None:
    parsed = parse_patch(None)
    assert parsed.lines == []
    assert parsed.right_commentable_lines() == set()


def test_prepare_review_inputs_skips_binary_and_missing_patches() -> None:
    from src.github.models import FileChange

    files = [
        FileChange("src/math.py", "added", patch=ADDED_PATCH, reviewable=True),
        FileChange("logo.png", "added", patch=None, reviewable=False, skip_reason="Binary or unsupported file type."),
        FileChange("notes.bin", "modified", patch=None, reviewable=False, skip_reason="No patch is available"),
        FileChange("old.py", "removed", patch=DELETED_PATCH, reviewable=True),
        FileChange("util.py", "renamed", previous_path="helpers.py", patch=RENAMED_PATCH, reviewable=True),
    ]
    changed_code, changed_lines, patches, skipped = prepare_review_inputs(files)
    assert "src/math.py" in changed_code
    assert changed_lines["src/math.py"] == [1, 2, 3]
    assert "old.py" in changed_code
    assert "util.py" in changed_code
    assert any("logo.png" in item for item in skipped)
    assert any("notes.bin" in item for item in skipped)
    assert "logo.png" not in patches


def test_analysis_fixture_fields_adapt_to_shared_contract() -> None:
    finding = normalize_analysis_findings(
        [
            {
                "file": "src/app.py",
                "line": 4,
                "category": "Security",
                "severity": "High",
                "message": "Potential injection.",
                "rule": "SQLI-001",
                "evidence": "query = build_query(user_input)",
            }
        ]
    )[0]

    assert finding.file_path == "src/app.py"
    assert finding.line_number == 4
    assert finding.category == "Security"
    assert finding.severity == "High"
    assert finding.evidence == "query = build_query(user_input)"
    assert finding.suggestion == ""


def test_pr_metadata_and_changed_files_from_client() -> None:
    session = Mock()
    session.headers = {}
    pr_payload = {
        "number": 12,
        "title": "Fix math",
        "body": "Correct subtraction.",
        "html_url": "https://github.com/acme/demo/pull/12",
        "state": "open",
        "base": {"ref": "main", "sha": "base123"},
        "head": {"ref": "feature", "sha": "abc123def"},
    }
    files_payload = [
        {
            "filename": "src/math.py",
            "status": "added",
            "patch": ADDED_PATCH,
            "additions": 3,
            "deletions": 0,
            "changes": 3,
            "sha": "file1",
        },
        {
            "filename": "src/app.py",
            "status": "modified",
            "patch": MODIFIED_PATCH,
            "additions": 2,
            "deletions": 1,
            "changes": 3,
        },
        {
            "filename": "helpers.py",
            "status": "renamed",
            "previous_filename": "old_helpers.py",
            "patch": RENAMED_PATCH,
        },
        {
            "filename": "gone.py",
            "status": "removed",
            "patch": DELETED_PATCH,
        },
        {
            "filename": "photo.png",
            "status": "added",
            "additions": 0,
            "deletions": 0,
        },
    ]

    def request(method, url, **kwargs):
        if url.endswith("/pulls/12"):
            return FakeResponse(200, pr_payload)
        if url.endswith("/files"):
            return FakeResponse(200, files_payload)
        raise AssertionError(url)

    session.request.side_effect = request
    client = GitHubClient(token="test-token", session=session, sleeper=lambda _: None)
    info, files = PullRequestCollector(client).collect("acme", "demo", 12)
    assert info.title == "Fix math"
    assert info.base_branch == "main"
    assert info.head_branch == "feature"
    assert info.head_sha == "abc123def"
    assert info.base_sha == "base123"
    assert info.number == 12
    paths = {item.path: item for item in files}
    assert paths["src/math.py"].status == "added"
    assert paths["helpers.py"].previous_path == "old_helpers.py"
    assert paths["photo.png"].reviewable is False


def test_client_paginates_file_lists() -> None:
    session = Mock()
    session.headers = {}
    page1 = FakeResponse(
        200,
        [{"filename": "a.py", "status": "modified", "patch": MODIFIED_PATCH}],
        headers={"Link": '<https://api.github.com/repos/a/b/pulls/1/files?page=2>; rel="next"'},
    )
    page2 = FakeResponse(200, [{"filename": "b.py", "status": "added", "patch": ADDED_PATCH}])

    def request(method, url, **kwargs):
        if "page=2" in url:
            return page2
        return page1

    session.request.side_effect = request
    client = GitHubClient(session=session, sleeper=lambda _: None)
    files = client.list_pull_request_files("a", "b", 1)
    assert [item["filename"] for item in files] == ["a.py", "b.py"]


def test_client_auth_and_not_found_errors() -> None:
    session = Mock()
    session.headers = {}
    session.request.return_value = FakeResponse(401, text="Unauthorized")
    client = GitHubClient(session=session, sleeper=lambda _: None)
    with pytest.raises(GitHubAuthError):
        client.get_pull_request("a", "b", 1)

    session.request.return_value = FakeResponse(404, text="Not Found")
    with pytest.raises(GitHubNotFoundError):
        client.get_pull_request("a", "b", 999)


def test_client_uses_bearer_token_without_network_request() -> None:
    session = Mock()
    session.headers = {}
    token = "synthetic-test-token"

    GitHubClient(token=token, session=session)

    assert session.headers["Authorization"] == f"Bearer {token}"
    session.request.assert_not_called()


def test_client_redacts_token_from_api_errors() -> None:
    session = Mock()
    session.headers = {}
    token = "synthetic-test-token"
    session.request.return_value = FakeResponse(500, text=f"rejected credential: {token}")
    client = GitHubClient(token=token, session=session, max_retries=1)

    with pytest.raises(GitHubAPIError) as error:
        client.get_pull_request("acme", "demo", 1)

    assert token not in str(error.value)

    session.request.return_value = FakeResponse(500, reason=token)
    with pytest.raises(GitHubAPIError) as error:
        client.get_pull_request("acme", "demo", 1)

    assert token not in str(error.value)


def test_client_redacts_token_from_request_exceptions() -> None:
    session = Mock()
    session.headers = {}
    token = "synthetic-test-token"
    session.request.side_effect = requests.ConnectionError(f"connection failed: {token}")
    client = GitHubClient(token=token, session=session, max_retries=1)

    with pytest.raises(GitHubAPIError) as error:
        client.get_pull_request("acme", "demo", 1)

    assert token not in str(error.value)


def test_client_retries_rate_limits() -> None:
    session = Mock()
    session.headers = {}
    limited = FakeResponse(
        403,
        text="API rate limit exceeded",
        headers={"X-RateLimit-Remaining": "0", "Retry-After": "1"},
    )
    ok = FakeResponse(200, {"number": 1, "title": "x", "body": "", "base": {"ref": "main"}, "head": {"ref": "h", "sha": "s"}})
    session.request.side_effect = [limited, ok]
    slept = []
    client = GitHubClient(session=session, sleeper=lambda seconds: slept.append(seconds), max_retries=3)
    payload = client.get_pull_request("a", "b", 1)
    assert payload["number"] == 1
    assert slept == [1.0]


def test_client_raises_after_rate_limit_retries() -> None:
    session = Mock()
    session.headers = {}
    limited = FakeResponse(429, text="rate limit", headers={"Retry-After": "1"})
    session.request.return_value = limited
    client = GitHubClient(session=session, sleeper=lambda _: None, max_retries=2)
    with pytest.raises(GitHubRateLimitError):
        client.get_pull_request("a", "b", 1)


def test_missing_configuration_isolated_from_github_actions_environment(
    monkeypatch, tmp_path
) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(
            {
                "repository": {"full_name": "actions/repository"},
                "number": 42,
                "pull_request": {
                    "number": 42,
                    "head": {"sha": "synthetic-head"},
                },
            }
        ),
        encoding="utf-8",
    )
    actions_environment = {
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_REPOSITORY": "actions/repository",
        "GITHUB_OWNER": "actions",
        "GITHUB_REPO": "repository",
        "GITHUB_PR_NUMBER": "42",
        "GITHUB_TOKEN": "synthetic-read-token",
        "GITHUB_API_URL": "https://api.github.test",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
        "REFLEXICODE_USE_MOCK_LLM": "true",
        "REFLEXICODE_MOCK_LLM_RESPONSE": '{"findings":[]}',
        "REFLEXICODE_RULES_FILE": str(tmp_path / "rules.json"),
        "REFLEXICODE_ANALYSIS_FILE": str(tmp_path / "analysis.json"),
    }
    for name, value in actions_environment.items():
        monkeypatch.setenv(name, value)

    actions_config = build_config()
    assert (actions_config.owner, actions_config.repo, actions_config.pr_number) == (
        "actions",
        "repository",
        42,
    )

    for name in actions_environment:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(GitHubConfigError):
        build_config()
    with pytest.raises(GitHubConfigError):
        build_config(owner="a", repo="b", pr_number=0)


def test_config_from_cli_values() -> None:
    config = build_config(owner="acme", repo="demo", pr_number=3, dry_run=True, use_mock_llm=True)
    assert config.owner == "acme"
    assert config.repo == "demo"
    assert config.pr_number == 3
    assert config.dry_run is True


def test_environment_cannot_enable_publishing_without_explicit_publish(monkeypatch) -> None:
    monkeypatch.setenv("REFLEXICODE_DRY_RUN", "false")
    config = build_config(owner="acme", repo="demo", pr_number=3, dry_run=False)
    assert config.dry_run is True

    publishing = build_config(owner="acme", repo="demo", pr_number=3, publish=True)
    assert publishing.dry_run is False


def test_finding_maps_to_right_side_comment() -> None:
    from src.github.models import FileChange

    files = [FileChange("src/math.py", "added", patch=ADDED_PATCH, reviewable=True)]
    _, _, patches, _ = prepare_review_inputs(files)
    comment = FindingCommentMapper(files, patches).map_finding(_finding())
    assert comment is not None
    assert comment.side == "RIGHT"
    assert comment.line == 2
    assert comment.path == "src/math.py"
    assert "High" in comment.body
    assert "Bug" in comment.body
    assert "return a - b" in comment.body
    assert "Return a + b" in comment.body


def test_unmappable_finding_is_rejected() -> None:
    from src.github.models import FileChange

    files = [FileChange("src/math.py", "added", patch=ADDED_PATCH, reviewable=True)]
    _, _, patches, _ = prepare_review_inputs(files)
    finding = _finding(line=99)
    assert FindingCommentMapper(files, patches).map_finding(finding) is None


def test_deleted_file_maps_left_side() -> None:
    from src.github.models import FileChange

    files = [FileChange("old.py", "removed", patch=DELETED_PATCH, reviewable=True)]
    _, _, patches, _ = prepare_review_inputs(files)
    finding = _finding(file="old.py", line=1, evidence="def unused():")
    comment = FindingCommentMapper(files, patches).map_finding(finding)
    assert comment is not None
    assert comment.side == "LEFT"


def test_fingerprint_extraction_supports_duplicate_prevention() -> None:
    finding = _finding()
    body = format_comment_body(finding, "abc123def4567890")
    assert "abc123def4567890" in extract_fingerprints([body])
    assert extract_fingerprints(["ordinary comment"]) == set()


def test_workflow_file_is_safe_and_present() -> None:
    workflow = Path(".github/workflows/pr-review.yml").read_text(encoding="utf-8")
    assert "pull_request:" in workflow
    assert "pull_request_target:" not in workflow
    assert "python -m pytest" in workflow
    assert "ref: ${{ github.sha }}" in workflow
    assert "review:" not in workflow
    assert "python -m src.github" not in workflow
    assert "secrets.GITHUB_TOKEN" not in workflow
    assert "permissions:\n      contents: read" in workflow
    assert "persist-credentials: false" in workflow
