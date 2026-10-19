"""Comprehensive regression test suite covering all 18 defect scenarios and contracts."""

import json
from unittest.mock import Mock, patch

import pytest

from src.analysis import analyze_diff, analyze_pr
from src.analysis.rules.base import RuleFinding
from src.github import GitHubClient, ReviewOrchestrator, publish_review, review_pr
from src.github.adapters import SharedContractReviewer
from src.github.collector import prepare_review_inputs
from src.github.mapper import FindingCommentMapper
from src.github.models import FileChange, PullRequestInfo, ReviewSummary
from src.llm import (
    MockProvider,
    OllamaProvider,
    ReviewContextBuilder,
    ReviewPipeline,
    ReviewPromptBuilder,
    ReviewResponseError,
    ReviewResponseParser,
)
from src.shared import (
    AnalysisFinding,
    AnalysisResult,
    FindingCategory,
    PRContext,
    ReviewResult,
    ReviewStatus,
    Severity,
)


def _make_context(
    files: dict[str, str],
    changed_lines: dict[str, list[int]],
    pr_title: str = "Regression PR",
) -> object:
    line_content = {
        path: {
            line: text
            for line, text in enumerate(source.splitlines(), start=1)
            if line in changed_lines.get(path, [])
        }
        for path, source in files.items()
    }
    return ReviewContextBuilder().build(
        pr_title=pr_title,
        pr_description="Test description",
        issue_context="Issue context",
        changed_code=files,
        changed_lines=changed_lines,
        changed_line_content=line_content,
    )


# 1. Empty diffs
def test_case_01_empty_diff_handling() -> None:
    ctx = PRContext(
        repository_owner="owner",
        repository_name="repo",
        pr_number=101,
        title="Empty PR",
        description="Nothing changed",
        base_sha="base",
        head_sha="head",
        diff="",
    )
    result = review_pr(ctx, initial_findings=[])
    assert result.findings == []
    assert result.status == ReviewStatus.COMPLETE
    assert result.head_sha == "head"
    assert result.block_merge is False
    assert result.severity_counts == {}


# 2. Large diffs and extremely large files
def test_case_02_large_diff_and_large_files() -> None:
    large_lines = [f"def func_{i}():\n    return {i}" for i in range(300)]
    content = "\n".join(large_lines)
    ctx = _make_context(
        files={"src/large.py": content},
        changed_lines={"src/large.py": list(range(1, 100))},
    )
    pipeline = ReviewPipeline(MockProvider('{"findings": []}'))
    findings = pipeline.review(ctx)
    assert findings == []
    assert "src/large.py" in pipeline.last_reviewed_files or "src/large.py" in pipeline.last_partially_reviewed_files


# 3. Unicode and unusually long lines
def test_case_03_unicode_and_unusually_long_lines() -> None:
    very_long_line = "value = '" + "🚀🔥unicode_test_accentué_漢字_" * 50 + "'"
    code = f"def test():\n    {very_long_line}\n"
    ctx = _make_context(
        files={"src/unicode_file.py": code},
        changed_lines={"src/unicode_file.py": [2]},
    )
    builder = ReviewPromptBuilder()
    prompt_result = builder.build_with_metadata(ctx)
    assert "src/unicode_file.py" in prompt_result.included_files
    # Verify the long line was safely bounded and didn't crash
    assert "..." in prompt_result.prompt or "unicode_test" in prompt_result.prompt
    assert len(prompt_result.prompt) <= builder.max_prompt_chars


# 4. Renamed, deleted, binary, and unsupported files
def test_case_04_renamed_deleted_binary_unsupported_files() -> None:
    files = [
        FileChange(
            path="binary.png",
            status="added",
            patch=None,
            reviewable=False,
            skip_reason="Binary or unsupported file type.",
        ),
        FileChange(
            path="src/old.py",
            status="removed",
            patch="@@ -1,2 +0,0 @@\n-import os\n-os.system('rm')",
            reviewable=True,
        ),
        FileChange(
            path="src/new.py",
            status="renamed",
            previous_path="src/legacy.py",
            patch="@@ -1,1 +1,1 @@\n-old_func()\n+new_func()",
            reviewable=True,
        ),
    ]
    changed_code, changed_lines, patches, skipped = prepare_review_inputs(files)
    assert any("binary.png" in item for item in skipped)
    assert "src/new.py" in changed_code
    assert "src/old.py" in changed_code
    # Removed file maps lines to deleted_line_map
    assert 1 in patches["src/old.py"].left_commentable_lines()
    assert 1 in patches["src/new.py"].right_commentable_lines()


# 5. Context truncation and token-budget enforcement
def test_case_05_ollama_context_overflow_enforced() -> None:
    # If prompt tokens + num_predict exceed num_ctx, OllamaProvider raises ValueError
    provider = OllamaProvider(num_predict=512, num_ctx=2048)
    # 2048 tokens context, 512 predict leaves 1536 tokens for prompt (~6144 characters)
    huge_prompt = "x" * 7000  # ~1750 tokens -> 1750 + 512 = 2262 > 2048
    with pytest.raises(ValueError, match="exceeds the configured Ollama context window"):
        provider.review(huge_prompt)


# 6. Overlapping contexts and duplicate findings (static + semantic)
def test_case_06_overlapping_static_and_semantic_deduplication() -> None:
    static_finding = AnalysisFinding(
        file_path="src/app.py",
        line_number=5,
        category="security",
        severity="high",
        message="eval() executes dynamically constructed code",
        evidence="eval(user_query)",
        suggestion="Avoid dynamic eval",
    )
    # LLM returns same issue with wider line context in evidence: 'return eval(user_query)'
    semantic_response = json.dumps(
        {
            "findings": [
                {
                    "file": "src/app.py",
                    "line": 5,
                    "category": "Security",
                    "severity": "High",
                    "message": "Dangerous eval call detected",
                    "evidence": "return eval(user_query)",
                    "suggestion": "Use safe parsing",
                }
            ]
        }
    )
    reviewer = SharedContractReviewer(
        pipeline=ReviewPipeline(MockProvider(semantic_response)),
        changed_code={"src/app.py": "def f():\n    return eval(user_query)\n"},
        changed_lines={"src/app.py": [5]},
        changed_line_content={"src/app.py": {5: "    return eval(user_query)"}},
    )
    ctx = PRContext("o", "r", 1, "t", "d", "b", "h", diff="")
    combined = reviewer.review(ctx, [static_finding])
    # The duplicate semantic finding must be deduplicated
    assert len(combined) == 1
    assert combined[0] == static_finding


# 7. Malformed, empty, and incomplete LLM responses
def test_case_07_malformed_empty_and_incomplete_responses() -> None:
    parser = ReviewResponseParser()
    with pytest.raises(ReviewResponseError, match="empty response"):
        parser.parse("")
    with pytest.raises(ReviewResponseError, match="valid JSON"):
        parser.parse("Here is my review: no JSON.")
    with pytest.raises(ReviewResponseError, match="truncated JSON"):
        parser.parse('{"findings": [{"file": "a.py"')


# 8. LLM timeout, rate limiting, and unavailable configuration
def test_case_08_llm_timeout_and_graceful_pipeline_handling() -> None:
    class FailingProvider:
        timeout = 10

        def review(self, _prompt: str) -> str:
            raise RuntimeError("Ollama connection timed out after 10s")

    pipeline = ReviewPipeline(FailingProvider())
    ctx = _make_context({"src/f.py": "def a(): return 1"}, {"src/f.py": [1]})
    findings = pipeline.review(ctx)
    assert findings == []
    assert pipeline.last_failed_batch_count == 1
    assert any("timed out" in f for f in pipeline.last_review_failures)
    assert pipeline.last_unreviewed_files == ("src/f.py",)


# 9. Invalid findings and incorrect line numbers
def test_case_09_invalid_findings_rejected() -> None:
    ctx = _make_context({"src/test.py": "line 1\nline 2\n"}, {"src/test.py": [1]})
    parser = ReviewResponseParser()

    # Wrong file
    bad_file = '{"findings": [{"file": "src/other.py", "line": 1, "category": "Bug", "severity": "High", "message": "msg", "evidence": "line 1"}]}'
    with pytest.raises(ReviewResponseError, match="not in changed_code"):
        parser.parse(bad_file, ctx)

    # Line not in changed_lines
    bad_line = '{"findings": [{"file": "src/test.py", "line": 2, "category": "Bug", "severity": "High", "message": "msg", "evidence": "line 2"}]}'
    with pytest.raises(ReviewResponseError, match="not listed as a changed line"):
        parser.parse(bad_line, ctx)

    # Line 0
    line_zero = '{"findings": [{"file": "src/test.py", "line": 0, "category": "Bug", "severity": "High", "message": "msg", "evidence": "line 1"}]}'
    with pytest.raises(ReviewResponseError, match="schema validation"):
        parser.parse(line_zero, ctx)


# 10. Findings on changed versus unchanged lines
def test_case_10_changed_vs_unchanged_lines_static() -> None:
    diff = (
        "diff --git a/src/sample.py b/src/sample.py\n"
        "--- a/src/sample.py\n"
        "+++ b/src/sample.py\n"
        "@@ -1,3 +1,3 @@\n"
        " eval(unchanged)\n"
        "+eval(new_added)\n"
        " eval(context_line)\n"
    )
    findings = analyze_diff(diff)
    assert len(findings) == 1
    assert findings[0].line_number == 2
    assert "new_added" in findings[0].evidence


# 11. Clean change that should not receive fabricated findings
def test_case_11_clean_change_receives_no_findings() -> None:
    diff = (
        "diff --git a/src/math.py b/src/math.py\n"
        "--- a/src/math.py\n"
        "+++ b/src/math.py\n"
        "@@ -0,0 +1,3 @@\n"
        "+def add(a: int, b: int) -> int:\n"
        "+    return a + b\n"
        "+\n"
    )
    findings = analyze_diff(diff)
    assert findings == []


# 12. Seeded defect detected by the relevant rule
def test_case_12_seeded_defect_detected() -> None:
    diff = (
        "diff --git a/src/vuln.py b/src/vuln.py\n"
        "--- a/src/vuln.py\n"
        "+++ b/src/vuln.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+import os\n"
        "+os.system('echo dangerous')\n"
    )
    findings = analyze_diff(diff)
    assert len(findings) == 1
    assert findings[0].category == "security"
    assert findings[0].severity == "high"
    assert "os.system" in findings[0].evidence


# 13. A static-analysis rule failing while other rules continue safely
def test_case_13_static_rule_failure_isolation() -> None:
    diff = (
        "diff --git a/src/mixed.py b/src/mixed.py\n"
        "--- a/src/mixed.py\n"
        "+++ b/src/mixed.py\n"
        "@@ -0,0 +1,3 @@\n"
        "+eval(user_data)\n"
        "+def bad_func(arg=[]):\n"
        "+    pass\n"
    )
    # Mock run_static_ast_rules to fail with an exception
    with patch(
        "src.analysis.analyzer.run_static_ast_rules",
        side_effect=RuntimeError("Simulated bug in static AST rule"),
    ):
        findings = analyze_diff(diff)
        # Security rule (eval) must STILL run and be reported!
        assert len(findings) == 1
        assert findings[0].category == "security"
        assert "eval" in findings[0].evidence


# 14. Severity-count and output-schema consistency
def test_case_14_severity_count_and_schema_consistency() -> None:
    f1 = AnalysisFinding("a.py", 1, "security", "high", "m1", "e1", "s1")
    f2 = AnalysisFinding("b.py", 2, "security", "critical", "m2", "e2", "s2")
    f3 = AnalysisFinding("c.py", 3, "static", "low", "m3", "e3", "s3")

    ctx = PRContext("o", "r", 42, "PR", "desc", "b", "h", diff="")
    res = review_pr(ctx, initial_findings=[f1, f2, f3])

    assert isinstance(res, ReviewResult)
    assert res.pr_number == 42
    assert res.head_sha == "h"
    assert res.block_merge is True  # High and Critical block merge
    assert res.severity_counts == {"High": 1, "Critical": 1, "Low": 1}

    # AnalysisResult model exists and is consistent
    ar = AnalysisResult(findings=[f1, f2, f3])
    assert len(ar.findings) == 3


# 15. Stale PR head SHA handling
def test_case_15_stale_pr_head_sha_handling() -> None:
    ctx = PRContext("o", "r", 42, "PR", "desc", "base", "current-sha-999")
    stale_review = ReviewResult(
        pr_number=42,
        findings=[],
        summary="ok",
        status=ReviewStatus.COMPLETE,
        head_sha="old-stale-sha-111",
    )
    with pytest.raises(ValueError, match="Stale pull request revision"):
        publish_review(ctx, stale_review)

    # PR number mismatch check
    mismatch_ctx = PRContext("o", "r", 99, "PR", "desc", "base", "head")
    with pytest.raises(ValueError, match="PR number mismatch"):
        publish_review(mismatch_ctx, stale_review)


# 16. GitHub API failures handled accurately
def test_case_16_github_api_failures_handled() -> None:
    session = Mock()
    session.headers = {}
    response = Mock()
    response.status_code = 503
    response.text = "Service Unavailable"
    response.reason = "Service Unavailable"
    session.request.return_value = response

    client = GitHubClient(token="fake", session=session, max_retries=1)
    with pytest.raises(Exception) as exc_info:
        client.get_pull_request("o", "r", 1)
    assert "HTTP 503" in str(exc_info.value) or "Service Unavailable" in str(exc_info.value)


# 17. Dry-run mode without real publication
def test_case_17_dry_run_mode_without_publication() -> None:
    session = Mock()
    session.headers = {}
    client = GitHubClient(token="fake", session=session)
    ctx = PRContext("owner", "repo", 10, "Title", "Body", "base", "head")
    review = ReviewResult(
        pr_number=10,
        findings=[],
        summary="Summary",
        status=ReviewStatus.COMPLETE,
        head_sha="head",
    )
    # When dry_run=True, no POST requests are sent
    publish_review(ctx, review, client=client, dry_run=True)
    assert not session.request.called


# 18. Repeat-review behavior and duplicate comments
def test_case_18_repeat_review_duplicate_comment_prevention() -> None:
    from src.github.comments import ReviewPublisher
    from src.github.mapper import finding_fingerprint, format_comment_body
    from src.llm.schemas import ReviewFinding

    finding = ReviewFinding.model_validate(
        {
            "file": "src/mod.py",
            "line": 10,
            "category": "Bug",
            "severity": "Medium",
            "message": "Potential index error",
            "evidence": "items[idx]",
            "suggestion": "Check length",
        }
    )
    fp = finding_fingerprint(finding, "RIGHT")
    body = format_comment_body(finding, fp)

    client = Mock()
    client.list_review_comments.return_value = [{"body": body}]
    publisher = ReviewPublisher(client)

    pr_info = PullRequestInfo("o", "r", 1, "t", "b", "main", "feat", "head", "url", "open")
    existing_fps = publisher.existing_fingerprints(pr_info)
    assert fp in existing_fps

    mapper = FindingCommentMapper(
        files=[FileChange("src/mod.py", "modified", patch="@@ -10,1 +10,1 @@\n+items[idx]")],
        patches={"src/mod.py": Mock(right_commentable_lines=lambda: {10}, left_commentable_lines=lambda: set())},
    )
    comments, unmapped = mapper.map_findings([finding])
    assert len(comments) == 1

    unique, duplicates = publisher.filter_duplicates(comments, existing_fps)
    assert len(duplicates) == 1
    assert len(unique) == 0
