import json
from pathlib import Path

from src.llm import MockProvider, ReviewContextBuilder, ReviewPipeline, ReviewPromptBuilder
from src.llm.parser import ReviewResponseParser


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> object:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_review_context_preserves_supporting_evidence_and_lines() -> None:
    rules = _fixture("sample_repository_rules.json")
    findings = _fixture("sample_analysis_findings.json")
    changed_code = {
        "src/services/user_service.py": (
            "def fetch_user(email):\n"
            "    query = \"SELECT * FROM users WHERE email = '\" + email + \"'\"\n"
            "    return db.execute(query)\n"
        )
    }

    context = ReviewContextBuilder().build(
        pr_title="Fix direct database access in user service",
        pr_description="Use the repository layer instead of direct SQL in the user service.",
        issue_context="All database access must go through repository classes.",
        changed_code=changed_code,
        repository_rules=rules,
        analysis_findings=findings,
        changed_lines={"src/services/user_service.py": [1, 2, 3]},
    )

    assert context.pr_title == "Fix direct database access in user service"
    assert context.issue_context == "All database access must go through repository classes."
    assert context.repository_rules[0]["rule"] == "Database access must use the repository layer."
    assert context.analysis_findings[0]["rule"] == "SQLI-001"
    assert context.changed_code["src/services/user_service.py"].startswith("def fetch_user")
    assert context.changed_lines == {"src/services/user_service.py": [1, 2, 3]}


def test_prompt_builder_includes_pr_issue_rules_and_static_finding_sections() -> None:
    rules = _fixture("sample_repository_rules.json")
    findings = _fixture("sample_analysis_findings.json")
    context = ReviewContextBuilder().build(
        pr_title="Fix direct database access in user service",
        pr_description="Use the repository layer instead of direct SQL in the user service.",
        issue_context="All database access must go through repository classes.",
        changed_code={"src/services/user_service.py": "query = \"SELECT * FROM users WHERE email = '\" + email + \"'\""},
        repository_rules=rules,
        analysis_findings=findings,
        changed_lines={"src/services/user_service.py": [1]},
    )

    prompt = ReviewPromptBuilder().build(context)

    for section in (
        "PR / Issue Intent",
        "Changed Code",
        "Repository Rules",
        "AST / Static Analysis Findings",
        "Changed-line information",
        "Focus on the changed code.",
        "Use PR/issue context to understand intent.",
        "Use repository rules as project-specific constraints.",
        "Use AST/static findings as supporting evidence.",
    ):
        assert section in prompt


def test_mock_provider_pipeline_accepts_full_context() -> None:
    rules = _fixture("sample_repository_rules.json")
    findings = _fixture("sample_analysis_findings.json")
    changed_code = {
        "src/services/user_service.py": (
            "def fetch_user(email):\n"
            "    query = \"SELECT * FROM users WHERE email = '\" + email + \"'\"\n"
            "    return db.execute(query)\n"
        )
    }
    context = ReviewContextBuilder().build(
        pr_title="Fix direct database access in user service",
        pr_description="Use the repository layer instead of direct SQL in the user service.",
        issue_context="All database access must go through repository classes.",
        changed_code=changed_code,
        repository_rules=rules,
        analysis_findings=findings,
        changed_lines={"src/services/user_service.py": [1, 2, 3]},
    )
    response = json.dumps(
        {
            "findings": [
                {
                    "file": "src/services/user_service.py",
                    "line": 2,
                    "category": "Security",
                    "severity": "High",
                    "message": "Direct SQL construction bypasses the repository layer and is vulnerable to injection.",
                    "evidence": "query = \"SELECT * FROM users WHERE email = '\" + email + \"'\"",
                    "suggestion": "Move the query into a repository method and pass parameters separately.",
                }
            ]
        }
    )

    pipeline = ReviewPipeline(MockProvider(response), parser=ReviewResponseParser())
    findings = pipeline.review(context)

    assert len(findings) == 1
    assert findings[0].file == "src/services/user_service.py"
    assert findings[0].severity == "High"
    assert "repository" in findings[0].message.lower()
    assert isinstance(pipeline.provider.last_input, str)


def test_invalid_or_unsupported_findings_do_not_reach_final_output() -> None:
    context = ReviewContextBuilder().build(
        pr_title="Fix direct database access in user service",
        issue_context="All database access must go through repository classes.",
        changed_code={"src/services/user_service.py": "def fetch_user(email):\n    return db.execute('SELECT * FROM users')"},
        repository_rules=[{"rule": "Database access must use repository layer."}],
        analysis_findings=[{"file": "src/services/user_service.py", "line": 1, "message": "Direct SQL access."}],
        changed_lines={"src/services/user_service.py": [1, 2]},
    )
    payload = json.dumps(
        {
            "findings": [
                {
                    "file": "src/services/user_service.py",
                    "line": 2,
                    "category": "Security",
                    "severity": "High",
                    "message": "Direct SQL construction bypasses the repository layer and is vulnerable to injection.",
                    "evidence": "return db.execute(\"DELETE FROM users\")",
                    "suggestion": "Use a repository method instead.",
                },
                {
                    "file": "src/services/other.py",
                    "line": 8,
                    "category": "Bug",
                    "severity": "Low",
                    "message": "Unrelated file issue.",
                    "evidence": "print(\"noop\")",
                    "suggestion": "None",
                },
            ]
        }
    )

    parser = ReviewResponseParser()
    try:
        parser.parse(payload, context)
        raise AssertionError("Expected invalid unsupported finding to be rejected")
    except Exception as exc:  # noqa: BLE001
        assert "not in changed_code" in str(exc) or "does not match supplied changed code" in str(exc)
