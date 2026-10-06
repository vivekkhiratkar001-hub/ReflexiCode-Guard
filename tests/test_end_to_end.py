import json

import pytest

from src.llm import MockProvider, ReviewContextBuilder, ReviewPipeline
from src.llm.parser import ReviewResponseError
from src.llm.schemas import ReviewFinding


def build_review_context() -> object:
    return ReviewContextBuilder().build(
        pr_title="Fix user database access",
        pr_description="Use the repository layer for user lookups and parameterized queries.",
        issue_context="All database access must go through repository classes.",
        changed_code={
            "src/services/user_service.py": (
                "def get_user(user_id):\n"
                "    query = f\"SELECT * FROM users WHERE id = {user_id}\"\n"
                "    return db.execute(query)\n"
            )
        },
        repository_rules=[
            {
                "rule": "Database access must use repository layer.",
                "category": "Architecture",
                "severity": "High",
                "source": "repository-policy",
            }
        ],
        analysis_findings=[
            {
                "file": "src/services/user_service.py",
                "line": 2,
                "category": "Security",
                "severity": "High",
                "message": "Potential SQL injection due to dynamically constructed SQL query.",
                "rule": "SQLI-001",
                "evidence": "query = f\"SELECT * FROM users WHERE id = {user_id}\"",
            }
        ],
        changed_lines={"src/services/user_service.py": [1, 2, 3]},
    )


def test_end_to_end_pipeline_returns_valid_review_findings() -> None:
    context = build_review_context()
    response = json.dumps(
        {
            "findings": [
                {
                    "file": "src/services/user_service.py",
                    "line": 2,
                    "category": "Security",
                    "severity": "High",
                    "message": "Direct SQL interpolation bypasses the repository layer and is vulnerable to injection.",
                    "evidence": "query = f\"SELECT * FROM users WHERE id = {user_id}\"",
                    "suggestion": "Move the query into a repository method and pass parameters separately.",
                }
            ]
        }
    )

    findings = ReviewPipeline(MockProvider(response)).review(context)

    assert len(findings) == 1
    assert all(isinstance(item, ReviewFinding) for item in findings)
    assert findings[0].file == "src/services/user_service.py"
    assert findings[0].line == 2
    assert findings[0].severity == "High"
    assert findings[0].evidence.startswith("query = f")
    assert "repository" in findings[0].message.lower() or "injection" in findings[0].message.lower()


def test_end_to_end_pipeline_returns_empty_findings_for_clean_review() -> None:
    context = build_review_context()
    response = '{"findings": []}'

    findings = ReviewPipeline(MockProvider(response)).review(context)

    assert findings == []


def test_end_to_end_pipeline_handles_multiple_valid_findings_and_deduplicates() -> None:
    context = build_review_context()
    response = json.dumps(
        {
            "findings": [
                {
                    "file": "src/services/user_service.py",
                    "line": 2,
                    "category": "Security",
                    "severity": "High",
                    "message": "Direct SQL interpolation bypasses the repository layer and is vulnerable to injection.",
                    "evidence": "query = f\"SELECT * FROM users WHERE id = {user_id}\"",
                    "suggestion": "Use a repository method.",
                },
                {
                    "file": "src/services/user_service.py",
                    "line": 2,
                    "category": "Security",
                    "severity": "High",
                    "message": "Direct SQL interpolation bypasses the repository layer and is vulnerable to injection.",
                    "evidence": "query = f\"SELECT * FROM users WHERE id = {user_id}\"",
                    "suggestion": "Use a repository method.",
                },
                {
                    "file": "src/services/user_service.py",
                    "line": 2,
                    "category": "Architecture",
                    "severity": "Medium",
                    "message": "User access should go through a repository abstraction.",
                    "evidence": "query = f\"SELECT * FROM users WHERE id = {user_id}\"",
                    "suggestion": "Add a UserRepository and call it here.",
                },
            ]
        }
    )

    findings = ReviewPipeline(MockProvider(response)).review(context)

    assert len(findings) == 2
    assert {finding.category for finding in findings} == {"Security", "Architecture"}


def test_end_to_end_pipeline_rejects_malformed_or_invalid_provider_output() -> None:
    context = build_review_context()

    with pytest.raises(ReviewResponseError):
        ReviewPipeline(MockProvider('{not valid json}')).review(context)
