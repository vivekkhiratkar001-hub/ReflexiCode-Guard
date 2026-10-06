"""Unit tests for structured review context, providers, and pipeline."""

import json
import unittest
from unittest.mock import Mock, patch

import requests

from src.llm.context import ReviewContextBuilder
from src.llm.mock_provider import MockProvider
from src.llm.ollama_provider import OllamaProvider
from src.llm.parser import ReviewResponseError, ReviewResponseParser
from src.llm.pipeline import ReviewPipeline
from src.llm.prompts import ReviewPromptBuilder
from src.llm.schemas import ReviewFinding


FILE = "src/math.py"
CODE = "def add(a, b):\n    return a - b"


def finding_data(**overrides: object) -> dict[str, object]:
    """Build a valid test finding with optional field overrides."""
    data: dict[str, object] = {
        "file": FILE,
        "line": 2,
        "category": "Bug",
        "severity": "High",
        "message": "The function subtracts instead of adding.",
        "evidence": "return a - b",
        "suggestion": "Return a + b instead.",
    }
    data.update(overrides)
    return data


def response_for(*findings: dict[str, object]) -> str:
    """Serialize a response in the expected model-output envelope."""
    return json.dumps({"findings": list(findings)})


def sample_context():
    """Build a context with code and an explicit changed-line map."""
    return ReviewContextBuilder().build(
        pr_title="Fix user addition function",
        pr_description="Correct the implementation of the addition function.",
        issue_context="The function should add two numbers.",
        changed_code={FILE: CODE},
        repository_rules=[{"rule_id": "R-1", "description": "Keep math behavior correct."}],
        analysis_findings=[],
        changed_lines={FILE: [1, 2]},
    )


class ReviewResponseParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ReviewResponseParser()
        self.context = sample_context()

    def test_valid_structured_response(self) -> None:
        findings = self.parser.parse(response_for(finding_data()), self.context)
        self.assertEqual(len(findings), 1)
        self.assertIsInstance(findings[0], ReviewFinding)
        self.assertEqual(findings[0].line, 2)

    def test_multiple_review_findings(self) -> None:
        second = finding_data(
            category="Correctness",
            severity="Medium",
            message="The intended addition is not performed.",
        )
        findings = self.parser.parse(response_for(finding_data(), second), self.context)
        self.assertEqual(len(findings), 2)

    def test_no_findings(self) -> None:
        self.assertEqual(self.parser.parse('{"findings": []}', self.context), [])

    def test_malformed_json_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReviewResponseError, "valid JSON"):
            self.parser.parse('{"findings": [')

    def test_missing_findings_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReviewResponseError, "schema validation"):
            self.parser.parse('{"items": []}')

    def test_missing_required_finding_field_is_rejected(self) -> None:
        malformed = finding_data()
        del malformed["message"]
        with self.assertRaisesRegex(ReviewResponseError, "schema validation"):
            self.parser.parse(response_for(malformed))

    def test_invalid_severity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReviewResponseError, "schema validation"):
            self.parser.parse(response_for(finding_data(severity="Urgent")))

    def test_invalid_line_number_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReviewResponseError, "schema validation"):
            self.parser.parse(response_for(finding_data(line=0)))

    def test_finding_on_unchanged_line_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReviewResponseError, "not listed as a changed line"):
            self.parser.parse(response_for(finding_data(line=3)), self.context)

    def test_unmatched_evidence_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReviewResponseError, "does not match supplied"):
            self.parser.parse(
                response_for(finding_data(evidence="invented code evidence")),
                self.context,
            )

    def test_empty_file_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReviewResponseError, "must not be empty or whitespace-only"):
            self.parser.parse(response_for(finding_data(file="   ")), self.context)

    def test_file_not_in_changed_code_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReviewResponseError, "not in changed_code"):
            self.parser.parse(
                response_for(finding_data(file="src/other.py")),
                self.context,
            )

    def test_duplicate_findings_are_removed(self) -> None:
        duplicate = finding_data(message="The SQL query is constructed unsafely.")
        findings = self.parser.parse(
            response_for(
                finding_data(message="The SQL query is constructed unsafely."),
                duplicate,
            ),
            self.context,
        )
        self.assertEqual(len(findings), 1)

    def test_different_findings_on_same_line_are_preserved(self) -> None:
        findings = self.parser.parse(
            response_for(
                finding_data(category="Security", message="SQL injection risk."),
                finding_data(category="Correctness", message="Wrong operator used."),
            ),
            self.context,
        )
        self.assertEqual(len(findings), 2)

    def test_valid_context_without_changed_lines_still_works(self) -> None:
        context = ReviewContextBuilder().build(
            pr_title="Fix math behavior",
            changed_code={FILE: CODE},
            repository_rules=[],
            analysis_findings=[],
        )
        findings = self.parser.parse(response_for(finding_data(line=2)), context)
        self.assertEqual(len(findings), 1)

    def test_empty_changed_lines_mapping_is_accepted(self) -> None:
        context = ReviewContextBuilder().build(
            pr_title="Fix math behavior",
            changed_code={FILE: CODE},
            repository_rules=[],
            analysis_findings=[],
            changed_lines={},
        )
        findings = self.parser.parse(response_for(finding_data(line=2)), context)
        self.assertEqual(len(findings), 1)

    def test_mixed_valid_and_invalid_findings_are_rejected(self) -> None:
        with self.assertRaisesRegex(ReviewResponseError, "not in changed_code"):
            self.parser.parse(
                response_for(
                    finding_data(),
                    finding_data(file="src/other.py"),
                ),
                self.context,
            )

    def test_evidence_from_correct_line_is_accepted(self) -> None:
        context = ReviewContextBuilder().build(
            pr_title="Fix math behavior",
            changed_code={FILE: "def add(a, b):\n    total = a + b\n    return total"},
            repository_rules=[],
            analysis_findings=[],
            changed_lines={FILE: [1, 2, 3]},
        )
        findings = self.parser.parse(
            response_for(finding_data(line=3, evidence="return total")),
            context,
        )
        self.assertEqual(len(findings), 1)

    def test_evidence_from_unrelated_line_is_rejected(self) -> None:
        context = ReviewContextBuilder().build(
            pr_title="Fix math behavior",
            changed_code={FILE: "def add(a, b):\n    total = a + b\n    return total"},
            repository_rules=[],
            analysis_findings=[],
            changed_lines={FILE: [1, 2, 3]},
        )
        with self.assertRaisesRegex(ReviewResponseError, "does not match supplied"):
            self.parser.parse(
                response_for(finding_data(line=3, evidence="total = a + b")),
                context,
            )

    def test_analysis_finding_can_support_valid_evidence(self) -> None:
        context = ReviewContextBuilder().build(
            pr_title="Fix math behavior",
            changed_code={FILE: CODE},
            repository_rules=[],
            analysis_findings=[
                {"file": FILE, "line": 2, "message": "The function subtracts instead of adding."}
            ],
            changed_lines={FILE: [1, 2]},
        )
        findings = self.parser.parse(
            response_for(finding_data(evidence="The function subtracts instead of adding.")),
            context,
        )
        self.assertEqual(len(findings), 1)

    def test_short_bug_message_is_not_treated_as_duplicate(self) -> None:
        findings = self.parser.parse(
            response_for(
                finding_data(message="bug"),
                finding_data(message="Null deref bug in x"),
            ),
            self.context,
        )
        self.assertEqual(len(findings), 2)

    def test_non_ascii_message_is_not_emptied_into_duplicate_match(self) -> None:
        findings = self.parser.parse(
            response_for(
                finding_data(message="Riesgo de SQL en español"),
                finding_data(message="SQL risk in English"),
            ),
            self.context,
        )
        self.assertEqual(len(findings), 2)

    def test_json_extraction_prefers_structured_response_over_prior_braces(self) -> None:
        raw = 'Some text {} and then {"findings":[{"file":"src/math.py","line":2,"category":"Bug","severity":"High","message":"The function subtracts instead of adding.","evidence":"return a - b","suggestion":"Return a + b instead."}]}'
        findings = self.parser.parse(raw, self.context)
        self.assertEqual(len(findings), 1)

    def test_empty_response_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReviewResponseError, "empty response"):
            self.parser.parse("  ")


class ProviderAndPipelineTests(unittest.TestCase):
    def test_ollama_connection_failure_is_explained(self) -> None:
        provider = OllamaProvider()
        with patch(
            "src.llm.ollama_provider.requests.post",
            side_effect=requests.exceptions.ConnectionError("offline"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Make sure Ollama is running"):
                provider.review("prompt")

    def test_ollama_unexpected_api_json_is_explained(self) -> None:
        provider = OllamaProvider()
        response = Mock()
        response.json.return_value = []
        with patch("src.llm.ollama_provider.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "unexpected format"):
                provider.review("prompt")

    def test_ollama_timeout_is_explained(self) -> None:
        provider = OllamaProvider()
        with patch(
            "src.llm.ollama_provider.requests.post",
            side_effect=requests.exceptions.Timeout("slow"),
        ):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                provider.review("prompt")

    def test_ollama_http_error_is_explained(self) -> None:
        provider = OllamaProvider()
        response = Mock()
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "bad gateway",
            response=response,
        )
        response.status_code = 500
        response.text = "bad gateway"
        with patch("src.llm.ollama_provider.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "Ollama returned HTTP 500"):
                provider.review("prompt")

    def test_mock_provider_runs_without_ollama(self) -> None:
        mock = MockProvider('{"findings": []}')
        self.assertEqual(mock.review("test prompt"), '{"findings": []}')
        self.assertEqual(mock.last_input, "test prompt")

    def test_full_pipeline_builds_prompt_and_returns_findings(self) -> None:
        context = sample_context()
        mock = MockProvider(response_for(finding_data()))
        pipeline = ReviewPipeline(mock)

        findings = pipeline.review(context)

        self.assertEqual(len(findings), 1)
        self.assertIsInstance(findings[0], ReviewFinding)
        self.assertIsInstance(mock.last_input, str)
        self.assertIn("Fix user addition function", mock.last_input)
        self.assertIn("AST/static-analysis findings", mock.last_input)
        self.assertIn("unchanged lines unless the issue is directly caused", mock.last_input)

    def test_context_builder_accepts_structured_pydantic_inputs(self) -> None:
        from pydantic import BaseModel

        class Rule(BaseModel):
            rule_id: str
            description: str

        context = ReviewContextBuilder().build(
            repository_rules=[Rule(rule_id="R1", description="Rule from Member 1")]
        )
        self.assertEqual(context.repository_rules[0]["rule_id"], "R1")

    def test_prompt_builder_includes_every_context_section(self) -> None:
        prompt = ReviewPromptBuilder().build(sample_context())
        for expected in ("pr_title", "pr_description", "issue_context", "changed_code", "repository_rules", "analysis_findings"):
            with self.subTest(section=expected):
                self.assertIn(expected, prompt)


if __name__ == "__main__":
    unittest.main()
