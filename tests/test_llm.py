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
from src.llm.prompts import MAX_REVIEW_PROMPT_CHARS, ReviewPromptBuilder
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


def ollama_stream(*chunks: dict[str, object]) -> Mock:
    response = Mock()
    response.iter_lines.return_value = iter(
        json.dumps(chunk, separators=(",", ":")) for chunk in chunks
    )
    return response


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

    def test_evidence_must_match_exact_changed_line_when_line_content_is_available(self) -> None:
        context = ReviewContextBuilder().build(
            pr_title="Fix math behavior",
            changed_code={FILE: "def add(a, b):\n    total = a + b\n    return total"},
            repository_rules=[],
            analysis_findings=[],
            changed_lines={FILE: [3]},
            changed_line_content={FILE: {3: "    return total"}},
        )
        with self.assertRaisesRegex(ReviewResponseError, "does not match the changed line"):
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
    def test_ollama_success_uses_configured_timeout_and_logs_only_prompt_size(self) -> None:
        provider = OllamaProvider(timeout=37, num_predict=512, num_ctx=4096)
        response = ollama_stream(
            {"response": '{"findings": ', "done": False},
            {
                "response": '[]}',
                "done": True,
                "done_reason": "stop",
                "eval_count": 4,
                "prompt_eval_duration": 1_000_000_000,
                "eval_duration": 2_000_000_000,
            },
        )

        with patch("src.llm.ollama_provider.requests.post", return_value=response) as post:
            with self.assertLogs("src.llm.ollama_provider", level="INFO") as logs:
                result = provider.review("sensitive source content")

        self.assertEqual(result, '{"findings": []}')
        self.assertEqual(post.call_args.kwargs["timeout"], 37)
        self.assertTrue(post.call_args.kwargs["stream"])
        self.assertEqual(post.call_args.kwargs["json"]["options"], {
            "num_predict": 512,
            "num_ctx": 4096,
        })
        log_output = "\n".join(logs.output)
        self.assertIn("prompt_chars=24", log_output)
        self.assertIn("estimated_prompt_tokens=6", log_output)
        self.assertIn("first_output_seconds=", log_output)
        self.assertIn("generated_tokens=4", log_output)
        self.assertNotIn("sensitive source content", log_output)

    def test_mocked_ollama_output_passes_through_pipeline_validation(self) -> None:
        payload = response_for(finding_data())
        response = ollama_stream(
            {"response": payload, "done": True, "done_reason": "stop", "eval_count": 80}
        )
        provider = OllamaProvider()
        with patch("src.llm.ollama_provider.requests.post", return_value=response):
            findings = ReviewPipeline(provider).review(sample_context())

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].file, FILE)
        self.assertEqual(findings[0].line, 2)
        self.assertEqual(findings[0].severity, "High")
        self.assertEqual(findings[0].evidence, "return a - b")

    def test_malformed_model_json_is_rejected_by_pipeline(self) -> None:
        response = ollama_stream(
            {"response": '{"findings":[', "done": True, "done_reason": "stop"}
        )
        with patch("src.llm.ollama_provider.requests.post", return_value=response):
            with self.assertRaisesRegex(ReviewResponseError, "valid JSON"):
                ReviewPipeline(OllamaProvider()).review(sample_context())

    def test_ollama_malformed_stream_chunk_is_explained(self) -> None:
        provider = OllamaProvider()
        response = Mock()
        response.iter_lines.return_value = iter(['{"response":'])
        with patch("src.llm.ollama_provider.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "malformed streaming JSON"):
                provider.review("prompt")

    def test_ollama_empty_review_response_is_rejected(self) -> None:
        provider = OllamaProvider()
        response = ollama_stream({"response": "  ", "done": True, "done_reason": "stop"})
        with patch("src.llm.ollama_provider.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "empty review response"):
                provider.review("prompt")

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
        response.iter_lines.return_value = iter(["[]"])
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

    def test_ollama_timeout_reports_that_no_output_was_received(self) -> None:
        provider = OllamaProvider()
        with patch(
            "src.llm.ollama_provider.requests.post",
            side_effect=requests.exceptions.Timeout("slow"),
        ):
            with self.assertRaisesRegex(RuntimeError, "before first output"):
                provider.review("prompt")

    def test_ollama_timeout_after_first_chunk_reports_partial_progress(self) -> None:
        provider = OllamaProvider()

        def chunks():
            yield '{"response":"{"}'
            raise requests.exceptions.ReadTimeout("slow")

        response = Mock()
        response.iter_lines.return_value = chunks()
        with patch("src.llm.ollama_provider.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "after first output"):
                provider.review("prompt")

    def test_ollama_generation_limit_reports_truncated_json(self) -> None:
        provider = OllamaProvider(num_predict=128)
        response = ollama_stream(
            {
                "response": '{"findings":[',
                "done": True,
                "done_reason": "length",
            }
        )
        with patch("src.llm.ollama_provider.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "truncated at the configured generation token limit"):
                provider.review("prompt")

    def test_ollama_stream_ending_without_done_marker_is_incomplete(self) -> None:
        provider = OllamaProvider()
        response = ollama_stream({"response": '{"findings":[]}', "done": False})
        with patch("src.llm.ollama_provider.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "ended before completion"):
                provider.review("prompt")

    def test_ollama_rejects_invalid_generation_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "smaller than num_ctx"):
            OllamaProvider(num_predict=2048, num_ctx=2048)

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

    def test_pipeline_logs_prompt_coverage_without_source_content(self) -> None:
        sensitive = "internal_identifier_should_not_be_logged"
        context = ReviewContextBuilder().build(
            changed_code={FILE: f"def add(a, b):\n    return a - b  # {sensitive}"},
            changed_lines={FILE: [2]},
            changed_line_content={FILE: {1: "def add(a, b):", 2: f"    return a - b  # {sensitive}"}},
            analysis_findings=[
                {
                    "file": FILE,
                    "line": 2,
                    "category": "Bug",
                    "severity": "High",
                    "message": "Incorrect operation",
                    "evidence": "return a - b",
                }
            ],
        )
        pipeline = ReviewPipeline(MockProvider('{"findings": []}'))

        with self.assertLogs("src.llm.pipeline", level="INFO") as logs:
            pipeline.review(context)

        output = "\n".join(logs.output)
        self.assertIn("files_available=1", output)
        self.assertIn("files_included=1", output)
        self.assertIn("changed_lines_available=1", output)
        self.assertIn("changed_lines_included=1", output)
        self.assertIn("findings_available=1", output)
        self.assertIn("findings_supplied=1", output)
        self.assertIn("findings_included=1", output)
        self.assertIn("estimated_prompt_tokens=", output)
        self.assertNotIn(sensitive, output)

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
        for expected in (
            "PR title",
            "PR description",
            "Issue context",
            "Prioritized changed code",
            "Repository Rules",
            "AST / Static Analysis Findings",
        ):
            with self.subTest(section=expected):
                self.assertIn(expected, prompt)

    def test_prompt_is_bounded_and_reports_prioritized_omissions(self) -> None:
        changed_code = {
            f"src/module_{index:02}.py": "\n".join(
                f"value_{line} = {line}" for line in range(1, 101)
            )
            for index in range(30)
        }
        changed_lines = {path: [1, 2, 3] for path in changed_code}
        line_content = {
            path: {line: f"value_{line} = {line}" for line in (1, 2, 3)}
            for path in changed_code
        }
        findings = [
            {
                "file": path,
                "line": 2,
                "category": "Security" if index == 29 else "Static Analysis",
                "severity": "Critical" if index == 29 else "Low",
                "message": f"Finding for module {index}.",
                "evidence": "value_2 = 2",
            }
            for index, path in enumerate(changed_code)
        ]
        context = ReviewContextBuilder().build(
            pr_title="Review many files",
            changed_code=changed_code,
            analysis_findings=findings,
            changed_lines=changed_lines,
            changed_line_content=line_content,
        )

        result = ReviewPromptBuilder().build_with_metadata(context)

        self.assertLessEqual(len(result.prompt), MAX_REVIEW_PROMPT_CHARS)
        self.assertIn("src/module_29.py", result.prompt)
        self.assertTrue(result.limitations)
        self.assertLessEqual(len(result.included_files), 5)
        self.assertIn("of 30 changed files", " ".join(result.limitations))
        self.assertGreater(result.included_changed_lines, 0)
        self.assertEqual(result.available_findings, 30)
        self.assertEqual(result.supplied_findings, 5)
        self.assertEqual(result.included_findings, 5)
        self.assertNotIn('"message":"Finding for module 0."', result.prompt)
        self.assertNotIn("changed_line_content", result.prompt)

    def test_prompt_configurable_limits_select_only_high_priority_files_and_findings(self) -> None:
        changed_code = {
            f"src/module_{index}.py": f"def run_{index}():\n    return {index}"
            for index in range(8)
        }
        changed_lines = {path: [2] for path in changed_code}
        line_content = {
            path: {1: "def run():", 2: "    return 1"}
            for path in changed_code
        }
        findings = [
            {
                "file": path,
                "line": 2,
                "category": "Security" if index == 7 else "Static Analysis",
                "severity": "Critical" if index == 7 else "Low",
                "message": f"Finding {index}",
                "evidence": "return 1",
            }
            for index, path in enumerate(changed_code)
        ]
        context = ReviewContextBuilder().build(
            changed_code=changed_code,
            changed_lines=changed_lines,
            changed_line_content=line_content,
            analysis_findings=findings,
        )
        builder = ReviewPromptBuilder(
            max_prompt_chars=6000,
            code_chars=1200,
            analysis_chars=1000,
            rules_chars=300,
            max_files=2,
        )

        result = builder.build_with_metadata(context)

        self.assertLessEqual(len(result.prompt), 6000)
        self.assertEqual(len(result.included_files), 2)
        self.assertIn("src/module_7.py", result.included_files)
        self.assertEqual(result.available_findings, 8)
        self.assertEqual(result.supplied_findings, 2)
        self.assertEqual(result.included_findings, 2)
        self.assertNotIn('"message":"Finding 0"', result.prompt)
        self.assertIn("Return at most 5 important", result.prompt)


if __name__ == "__main__":
    unittest.main()
