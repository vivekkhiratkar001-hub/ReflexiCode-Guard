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
from src.llm.provider import LLMProvider
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


class SequenceMockProvider(LLMProvider):
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def review(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return '{"findings": []}'


class ReviewResponseParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ReviewResponseParser()
        self.context = sample_context()

    def test_empty_evidence_rejected_by_schema(self) -> None:
        bad = finding_data(evidence="")
        with self.assertRaises(ReviewResponseError) as caught:
            self.parser.parse(response_for(bad), self.context)
        message = str(caught.exception)
        self.assertIn("review-schema validation", message)
        self.assertIn("evidence", message)

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

    def test_parser_enforces_configured_semantic_finding_limit(self) -> None:
        parser = ReviewResponseParser(max_findings=1)
        with self.assertRaisesRegex(ReviewResponseError, "exceeded the configured"):
            parser.parse(response_for(finding_data(), finding_data(message="Another defect.")), self.context)

    def test_malformed_json_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReviewResponseError, "valid JSON"):
            self.parser.parse("not a JSON response")

    def test_truncated_json_is_reported_separately(self) -> None:
        with self.assertRaisesRegex(ReviewResponseError, "truncated JSON"):
            self.parser.parse('{"findings": [')

    def test_schema_validation_error_includes_sanitized_details(self) -> None:
        bad = finding_data(line="2")
        with self.assertRaises(ReviewResponseError) as caught:
            self.parser.parse(response_for(bad), self.context)

        message = str(caught.exception)
        self.assertIn("review-schema validation:", message)
        self.assertIn("findings.0.line", message)
        self.assertIn("Input should be a valid integer", message)
        # The raw model output (including source evidence) must not be echoed.
        self.assertNotIn('"line": "2"', message)
        self.assertNotIn(CODE, message)

    def test_severity_casing_is_normalized_and_unknown_levels_rejected(self) -> None:
        findings = self.parser.parse(response_for(finding_data(severity="HIGH")), self.context)
        self.assertEqual(findings[0].severity, "High")

        findings = self.parser.parse(response_for(finding_data(severity="critical")), self.context)
        self.assertEqual(findings[0].severity, "Critical")

        with self.assertRaisesRegex(ReviewResponseError, "review-schema validation"):
            self.parser.parse(response_for(finding_data(severity="urgent")), self.context)

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
        with self.assertRaisesRegex(ReviewResponseError, "schema validation"):
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

    def test_mixed_valid_and_invalid_findings_preserves_valid_findings(self) -> None:
        findings = self.parser.parse(
            response_for(
                finding_data(),
                finding_data(file="src/other.py"),
            ),
            self.context,
        )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].file, FILE)
        self.assertEqual(len(self.parser.last_rejections), 1)
        rejection = self.parser.last_rejections[0]
        self.assertEqual(rejection.index, 1)
        self.assertEqual(rejection.code, "unknown_file")
        self.assertEqual(rejection.file, "src/other.py")
        self.assertIn("not in changed_code", rejection.reason)
        self.assertEqual(rejection.as_dict()["code"], "unknown_file")

    def test_mixed_schema_invalid_finding_does_not_discard_valid_finding(self) -> None:
        findings = self.parser.parse(
            response_for(
                finding_data(),
                finding_data(line="2"),
            ),
            self.context,
        )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].file, FILE)
        self.assertEqual(len(self.parser.last_rejections), 1)
        rejection = self.parser.last_rejections[0]
        self.assertEqual(rejection.code, "schema_validation")
        self.assertIn("findings.1.line", rejection.reason)

    def test_all_invalid_findings_still_reject_response_and_record_reasons(self) -> None:
        with self.assertRaisesRegex(ReviewResponseError, "not in changed_code"):
            self.parser.parse(
                response_for(finding_data(file="src/other.py")),
                self.context,
            )

        self.assertEqual(len(self.parser.last_rejections), 1)
        self.assertEqual(self.parser.last_rejections[0].code, "unknown_file")

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

    def test_static_analysis_message_alone_cannot_support_finding_evidence(self) -> None:
        context = ReviewContextBuilder().build(
            pr_title="Fix math behavior",
            changed_code={FILE: CODE},
            repository_rules=[],
            analysis_findings=[
                {
                    "file": FILE,
                    "line": 2,
                    "message": "The function subtracts instead of adding.",
                    "evidence": "return a - b",
                }
            ],
            changed_lines={FILE: [1, 2]},
        )
        findings = self.parser.parse(
            response_for(finding_data(evidence="return a - b")),
            context,
        )
        self.assertEqual(len(findings), 1)

    def test_static_evidence_cannot_support_a_different_changed_line(self) -> None:
        context = ReviewContextBuilder().build(
            changed_code={FILE: CODE},
            repository_rules=[],
            analysis_findings=[
                {
                    "file": FILE,
                    "line": 2,
                    "message": "The function subtracts instead of adding.",
                    "evidence": "return a - b",
                }
            ],
            changed_lines={FILE: [1, 2]},
            changed_line_content={FILE: {1: "def add(a, b):", 2: "    return a - b"}},
        )
        with self.assertRaisesRegex(ReviewResponseError, "does not match the changed line"):
            self.parser.parse(
                response_for(
                    finding_data(line=1, evidence="return a - b")
                ),
                context,
            )

    def test_pr3_raw_response_shape_with_prose_evidence_is_rejected(self) -> None:
        path = "src/analysis/rules/security_rules.py"
        source = "def _is_plausible_private_key(block: str) -> bool:"
        context = ReviewContextBuilder().build(
            changed_code={path: source},
            repository_rules=[],
            analysis_findings=[],
            changed_lines={path: [125]},
            changed_line_content={path: {125: source}},
        )
        raw_response = json.dumps(
            {
                "findings": [
                    {
                        "file": path,
                        "line": 125,
                        "category": "Security",
                        "severity": "High",
                        "message": "Function has high cyclomatic complexity.",
                        "evidence": "Function has 34 branches, exceeding the threshold.",
                        "suggestion": "Refactor into smaller functions.",
                    },
                    {
                        "file": "tests/test_github.py",
                        "line": 382,
                        "category": "Correctness",
                        "severity": "Medium",
                        "message": "The test has high cyclomatic complexity.",
                        "evidence": "Function has 12 branches, exceeding the threshold.",
                        "suggestion": "Split into smaller tests.",
                    },
                    {
                        "file": path,
                        "line": 193,
                        "category": "Correctness",
                        "severity": "Medium",
                        "message": "Unreachable code follows a return.",
                        "evidence": "The return makes the following code unreachable.",
                        "suggestion": "Remove unreachable code.",
                    },
                ]
            }
        )

        with self.assertRaisesRegex(ReviewResponseError, "does not match the changed line"):
            self.parser.parse(raw_response, context)

    def test_pr3_shape_preserves_valid_finding_when_sibling_uses_prose_evidence(self) -> None:
        path = "src/analysis/rules/security_rules.py"
        source = "def _is_plausible_private_key(block: str) -> bool:"
        context = ReviewContextBuilder().build(
            changed_code={path: source},
            repository_rules=[],
            analysis_findings=[],
            changed_lines={path: [125]},
            changed_line_content={path: {125: source}},
        )
        raw_response = response_for(
            {
                "file": path,
                "line": 125,
                "category": "Security",
                "severity": "High",
                "message": "Private-key detection boundary is incorrect.",
                "evidence": source,
                "suggestion": None,
            },
            {
                "file": path,
                "line": 125,
                "category": "Performance",
                "severity": "Low",
                "message": "Function has high cyclomatic complexity.",
                "evidence": "Function has 34 branches, exceeding the threshold.",
                "suggestion": None,
            },
        )

        findings = self.parser.parse(raw_response, context)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].evidence, source)
        self.assertEqual(len(self.parser.last_rejections), 1)
        self.assertEqual(
            self.parser.last_rejections[0].code,
            "unsupported_evidence",
        )

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

    def test_json_prose_around_response_is_rejected(self) -> None:
        raw = response_for(finding_data()) + " untrusted trailing text"
        with self.assertRaisesRegex(ReviewResponseError, "only one JSON object"):
            self.parser.parse(raw, self.context)

    def test_markdown_wrapped_json_is_rejected(self) -> None:
        with self.assertRaises(ReviewResponseError):
            self.parser.parse(f"```json\n{response_for()}\n```", self.context)

    def test_unsupported_finding_category_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReviewResponseError, "unsupported finding category"):
            self.parser.parse(response_for(finding_data(category="Mystery")), self.context)

    def test_empty_response_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReviewResponseError, "empty response"):
            self.parser.parse("  ")


    def test_evidence_from_another_line_cannot_support_finding(self) -> None:
        context = ReviewContextBuilder().build(
            pr_title="Fix math behavior",
            changed_code={FILE: "def add(a, b):\n    total = a + b\n    return total"},
            repository_rules=[],
            analysis_findings=[],
            changed_lines={FILE: [1, 3]},
            changed_line_content={
                FILE: {
                    1: "def add(a, b):",
                    3: "    return total",
                }
            },
        )
        finding = finding_data(line=1, evidence="return total")
        with self.assertRaises(ReviewResponseError):
            self.parser.parse(response_for(finding), context)
class ProviderAndPipelineTests(unittest.TestCase):
    def test_ollama_wall_deadline_closes_active_response_and_reports_wall_timeout(self) -> None:
        class ImmediateDeadlineTimer:
            def __init__(self, interval, callback) -> None:
                self.interval = interval
                self.callback = callback

            def start(self) -> None:
                self.callback()

            def cancel(self) -> None:
                pass

            def is_alive(self) -> bool:
                return False

            def join(self, timeout=None) -> None:
                pass

        provider = OllamaProvider(timeout=120)
        response = ollama_stream(
            {"response": '{"findings":[]}', "done": True, "done_reason": "stop"}
        )
        with (
            patch("src.llm.ollama_provider.Timer", ImmediateDeadlineTimer),
            patch("src.llm.ollama_provider.requests.post", return_value=response) as post,
        ):
            with self.assertRaisesRegex(RuntimeError, "wall-clock deadline"):
                provider.review_with_deadline("prompt", timeout_seconds=12)

        self.assertEqual(post.call_args.kwargs["timeout"], 12)
        response.close.assert_called_once()
        response.iter_lines.assert_not_called()

    def test_wall_deadline_interrupts_blocked_stream_read(self) -> None:
        import threading
        import time
        from threading import Timer as RealThreadTimer

        provider = OllamaProvider(timeout=10)
        entered_read = threading.Event()
        response_closed = threading.Event()
        timer_threads = []

        class BlockingResponse:
            def raise_for_status(self):
                pass

            def iter_lines(self, decode_unicode=True):
                entered_read.set()
                if not response_closed.wait(2):
                    raise AssertionError("Response was not closed by deadline.")
                raise requests.exceptions.ConnectionError("response closed")

            def close(self):
                response_closed.set()

        class RealTimer:
            def __init__(self, interval, callback):
                self.timer = RealThreadTimer(interval, callback)
                self.interval = interval
                self.daemon = False

            def start(self):
                self.timer.daemon = self.daemon
                self.timer.start()

            def cancel(self):
                self.timer.cancel()

            def join(self, timeout=None):
                self.timer.join(timeout)

            def is_alive(self):
                return self.timer.is_alive()

        response = BlockingResponse()
        with (
            patch("src.llm.ollama_provider.Timer", RealTimer),
            patch("src.llm.ollama_provider.requests.post", return_value=response),
        ):
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "wall-clock deadline"):
                provider.review_with_deadline("prompt", timeout_seconds=0.15)
            elapsed = time.monotonic() - started

        self.assertTrue(entered_read.is_set())
        self.assertTrue(response_closed.is_set())
        self.assertLess(elapsed, 1.0)
    def test_wall_deadline_timeout_is_logged_as_total_deadline(self) -> None:
        provider = OllamaProvider(timeout=120)
        response = Mock()

        class TimerThatExpiresDuringRead:
            def __init__(self, interval, callback) -> None:
                self.callback = callback

            def start(self) -> None:
                pass

            def cancel(self) -> None:
                pass

            def is_alive(self) -> bool:
                return False

            def join(self, timeout=None) -> None:
                pass

        def iter_lines(decode_unicode=True):
            timer_instances[0].callback()
            raise requests.exceptions.ConnectionError("deadline closed response")

        timer_instances = []

        def make_timer(interval, callback):
            timer = TimerThatExpiresDuringRead(interval, callback)
            timer_instances.append(timer)
            return timer

        response.iter_lines.side_effect = iter_lines
        with (
            patch("src.llm.ollama_provider.Timer", side_effect=make_timer),
            patch("src.llm.ollama_provider.requests.post", return_value=response),
        ):
            with self.assertLogs("src.llm.ollama_provider", level="WARNING") as logs:
                with self.assertRaisesRegex(RuntimeError, "wall-clock deadline"):
                    provider.review_with_deadline("prompt", timeout_seconds=15)

        self.assertIn("outcome=timed_out", "\n".join(logs.output))
        response.close.assert_called_once()

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
            pipeline = ReviewPipeline(OllamaProvider())
            findings = pipeline.review(sample_context())
            assert findings == []
            assert pipeline.last_failed_batch_count == 1
            assert any("rejected the model response" in f or "failed" in f.lower() for f in pipeline.last_review_failures)

    def test_schema_failure_failure_message_includes_validation_details(self) -> None:
        bad_payload = response_for(finding_data(line="2"))
        response = ollama_stream(
            {"response": bad_payload, "done": True, "done_reason": "stop", "eval_count": 90}
        )
        with patch("src.llm.ollama_provider.requests.post", return_value=response):
            pipeline = ReviewPipeline(OllamaProvider())
            pipeline.review(sample_context())

        assert pipeline.last_failed_batch_count == 1
        joined = " ".join(pipeline.last_review_failures)
        assert "review-schema validation:" in joined
        assert "Input should be a valid integer" in joined
        assert CODE not in joined

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

    def test_ollama_token_limit_is_reported_as_truncated_output(self) -> None:
        provider = OllamaProvider(num_predict=128)
        response = Mock()
        response.iter_lines.return_value = iter([
            json.dumps({"response": '{"findings": [', "done": True, "done_reason": "length"})
        ])
        with patch("src.llm.ollama_provider.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "truncated at the configured generation token limit"):
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

    def test_pipeline_handles_malformed_output_gracefully(self) -> None:
        context = sample_context()
        pipeline = ReviewPipeline(MockProvider("{not valid json}"))
        findings = pipeline.review(context)

        assert findings == []
        assert pipeline.last_failed_batch_count == 1
        # The failure message contains "Member 3 rejected the model response" for ReviewResponseError
        assert any("rejected the model response" in f or "failed" in f.lower() for f in pipeline.last_review_failures)
        assert len(pipeline.last_unreviewed_files) > 0

    def test_pipeline_handles_provider_timeout_gracefully(self) -> None:
        context = sample_context()
        class TimeoutProvider(MockProvider):
            def review(self, prompt: str) -> str:
                raise RuntimeError("simulated timeout")

        pipeline = ReviewPipeline(TimeoutProvider('{"findings": []}'))
        findings = pipeline.review(context)

        assert findings == []
        assert pipeline.last_failed_batch_count == 1
        assert any("simulated timeout" in f for f in pipeline.last_review_failures)
        assert len(pipeline.last_unreviewed_files) > 0

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

    def test_ollama_timeout_salvages_complete_json_already_received(self) -> None:
        provider = OllamaProvider()
        payload = response_for(finding_data())

        def chunks():
            yield json.dumps({"response": payload, "done": False})
            raise requests.exceptions.ReadTimeout("idle after full JSON")

        response = Mock()
        response.iter_lines.return_value = chunks()
        with patch("src.llm.ollama_provider.requests.post", return_value=response):
            with self.assertLogs("src.llm.ollama_provider", level="WARNING") as logs:
                result = provider.review("prompt")

        self.assertEqual(result, payload)
        log_output = "\n".join(logs.output)
        self.assertIn("completed_after_timeout", log_output)
        self.assertNotIn(payload, log_output)

    def test_ollama_timeout_still_rejects_truncated_json(self) -> None:
        provider = OllamaProvider()

        def chunks():
            yield json.dumps({"response": '{"findings": [', "done": False})
            raise requests.exceptions.ReadTimeout("idle mid-object")

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

    def test_ollama_sends_keep_alive_to_keep_model_resident_between_chunks(self) -> None:
        provider = OllamaProvider()
        response = ollama_stream(
            {"response": '{"findings":[]}', "done": True, "done_reason": "stop", "eval_count": 4}
        )
        with patch("src.llm.ollama_provider.requests.post", return_value=response) as post:
            provider.review("prompt")

        self.assertEqual(post.call_args.kwargs["json"]["keep_alive"], "30m")

    def test_ollama_rejects_blank_keep_alive(self) -> None:
        with self.assertRaisesRegex(ValueError, "keep_alive"):
            OllamaProvider(keep_alive="   ")

    def test_ollama_chunk_non_dict_is_rejected(self) -> None:
        provider = OllamaProvider()
        response = Mock()
        response.iter_lines.return_value = ["12345"]
        with patch("src.llm.ollama_provider.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "unexpected format"):
                provider.review("prompt")

    def test_ollama_chunk_non_string_response_is_rejected(self) -> None:
        provider = OllamaProvider()
        response = Mock()
        response.iter_lines.return_value = ['{"response": 1234}']
        with patch("src.llm.ollama_provider.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "without valid review text"):
                provider.review("prompt")

    def test_ollama_empty_stream_is_rejected(self) -> None:
        provider = OllamaProvider()
        response = Mock()
        response.iter_lines.return_value = []
        with patch("src.llm.ollama_provider.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "ended before completion"):
                provider.review("prompt")

    def test_ollama_deadline_controller_safe_double_cleanup(self) -> None:
        from src.llm.ollama_provider import _RequestDeadlineController
        controller = _RequestDeadlineController(wall_clock_timeout=10.0)
        mock_resp = Mock()
        controller.response_holder["response"] = mock_resp
        controller.cleanup()
        controller.cleanup()
        self.assertEqual(mock_resp.close.call_count, 1)

    def test_ollama_options_num_thread_env_override(self) -> None:
        import os
        provider = OllamaProvider()
        response = ollama_stream({"response": '{"findings":[]}', "done": True})
        with patch.dict(os.environ, {"REFLEXICODE_OLLAMA_NUM_THREAD": "6"}):
            with patch("src.llm.ollama_provider.requests.post", return_value=response) as post:
                provider.review("prompt")
        self.assertEqual(post.call_args.kwargs["json"]["options"]["num_thread"], 6)

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
        self.assertIn('"kind":"changed"', mock.last_input)
        self.assertIn("Copy the complete", mock.last_input)

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
        self.assertIn("selected_files=1", output)
        self.assertIn("changed_lines_available=1", output)
        self.assertGreater(
            int(output.split("changed_lines_included=")[1].split(" ")[0]), 0
        )
        self.assertIn("static_findings_available=1", output)
        self.assertIn("static_findings_supplied=1", output)
        self.assertIn("static_findings_included=1", output)
        self.assertIn("estimated_prompt_tokens=", output)
        self.assertNotIn(sensitive, output)

    def test_large_pr_uses_one_bounded_request_and_reports_omissions(self) -> None:
        from src.llm.context import ReviewContextBuilder

        class RecordingProvider:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def review(self, prompt: str) -> str:
                self.prompts.append(prompt)
                return '{"findings":[]}'

        file_count = 11
        changed_code = {
            f"src/module_{index}.py": f"def run_{index}():\n    return {index}\n"
            for index in range(file_count)
        }
        changed_lines = {path: [2] for path in changed_code}
        line_content = {path: {2: source.splitlines()[1]} for path, source in changed_code.items()}
        context = ReviewContextBuilder().build(
            pr_title="Review a large change",
            changed_code=changed_code,
            changed_lines=changed_lines,
            changed_line_content=line_content,
        )
        provider = RecordingProvider()
        pipeline = ReviewPipeline(provider, max_requests=1)

        findings = pipeline.review(context)

        assert findings == []
        assert len(provider.prompts) == 1
        assert all(len(prompt) <= 8000 for prompt in provider.prompts)
        assert int(provider.prompts[0].split("Selected changed files: ", 1)[1].splitlines()[0]) == 5
        assert len(pipeline.last_reviewed_files) == 5
        assert len(pipeline.last_unreviewed_files) == file_count - 5
        assert any("scope omitted changed files" in error for error in pipeline.last_review_failures)
        assert pipeline.last_failed_batch_count == 0

    def test_chunks_prioritize_findings_and_review_sequentially(self) -> None:
        from src.llm.context import ReviewContextBuilder

        class RecordingProvider:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def review(self, prompt: str) -> str:
                self.prompts.append(prompt)
                return '{"findings": []}'

        paths = [f"src/module_{index}.py" for index in range(6)]
        analysis = [{
            "file": paths[5],
            "line": 2,
            "category": "security",
            "severity": "critical",
            "message": "High-priority security finding",
            "evidence": "    return 5",
        }]
        context = ReviewContextBuilder().build(
            changed_code={path: f"def run():\n    return {index}" for index, path in enumerate(paths)},
            changed_lines={path: [2] for path in paths},
            changed_line_content={path: {2: f"    return {index}"} for index, path in enumerate(paths)},
            analysis_findings=analysis,
        )
        provider = RecordingProvider()
        builder = ReviewPromptBuilder(max_files=2)

        pipeline = ReviewPipeline(provider, prompt_builder=builder, max_requests=2)
        findings = pipeline.review(context)

        self.assertEqual(findings, [])
        self.assertEqual(len(provider.prompts), 2)
        self.assertTrue(all('"kind":"changed"' in prompt for prompt in provider.prompts))
        self.assertIn(paths[5], provider.prompts[0])
        self.assertEqual(len(pipeline.last_reviewed_files), 4)
        self.assertIn(paths[5], pipeline.last_reviewed_files[:2])
        self.assertEqual(len(pipeline.last_unreviewed_files), 2)

    def test_file_with_omitted_changed_lines_is_not_marked_fully_reviewed(self) -> None:
        path = "src/large.py"
        source_lines = [f"value_{index} = '" + ("x" * 60) + "'" for index in range(40)]
        context = ReviewContextBuilder().build(
            changed_code={path: "\n".join(source_lines)},
            changed_lines={path: list(range(1, len(source_lines) + 1))},
            changed_line_content={
                path: {line: source for line, source in enumerate(source_lines, start=1)}
            },
        )
        builder = ReviewPromptBuilder(
            max_prompt_chars=8000,
            code_chars=1800,
            analysis_chars=600,
            rules_chars=300,
            max_files=1,
        )
        pipeline = ReviewPipeline(MockProvider('{"findings": []}'), builder)

        pipeline.review(context)

        self.assertEqual(pipeline.last_reviewed_files, ())
        self.assertEqual(pipeline.last_partially_reviewed_files, (path,))
        self.assertEqual(pipeline.last_unreviewed_files, ())
        self.assertTrue(any("partially reviewed files" in item.casefold() for item in pipeline.last_context_limitations))

    def test_aggregate_budget_skips_chunk_without_minimum_request_window(self) -> None:
        from src.llm.context import ReviewContextBuilder

        now = [0.0]

        class AdvancingProvider:
            timeout = 60

            def __init__(self) -> None:
                self.calls: list[tuple[str, int]] = []

            def review(self, prompt: str) -> str:
                self.calls.append((prompt, self.timeout))
                now[0] += 36
                return '{"findings": []}'

        paths = [f"src/module_{index}.py" for index in range(6)]
        context = ReviewContextBuilder().build(
            changed_code={path: f"def run():\n    return {index}" for index, path in enumerate(paths)},
            changed_lines={path: [2] for path in paths},
            changed_line_content={path: {2: f"    return {index}"} for index, path in enumerate(paths)},
        )
        provider = AdvancingProvider()
        pipeline = ReviewPipeline(
            provider,
            prompt_builder=ReviewPromptBuilder(max_files=2),
            max_requests=3,
            review_budget_seconds=90,
            clock=lambda: now[0],
        )

        self.assertEqual(pipeline.review(context), [])
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual([timeout for _, timeout in provider.calls], [60])
        self.assertEqual(len(pipeline.last_reviewed_files), 2)
        self.assertEqual(len(pipeline.last_unreviewed_files), 4)
        self.assertEqual(pipeline.last_batch_count, 1)
        self.assertEqual(pipeline.last_failed_batch_count, 0)
        self.assertEqual(pipeline.last_review_elapsed_seconds, 36)
        self.assertTrue(any("below the 60-second minimum request window" in item for item in pipeline.last_context_limitations))

    def test_inflight_chunk_overrun_does_not_schedule_another_request(self) -> None:
        from src.llm.context import ReviewContextBuilder

        now = [0.0]

        class SlowProvider:
            timeout = 30

            def __init__(self) -> None:
                self.calls = 0

            def review(self, prompt: str) -> str:
                self.calls += 1
                now[0] += 36
                return '{"findings": []}'

        paths = [f"src/module_{index}.py" for index in range(4)]
        context = ReviewContextBuilder().build(
            changed_code={path: f"def run():\n    return {index}" for index, path in enumerate(paths)},
            changed_lines={path: [2] for path in paths},
            changed_line_content={path: {2: f"    return {index}"} for index, path in enumerate(paths)},
        )
        provider = SlowProvider()
        pipeline = ReviewPipeline(
            provider,
            prompt_builder=ReviewPromptBuilder(max_files=2),
            review_budget_seconds=60,
            clock=lambda: now[0],
        )

        pipeline.review(context)

        self.assertEqual(provider.calls, 1)
        self.assertEqual(pipeline.last_review_elapsed_seconds, 36)
        self.assertEqual(pipeline.last_reviewed_files, ())
        self.assertEqual(len(pipeline.last_unreviewed_files), 4)
        self.assertEqual(pipeline.last_failed_batch_count, 1)
        self.assertTrue(any("exceeded its 30.0-second wall-clock request deadline" in item for item in pipeline.last_review_failures))

    def test_failed_chunk_does_not_discard_later_success_or_hide_omissions(self) -> None:
        from src.llm.context import ReviewContextBuilder

        class FailThenSucceedProvider:
            def __init__(self) -> None:
                self.calls = 0

            def review(self, prompt: str) -> str:
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("simulated Ollama timeout")
                changed_record = next(
                    json.loads(line)
                    for line in prompt.splitlines()
                    if line.startswith('{"file":')
                    and '"kind":"changed"' in line
                )
                return response_for(
                    finding_data(
                        file=changed_record["file"],
                        line=changed_record["line"],
                        evidence=changed_record["source"].strip(),
                        category="Correctness",
                        severity="Medium",
                        message="The returned value is incorrect.",
                    )
                )

        paths = [f"src/module_{index}.py" for index in range(6)]
        context = ReviewContextBuilder().build(
            changed_code={path: f"def run():\n    return {index}" for index, path in enumerate(paths)},
            changed_lines={path: [2] for path in paths},
            changed_line_content={path: {2: f"    return {index}"} for index, path in enumerate(paths)},
        )
        provider = FailThenSucceedProvider()
        pipeline = ReviewPipeline(
            provider,
            prompt_builder=ReviewPromptBuilder(max_files=2),
            max_requests=2,
        )

        findings = pipeline.review(context)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].message, "The returned value is incorrect.")
        self.assertEqual(provider.calls, 2)
        self.assertEqual(pipeline.last_batch_count, 2)
        self.assertEqual(pipeline.last_failed_batch_count, 1)
        self.assertEqual(len(pipeline.last_reviewed_files), 2)
        self.assertEqual(len(pipeline.last_unreviewed_files), 4)
        self.assertEqual(pipeline.last_findings_count, 1)
        self.assertTrue(any("simulated Ollama timeout" in item for item in pipeline.last_review_failures))

    def test_semantic_request_failure_reports_all_selected_files_unreviewed(self) -> None:
        from src.llm.context import ReviewContextBuilder

        class FailingProvider:
            def review(self, prompt: str) -> str:
                raise RuntimeError("simulated timeout")

        paths = [f"src/module_{index}.py" for index in range(4)]
        context = ReviewContextBuilder().build(
            changed_code={path: "def run():\n    return 1" for path in paths},
            changed_lines={path: [2] for path in paths},
            changed_line_content={path: {2: "    return 1"} for path in paths},
        )
        pipeline = ReviewPipeline(FailingProvider())

        findings = pipeline.review(context)
        assert findings == []
        assert pipeline.last_batch_count == 1
        assert pipeline.last_failed_batch_count == 1
        assert pipeline.last_reviewed_files == ()
        assert len(pipeline.last_unreviewed_files) == 4
        assert any("simulated timeout" in failure for failure in pipeline.last_review_failures)

    def test_case_mismatched_file_path_is_rejected(self) -> None:
        from src.llm.context import ReviewContextBuilder

        class WrongPathProvider:
            def review(self, prompt: str) -> str:
                return response_for(finding_data(file="SRC/MATH.PY"))

        context = ReviewContextBuilder().build(
            changed_code={"src/math.py": CODE},
            changed_lines={"src/math.py": [2]},
            changed_line_content={"src/math.py": {2: "    return a - b"}},
        )

        pipeline = ReviewPipeline(WrongPathProvider())
        findings = pipeline.review(context)

        self.assertEqual(findings, [])
        self.assertEqual(pipeline.last_failed_batch_count, 1)

    def test_pipeline_preserves_valid_finding_when_sibling_is_invalid(self) -> None:
        provider = MockProvider(
            response_for(
                finding_data(),
                finding_data(file="src/other.py"),
            )
        )
        pipeline = ReviewPipeline(provider)

        findings = pipeline.review(sample_context())

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].file, FILE)
        self.assertEqual(pipeline.last_failed_batch_count, 0)
        self.assertEqual(len(pipeline.last_rejections), 1)
        self.assertEqual(pipeline.last_rejections[0].code, "unknown_file")
        self.assertTrue(
            any(
                "rejected 1 invalid finding" in item.casefold()
                for item in pipeline.last_context_limitations
            )
        )

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
        ) :
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
        self.assertEqual(result.included_findings, 3)
        self.assertNotIn('"message":"Finding for module 0."', result.prompt)
        self.assertNotIn("changed_line_content", result.prompt)

    def test_prompt_context_lines_are_not_eligible_changed_lines(self) -> None:
        context = ReviewContextBuilder().build(
            changed_code={FILE: "one\ntwo\nthree\nfour\nfive\nsix"},
            changed_lines={FILE: [4]},
            changed_line_content={
                FILE: {line: text for line, text in enumerate(
                    ("one", "two", "three", "four", "five", "six"), start=1
                )}
            },
        )

        result = ReviewPromptBuilder().build_with_metadata(context)

        self.assertIn(
            json.dumps(
                {"file": FILE, "line": 2, "kind": "context", "source": "two"},
                separators=(",", ":"),
            ),
            result.prompt,
        )
        self.assertIn(
            json.dumps(
                {"file": FILE, "line": 4, "kind": "changed", "source": "four"},
                separators=(",", ":"),
            ),
            result.prompt,
        )
        self.assertIn(
            json.dumps(
                {"file": FILE, "line": 6, "kind": "context", "source": "six"},
                separators=(",", ":"),
            ),
            result.prompt,
        )
        self.assertEqual(result.eligible_changed_lines, {FILE: (4,)})

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
        self.assertEqual(result.available_findings, 8)
        self.assertEqual(result.supplied_findings, 2)
        self.assertEqual(result.included_findings, 2)
        self.assertNotIn('"message":"Finding 0"', result.prompt)
        self.assertIn("Return at most 3 important", result.prompt)


class RetryAndCoverageTests(unittest.TestCase):
    """Regression tests for bounded retry and coverage reporting."""

    def _context(self):
        return sample_context()

    def test_empty_evidence_rejected_by_schema(self) -> None:
        """An empty evidence string is rejected before context validation."""
        parser = ReviewResponseParser()
        bad = finding_data(evidence="")
        with self.assertRaisesRegex(ReviewResponseError, "schema validation"):
            parser.parse(response_for(bad))

    def test_valid_empty_findings_accepted(self) -> None:
        """A response with no findings is valid under the contract."""
        parser = ReviewResponseParser()
        result = parser.parse('{"findings": []}')
        self.assertEqual(result, [])

    def test_invalid_json_rejected_safely(self) -> None:
        parser = ReviewResponseParser()
        with self.assertRaisesRegex(ReviewResponseError, "valid JSON"):
            parser.parse("not json at all")

    def test_runtime_error_not_retried_as_schema_error(self) -> None:
        """Transport/connection failures (RuntimeError) bypass the retry path."""
        class ConnectionFailProvider:
            def review(self, prompt: str) -> str:
                raise RuntimeError("simulated connection failure")

        pipeline = ReviewPipeline(ConnectionFailProvider())
        findings = pipeline.review(self._context())
        self.assertEqual(findings, [])
        self.assertEqual(pipeline.last_failed_batch_count, 1)
        joined = " ".join(pipeline.last_review_failures)
        # Must NOT say "correction retry" — no retry was attempted
        self.assertNotIn("correction retry", joined.lower())
        self.assertIn("simulated connection failure", joined)

    def test_successful_retry_returns_corrected_findings(self) -> None:
        """A schema failure on the first call is corrected by the retry."""
        good_response = response_for(finding_data())
        call_count = [0]

        class TwoPhaseProvider:
            def review(self, prompt: str) -> str:
                call_count[0] += 1
                if call_count[0] == 1:
                    # First call: return empty evidence (schema failure)
                    return response_for(finding_data(evidence=""))
                # Second call (correction retry): return valid response
                return good_response

        pipeline = ReviewPipeline(TwoPhaseProvider())
        findings = pipeline.review(self._context())
        self.assertEqual(call_count[0], 2)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].evidence, "return a - b")
        self.assertEqual(pipeline.last_failed_batch_count, 0)

    def test_retry_exhaustion_reported_correctly(self) -> None:
        """If the retry also fails with a schema error, the batch is failed."""
        call_count = [0]

        class AlwaysBadProvider:
            def review(self, prompt: str) -> str:
                call_count[0] += 1
                return response_for(finding_data(evidence=""))

        pipeline = ReviewPipeline(AlwaysBadProvider())
        findings = pipeline.review(self._context())
        self.assertEqual(findings, [])
        self.assertEqual(call_count[0], 2)  # original + one retry
        self.assertEqual(pipeline.last_failed_batch_count, 1)
        joined = " ".join(pipeline.last_review_failures)
        self.assertIn("review-schema validation:", joined)

    def test_retry_does_not_duplicate_findings(self) -> None:
        """A successful retry replaces the first attempt; findings are not doubled."""
        good_response = response_for(finding_data())
        call_count = [0]

        class TwoPhaseProvider:
            def review(self, prompt: str) -> str:
                call_count[0] += 1
                if call_count[0] == 1:
                    return response_for(finding_data(evidence=""))
                return good_response

        pipeline = ReviewPipeline(TwoPhaseProvider())
        findings = pipeline.review(self._context())
        self.assertEqual(len(findings), 1)  # not 2

    def test_original_schema_error_preserved_when_retry_transport_fails(self) -> None:
        """When retry hits a transport error, the original schema-validation
        error details are reported, not the transport error."""
        call_count = [0]

        class FirstBadThenTransportFailProvider:
            def review(self, prompt: str) -> str:
                call_count[0] += 1
                if call_count[0] == 1:
                    return response_for(finding_data(evidence=""))
                raise RuntimeError("network down during retry")

        pipeline = ReviewPipeline(FirstBadThenTransportFailProvider())
        findings = pipeline.review(self._context())
        self.assertEqual(findings, [])
        self.assertEqual(pipeline.last_failed_batch_count, 1)
        joined = " ".join(pipeline.last_review_failures)
        # The original schema error is what gets reported
        self.assertIn("review-schema validation:", joined)
        # The transport error during retry does NOT replace the schema error
        self.assertNotIn("network down during retry", joined)

    def test_def004_does_not_fire_on_except_handler_body(self) -> None:
        """DEF004 must not flag code inside an except handler as unreachable."""
        import ast
        from src.analysis.diff_parser import DiffLine
        from src.analysis.rules.static_rules import run_static_ast_rules

        source = (
            "def f():\n"
            "    try:\n"
            "        x = 1\n"
            "    except ValueError:\n"
            "        raise RuntimeError('bad')\n"
            "    return x\n"
        )
        tree = ast.parse(source)
        lines = [
            DiffLine(
                number=i + 1,
                old_number=None,
                text=line,
                raw_prefix="+",
                is_added=True,
                is_deleted=False,
                is_context=False,
            )
            for i, line in enumerate(source.splitlines())
        ]
        findings = run_static_ast_rules(tree, lines, line_offset=0)
        def004_findings = [f for f in findings if f.rule_id == "DEF004"]
        self.assertEqual(def004_findings, [], msg=f"Unexpected DEF004: {def004_findings}")


if __name__ == "__main__":
    unittest.main()
