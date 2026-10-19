"""Deterministic tests for aggregate semantic-review deadlines and coverage."""

import json
from unittest.mock import patch

import pytest

from src.llm.context import ReviewContextBuilder
from src.llm.ollama_provider import OllamaProvider
from src.llm.parser import ReviewResponseError, ReviewResponseParser
from src.llm.pipeline import ReviewPipeline
from src.llm.prompts import ReviewPromptBuilder
from src.llm.schemas import ReviewFinding


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_context(file_count: int):
    changed_code = {
        f"src/module_{index}.py": f"def run_{index}():\n    return {index}\n"
        for index in range(file_count)
    }
    return ReviewContextBuilder().build(
        changed_code=changed_code,
        changed_lines={path: [2] for path in changed_code},
        changed_line_content={
            path: {2: source.splitlines()[1]}
            for path, source in changed_code.items()
        },
    )


def test_ollama_request_timeout_is_total_and_response_is_closed() -> None:
    clock = FakeClock()
    provider = OllamaProvider(timeout=4)
    response = patch("src.llm.ollama_provider.requests.post")

    class LateStream:
        def __iter__(self):
            clock.advance(5)
            yield json.dumps(
                {"response": '{"findings":[]}', "done": True, "done_reason": "stop"}
            )

    with response as post:
        post.return_value.iter_lines.return_value = LateStream()
        with patch("src.llm.ollama_provider.monotonic", clock.monotonic):
            with pytest.raises(RuntimeError, match="timed out after 4 seconds before first output"):
                provider.review("prompt")

    assert post.call_args.kwargs["timeout"] == 4
    post.return_value.close.assert_called_once()


def test_budget_exhausted_during_prompt_preparation_starts_no_request() -> None:
    clock = FakeClock()

    class SlowPromptBuilder(ReviewPromptBuilder):
        def build_with_metadata(self, context):
            clock.advance(11)
            return super().build_with_metadata(context)

    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        def review(self, _prompt: str) -> str:
            self.calls += 1
            return '{"findings":[]}'

    provider = Provider()
    pipeline = ReviewPipeline(
        provider,
        SlowPromptBuilder(),
        review_budget_seconds=10,
    )
    with patch("src.llm.pipeline.monotonic", clock.monotonic):
        pipeline.review(make_context(1))

    assert provider.calls == 0
    assert pipeline.last_batch_count == 0
    assert pipeline.last_failed_batch_count == 1
    assert pipeline.last_reviewed_files == ()
    assert pipeline.last_unreviewed_files == ("src/module_0.py",)
    assert any("during prompt preparation" in failure for failure in pipeline.last_review_failures)


def test_budget_exhausted_during_model_execution_is_reported_and_timeout_restored() -> None:
    clock = FakeClock()

    class SlowProvider:
        timeout = 100

        def review(self, _prompt: str) -> str:
            assert self.timeout == 11
            clock.advance(17)
            return '{"findings":[]}'

    provider = SlowProvider()
    pipeline = ReviewPipeline(provider, review_budget_seconds=16)
    with patch("src.llm.pipeline.monotonic", clock.monotonic):
        pipeline.review(make_context(1))

    assert provider.timeout == 100
    assert pipeline.last_batch_count == 1
    assert pipeline.last_failed_batch_count == 1
    assert pipeline.last_reviewed_files == ()
    assert pipeline.last_unreviewed_files == ("src/module_0.py",)
    assert any("during model execution" in failure for failure in pipeline.last_review_failures)
    assert pipeline.last_elapsed_seconds == 17


def test_provider_timeout_message_is_preserved_and_configured_timeout_restored() -> None:
    clock = FakeClock()

    class TimedOutProvider:
        timeout = 100

        def review(self, _prompt: str) -> str:
            assert self.timeout == 11
            raise RuntimeError(
                "The Ollama request timed out after 11 seconds before first output."
            )

    provider = TimedOutProvider()
    pipeline = ReviewPipeline(provider, review_budget_seconds=16)
    with patch("src.llm.pipeline.monotonic", clock.monotonic):
        pipeline.review(make_context(1))

    assert provider.timeout == 100
    assert pipeline.last_batch_count == 1
    assert pipeline.last_failed_batch_count == 1
    assert any(
        "timed out after 11 seconds before first output"
        in failure
        for failure in pipeline.last_review_failures
    )


def test_budget_exhausted_during_parsing_retains_findings_without_claiming_coverage() -> None:
    clock = FakeClock()
    parsed_finding = ReviewFinding.model_validate(
        {
            "file": "src/module_0.py",
            "line": 2,
            "category": "Bug",
            "severity": "High",
            "message": "The function returns the wrong value.",
            "evidence": "return 0",
            "suggestion": None,
        }
    )

    class SlowParser:
        def parse(self, _raw_response: str, _context):
            clock.advance(10.1)
            return [parsed_finding]

    class Provider:
        def review(self, _prompt: str) -> str:
            return '{"findings":[]}'

    pipeline = ReviewPipeline(
        Provider(),
        parser=SlowParser(),
        review_budget_seconds=10,
    )
    with patch("src.llm.pipeline.monotonic", clock.monotonic):
        findings = pipeline.review(make_context(1))

    assert findings == [parsed_finding]
    assert pipeline.last_findings_count == 1
    assert pipeline.last_failed_batch_count == 1
    assert pipeline.last_reviewed_files == ()
    assert pipeline.last_unreviewed_files == ("src/module_0.py",)
    assert any("during response parsing" in failure for failure in pipeline.last_review_failures)


def test_request_timeouts_keep_safety_margin_and_no_request_starts_without_budget() -> None:
    clock = FakeClock()

    class TimedProvider:
        timeout = 100

        def __init__(self) -> None:
            self.timeouts: list[float] = []

        def review(self, _prompt: str) -> str:
            self.timeouts.append(self.timeout)
            clock.advance(2.5)
            return '{"findings":[]}'

    provider = TimedProvider()
    pipeline = ReviewPipeline(
        provider,
        ReviewPromptBuilder(max_files=1),
        review_budget_seconds=8,
    )
    with patch("src.llm.pipeline.monotonic", clock.monotonic):
        pipeline.review(make_context(4))

    assert provider.timeouts == [3]
    assert provider.timeout == 100
    assert pipeline.last_batch_count == 1
    assert pipeline.last_failed_batch_count == 0
    assert len(pipeline.last_reviewed_files) == 1
    assert len(pipeline.last_unreviewed_files) == 3
    assert any("no additional model request was started" in failure for failure in pipeline.last_review_failures)


def test_all_batches_complete_when_they_fit_within_the_aggregate_budget() -> None:
    clock = FakeClock()

    class FastProvider:
        def review(self, _prompt: str) -> str:
            clock.advance(2)
            return '{"findings":[]}'

    pipeline = ReviewPipeline(
        FastProvider(),
        ReviewPromptBuilder(max_files=2),
        review_budget_seconds=10,
    )
    with patch("src.llm.pipeline.monotonic", clock.monotonic):
        findings = pipeline.review(make_context(4))

    assert findings == []
    assert pipeline.last_batch_count == 2
    assert pipeline.last_failed_batch_count == 0
    assert len(pipeline.last_reviewed_files) == 4
    assert pipeline.last_partially_reviewed_files == ()
    assert pipeline.last_unreviewed_files == ()
    assert pipeline.last_elapsed_seconds == 4


def test_a_file_with_only_some_changed_lines_in_the_prompt_is_partial() -> None:
    path = "src/large_module.py"
    changed_code = "\n".join(
        f"value_{line} = {line}  # preserve changed line {line}"
        for line in range(1, 121)
    )
    changed_lines = list(range(1, 121))
    context = ReviewContextBuilder().build(
        changed_code={path: changed_code},
        changed_lines={path: changed_lines},
        changed_line_content={
            path: {
                line: text
                for line, text in enumerate(changed_code.splitlines(), start=1)
            }
        },
    )

    class EmptyProvider:
        def review(self, _prompt: str) -> str:
            return '{"findings":[]}'

    pipeline = ReviewPipeline(EmptyProvider())

    pipeline.review(context)

    assert pipeline.last_reviewed_files == ()
    assert pipeline.last_partially_reviewed_files == (path,)
    assert pipeline.last_unreviewed_files == ()
    assert any("partially covered changed lines" in failure for failure in pipeline.last_review_failures)


@pytest.mark.parametrize(
    ("path", "line", "source", "unsupported_evidence"),
    [
        (
            "src/llm/parser.py",
            27,
            'if text.count("{") > text.count("}") or text.count("[") > text.count("]"):',
            'if text.count("{") > text.count(")") or text.count("[") > text.count("]"):',
        ),
        (
            "src/github/cli.py",
            35,
            "    ):",
            'parser.add_argument(\n    "--ollama-model",\n    help="Installed local Ollama model (default qwen2.5-coder:7b)",\n):',
        ),
        (
            "README.md",
            68,
            "`REFLEXICODE_OLLAMA_MODEL`; the application does not download or silently switch",
            "models. The default timeout is 120 seconds (configurable from 1 to 300), output",
        ),
    ],
)
def test_actual_pr_model_evidence_mismatches_remain_rejected(
    path: str,
    line: int,
    source: str,
    unsupported_evidence: str,
) -> None:
    context = ReviewContextBuilder().build(
        changed_code={path: source},
        changed_lines={path: [line]},
        changed_line_content={path: {line: source}},
    )
    raw_response = json.dumps(
        {
            "findings": [
                {
                    "file": path,
                    "line": line,
                    "category": "Security",
                    "severity": "High",
                    "message": "Captured unsupported model claim.",
                    "evidence": unsupported_evidence,
                    "suggestion": "Change the code.",
                }
            ]
        }
    )

    with pytest.raises(ReviewResponseError, match="does not match the changed line"):
        ReviewResponseParser().parse(raw_response, context)
