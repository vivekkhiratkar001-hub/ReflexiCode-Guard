"""Regression tests for the aggregate semantic-review scheduling budget."""

import unittest
import json

from src.llm.context import ReviewContextBuilder
from src.llm.pipeline import MIN_REQUEST_WINDOW_SECONDS, ReviewPipeline
from src.llm.prompts import ReviewPromptBuilder
from src.llm.parser import ReviewResponseParser


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TimedProvider:
    """Consumes fake time per request and records the timeout it was given."""

    model = "fake"

    def __init__(self, clock: FakeClock, timeout: int, seconds_per_request: float) -> None:
        self.clock = clock
        self.timeout = timeout
        self.seconds_per_request = seconds_per_request
        self.timeouts_seen: list[int] = []

    def review(self, prompt: str) -> str:
        self.timeouts_seen.append(self.timeout)
        elapsed = min(self.timeout, self.seconds_per_request)
        self.clock.now += elapsed
        if self.seconds_per_request > self.timeout:
            raise RuntimeError(
                f"simulated request timeout at {self.timeout} seconds"
            )
        return '{"findings":[]}'


def four_file_context():
    code, lines, content = {}, {}, {}
    for index in range(4):
        path = f"src/mod{index}.py"
        body = ["value = compute(input)", "result = value + 1"]
        code[path] = "\n".join(body)
        lines[path] = [1, 2]
        content[path] = {1: body[0], 2: body[1]}
    return ReviewContextBuilder().build(
        changed_code=code, changed_lines=lines, changed_line_content=content
    )


def make_pipeline(provider_timeout: int, seconds_per_request: float):
    clock = FakeClock()
    provider = TimedProvider(clock, provider_timeout, seconds_per_request)
    pipeline = ReviewPipeline(
        provider,
        ReviewPromptBuilder(max_files=1),
        max_requests=5,
        review_budget_seconds=240,
        clock=clock,
    )
    return pipeline, provider


class ReviewBudgetSchedulingTests(unittest.TestCase):
    def test_all_chunks_complete_when_each_fits_within_budget(self) -> None:
        pipeline, provider = make_pipeline(provider_timeout=120, seconds_per_request=30)

        pipeline.review(four_file_context())

        self.assertEqual(provider.timeouts_seen, [120, 120, 120, 120])
        self.assertEqual(pipeline.last_batch_count, 4)
        self.assertEqual(pipeline.last_failed_batch_count, 0)
        self.assertEqual(len(pipeline.last_reviewed_files), 4)
        self.assertEqual(pipeline.last_partially_reviewed_files, ())
        self.assertEqual(pipeline.last_unreviewed_files, ())
        self.assertEqual(pipeline.last_review_failures, ())
        self.assertEqual(pipeline.last_review_elapsed_seconds, 120)

    def test_request_runs_with_shortened_timeout_when_remaining_budget_is_above_minimum(
        self,
    ) -> None:
        # 85 s per request leaves 70 s after two requests: below the 120 s provider
        # timeout but above the minimum request window, so a shorter request must run.
        pipeline, provider = make_pipeline(provider_timeout=120, seconds_per_request=85)

        pipeline.review(four_file_context())

        self.assertEqual(pipeline.last_batch_count, 3)
        self.assertEqual(provider.timeouts_seen, [120, 120, 67])
        self.assertEqual(provider.timeout, 120, "provider timeout must be restored")
        self.assertEqual(len(pipeline.last_reviewed_files), 2)
        self.assertEqual(len(pipeline.last_unreviewed_files), 2)
        self.assertEqual(pipeline.last_failed_batch_count, 1)
        self.assertTrue(any("simulated request timeout" in item for item in pipeline.last_review_failures))

    def test_request_is_skipped_and_reported_below_minimum_window(self) -> None:
        # 90.5 s per request leaves 59 s after two requests, below the minimum window.
        pipeline, provider = make_pipeline(provider_timeout=120, seconds_per_request=90.5)

        pipeline.review(four_file_context())

        self.assertEqual(pipeline.last_batch_count, 2)
        self.assertEqual(len(provider.timeouts_seen), 2)
        self.assertEqual(len(pipeline.last_unreviewed_files), 2)
        self.assertTrue(
            any("minimum request window" in item for item in pipeline.last_context_limitations)
        )

    def test_provider_timeout_below_minimum_window_remains_usable(self) -> None:
        self.assertGreater(MIN_REQUEST_WINDOW_SECONDS, 30)
        pipeline, provider = make_pipeline(provider_timeout=30, seconds_per_request=100)

        pipeline.review(four_file_context())

        self.assertEqual(provider.timeouts_seen, [30, 30, 30, 30])
        self.assertEqual(pipeline.last_batch_count, 4)
        self.assertEqual(pipeline.last_failed_batch_count, 4)
        self.assertEqual(pipeline.last_reviewed_files, ())

    def test_prompt_preparation_exhaustion_prevents_model_request(self) -> None:
        clock = FakeClock()
        provider = TimedProvider(clock, timeout=120, seconds_per_request=1)
        normal_builder = ReviewPromptBuilder(max_files=1)

        class SlowBuilder:
            max_files = 1
            max_findings = normal_builder.max_findings

            def build_with_metadata(self, context):
                result = normal_builder.build_with_metadata(context)
                clock.now += 238
                return result

        pipeline = ReviewPipeline(
            provider,
            SlowBuilder(),
            max_requests=5,
            review_budget_seconds=240,
            clock=clock,
        )

        pipeline.review(four_file_context())

        self.assertEqual(provider.timeouts_seen, [])
        self.assertEqual(pipeline.last_batch_count, 0)
        self.assertEqual(pipeline.last_reviewed_files, ())
        self.assertEqual(len(pipeline.last_unreviewed_files), 4)
        self.assertTrue(any("after prompt preparation" in item for item in pipeline.last_context_limitations))

    def test_parse_overrun_keeps_validated_findings_and_stops_next_chunk(self) -> None:
        clock = FakeClock()
        class FindingProvider:
            timeout = 120

            def __init__(self) -> None:
                self.calls = 0

            def review(self, prompt: str) -> str:
                self.calls += 1
                clock.now += 1
                record = next(
                    json.loads(line)
                    for line in prompt.splitlines()
                    if line.startswith('{"file":')
                    and '"kind":"changed"' in line
                )
                return json.dumps({
                    "findings": [{
                        "file": record["file"],
                        "line": record["line"],
                        "category": "Bug",
                        "severity": "Medium",
                        "message": "The changed calculation is wrong.",
                        "evidence": record["source"].strip(),
                        "suggestion": "Correct the calculation.",
                    }]
                })

        provider = FindingProvider()
        real_parser = ReviewResponseParser()

        class SlowParser:
            def parse(self, raw_response, context):
                clock.now += 240
                return real_parser.parse(raw_response, context)

        pipeline = ReviewPipeline(
            provider,
            ReviewPromptBuilder(max_files=1),
            SlowParser(),
            max_requests=4,
            review_budget_seconds=240,
            clock=clock,
        )

        findings = pipeline.review(four_file_context())

        self.assertEqual(provider.calls, 1)
        self.assertEqual(len(findings), 1)
        self.assertEqual(pipeline.last_findings_count, 1)
        self.assertEqual(pipeline.last_batch_count, 1)
        self.assertEqual(pipeline.last_failed_batch_count, 1)
        self.assertEqual(len(pipeline.last_reviewed_files), 1)
        self.assertEqual(len(pipeline.last_unreviewed_files), 3)
        self.assertTrue(any("during response parsing" in item for item in pipeline.last_review_failures))


    def test_request_skipped_when_remaining_below_overrun_threshold(self) -> None:
        # After two 88-second requests, 64 seconds remain: above the old
        # 60-second minimum window but below the new 68-second threshold
        # (60-second window + 3-second reserve + 5-second overrun margin).
        # The third request must not be started because its deadline
        # enforcement could overrun and breach the aggregate budget.
        pipeline, provider = make_pipeline(provider_timeout=120, seconds_per_request=88)

        pipeline.review(four_file_context())

        self.assertEqual(pipeline.last_batch_count, 2)
        self.assertEqual(provider.timeouts_seen, [120, 120])
        self.assertEqual(pipeline.last_failed_batch_count, 0)
        self.assertEqual(len(pipeline.last_reviewed_files), 2)
        self.assertEqual(len(pipeline.last_unreviewed_files), 2)
        self.assertEqual(pipeline.last_review_elapsed_seconds, 176)
        self.assertLessEqual(pipeline.last_review_elapsed_seconds, 240)
        self.assertTrue(
            any("minimum request window" in item for item in pipeline.last_context_limitations)
        )

    def test_budget_exhaustion_stops_before_breaching_aggregate(self) -> None:
        # Three 60-second requests consume 180 seconds, leaving 60 seconds.
        # The old 60-second minimum window would start a fourth request with a
        # 57-second timeout, risking overrun. The new 68-second threshold
        # prevents it, keeping total elapsed within the 240-second budget.
        pipeline, provider = make_pipeline(provider_timeout=120, seconds_per_request=60)

        # Use 5 files so there are 5 chunks; request_limit caps at 5.
        pipeline.max_requests = 5
        context = ReviewContextBuilder().build(
            changed_code={f"src/mod{i}.py": "x = 1\n" for i in range(5)},
            changed_lines={f"src/mod{i}.py": [1] for i in range(5)},
            changed_line_content={f"src/mod{i}.py": {1: "x = 1"} for i in range(5)},
        )

        pipeline.review(context)

        self.assertEqual(pipeline.last_batch_count, 3)
        # Third request: remaining=120, request_timeout=min(120, 120-3)=117.
        self.assertEqual(provider.timeouts_seen, [120, 120, 117])
        self.assertEqual(pipeline.last_failed_batch_count, 0)
        self.assertEqual(len(pipeline.last_reviewed_files), 3)
        self.assertEqual(len(pipeline.last_unreviewed_files), 2)
        self.assertEqual(pipeline.last_review_elapsed_seconds, 180)
        self.assertLessEqual(pipeline.last_review_elapsed_seconds, 240)
        self.assertTrue(
            any("minimum request window" in item for item in pipeline.last_context_limitations)
        )

    def test_incomplete_coverage_honestly_reported(self) -> None:
        # Two 100-second requests consume 200 seconds, leaving 40 seconds:
        # below the 68-second threshold, so no third request is started. Two of
        # four files remain unreviewed and are reported honestly in failures.
        pipeline, provider = make_pipeline(provider_timeout=120, seconds_per_request=100)

        pipeline.review(four_file_context())

        self.assertEqual(pipeline.last_batch_count, 2)
        self.assertEqual(pipeline.last_failed_batch_count, 0)
        self.assertEqual(len(pipeline.last_reviewed_files), 2)
        self.assertEqual(len(pipeline.last_unreviewed_files), 2)
        self.assertEqual(pipeline.last_review_elapsed_seconds, 200)
        self.assertLessEqual(pipeline.last_review_elapsed_seconds, 240)
        self.assertTrue(
            any("minimum request window" in item for item in pipeline.last_context_limitations)
        )
        self.assertTrue(
            any("omitted changed files" in item for item in pipeline.last_review_failures)
        )


if __name__ == "__main__":
    unittest.main()
