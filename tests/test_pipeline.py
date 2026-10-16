import tempfile
import unittest
from pathlib import Path

from src.github import review_pr
from src.llm import OllamaReviewer
from src.shared.enums import ReviewStatus
from src.shared.models import PRContext


def sample_context() -> PRContext:
    return PRContext(
        repository_owner="team",
        repository_name="demo",
        pr_number=7,
        title="Add helper",
        description="",
        base_sha="base",
        head_sha="head",
        diff=(
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1,1 +1,2 @@\n"
            " existing()\n"
            "+result = eval(user_input)\n"
        ),
    )


class PipelineTests(unittest.TestCase):
    def test_pipeline_produces_shared_finding_contract(self) -> None:
        result = review_pr(sample_context())

        self.assertEqual(result.pr_number, 7)
        self.assertEqual(result.status, ReviewStatus.COMPLETE)
        self.assertEqual(len(result.findings), 1)
        finding = result.findings[0]
        self.assertEqual(finding.file_path, "app.py")
        self.assertEqual(finding.line_number, 2)
        self.assertEqual(finding.category, "security")

    def test_repository_rules_are_passed_to_reviewer(self) -> None:
        class RecordingReviewer:
            received_rules = None

            def review(self, context, findings, repository_rules=""):
                self.received_rules = repository_rules
                return findings

        reviewer = RecordingReviewer()
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, ".reviewrules").write_text(
                "Prefer explicit parsing.", encoding="utf-8"
            )
            result = review_pr(sample_context(), directory, reviewer)

        self.assertEqual(reviewer.received_rules, "Prefer explicit parsing.")
        self.assertEqual(result.status, ReviewStatus.COMPLETE)

    def test_ollama_failure_keeps_static_findings(self) -> None:
        class FailingReviewer:
            def review(self, context, findings, repository_rules=""):
                raise RuntimeError("Ollama is unavailable")

        result = review_pr(sample_context(), reviewer=FailingReviewer())

        self.assertEqual(result.status, ReviewStatus.PARTIAL)
        self.assertEqual(len(result.findings), 1)
        self.assertIn("Ollama is unavailable", result.summary)

    def test_ollama_findings_use_the_shared_contract(self) -> None:
        findings = OllamaReviewer._parse_findings(
            {
                "findings": [
                    {
                        "file_path": "app.py",
                        "line_number": 4,
                        "category": "semantic",
                        "severity": "medium",
                        "message": "Handle the empty input case.",
                        "evidence": "items[0]",
                        "suggestion": "Check that items is not empty first.",
                    }
                ]
            }
        )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].file_path, "app.py")
        self.assertEqual(findings[0].line_number, 4)
        self.assertEqual(findings[0].category, "semantic")

    def test_invalid_ollama_output_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "findings list"):
            OllamaReviewer._parse_findings({})


if __name__ == "__main__":
    unittest.main()
