"""Tests for tests/dashboard.py rendering, finding extraction, and model integration."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.github.mapper import FindingCommentMapper, format_comment_body
from src.github.models import InlineComment, ReviewSummary
from src.llm.schemas import ReviewFinding
from src.shared.models import AnalysisFinding
from tests.dashboard import (
    _check_ollama_model,
    _extract_findings,
    _finding_category,
    _finding_from_comment,
    _finding_from_unmapped,
    _render_comments,
    _render_coverage,
    _render_finding,
    _render_findings,
    _render_model_info,
    _render_summary,
    _severity_key,
)


class TestDashboardFindingExtraction(unittest.TestCase):
    """Test converting InlineComment and unmapped findings into AnalysisFinding."""

    def test_finding_from_comment_standard_body(self) -> None:
        finding = ReviewFinding(
            file="src/auth.py",
            line=10,
            category="Security",
            severity="High",
            message="Password check uses prefix matching startswith instead of exact equality.",
            evidence="return supplied_password.startswith(stored_password)",
            suggestion="return supplied_password == stored_password",
        )
        body = format_comment_body(finding, "abc123def456")
        comment = InlineComment(
            path="src/auth.py",
            line=10,
            side="RIGHT",
            body=body,
            fingerprint="abc123def456",
            finding_file="src/auth.py",
            finding_line=10,
        )

        extracted = _finding_from_comment(comment)
        self.assertEqual(extracted.file_path, "src/auth.py")
        self.assertEqual(extracted.line_number, 10)
        self.assertEqual(extracted.severity, "High")
        self.assertEqual(extracted.category, "Security")
        self.assertIn("Password check uses prefix matching", extracted.message)
        self.assertEqual(extracted.evidence, "return supplied_password.startswith(stored_password)")
        self.assertEqual(extracted.suggestion, "return supplied_password == stored_password")

    def test_finding_from_comment_no_suggestion(self) -> None:
        finding = ReviewFinding(
            file="src/tax.py",
            line=15,
            category="Bug",
            severity="Medium",
            message="Tax rate calculation error.",
            evidence="tax = amount * 0.05",
            suggestion=None,
        )
        body = format_comment_body(finding, "fp789")
        comment = InlineComment(
            path="src/tax.py",
            line=15,
            side="RIGHT",
            body=body,
            fingerprint="fp789",
            finding_file="src/tax.py",
            finding_line=15,
        )

        extracted = _finding_from_comment(comment)
        self.assertEqual(extracted.file_path, "src/tax.py")
        self.assertEqual(extracted.line_number, 15)
        self.assertEqual(extracted.severity, "Medium")
        self.assertEqual(extracted.category, "Bug")
        self.assertEqual(extracted.suggestion, "")

    def test_finding_from_raw_unstructured_comment(self) -> None:
        comment = InlineComment(
            path="src/cart.py",
            line=20,
            side="RIGHT",
            body="Just a simple warning comment without markdown header.",
            fingerprint="raw_fp",
            finding_file="src/cart.py",
            finding_line=20,
        )
        extracted = _finding_from_comment(comment)
        self.assertEqual(extracted.file_path, "src/cart.py")
        self.assertEqual(extracted.line_number, 20)
        self.assertEqual(extracted.severity, "Medium")
        self.assertEqual(extracted.message, "Just a simple warning comment without markdown header.")
        self.assertEqual(extracted.evidence, "")
        self.assertEqual(extracted.suggestion, "")

    def test_finding_from_unmapped(self) -> None:
        raw = "src/shipping.py:30 (Logic Error; missing evidence)"
        extracted = _finding_from_unmapped(raw)
        self.assertEqual(extracted.file_path, "src/shipping.py")
        self.assertEqual(extracted.line_number, 30)
        self.assertEqual(extracted.category, "Logic Error")
        self.assertIn("missing evidence", extracted.message)

    def test_extract_findings_combines_comments_and_unmapped(self) -> None:
        summary = ReviewSummary(
            dry_run=True,
            owner="omkar",
            repo="minishop",
            pr_number=1,
            head_sha="sha123",
            total_findings=2,
            intended_comments=[
                InlineComment(
                    path="src/a.py",
                    line=1,
                    side="RIGHT",
                    body="**[High] Bug** in `src/a.py:1`\n\nBug description\n\n**Evidence:** `x=1`\n\n**Suggestion:** x=2",
                    fingerprint="fp1",
                    finding_file="src/a.py",
                    finding_line=1,
                )
            ],
            unmapped_findings=["src/b.py:5 (Correctness)"],
        )
        findings = _extract_findings(summary)
        self.assertEqual(len(findings), 2)
        self.assertEqual(findings[0].file_path, "src/a.py")
        self.assertEqual(findings[0].severity, "High")
        self.assertEqual(findings[1].file_path, "src/b.py")

    def test_extract_findings_empty_summary(self) -> None:
        summary = ReviewSummary(
            dry_run=True,
            owner="omkar",
            repo="minishop",
            pr_number=1,
            head_sha="sha123",
        )
        findings = _extract_findings(summary)
        self.assertEqual(findings, [])

    def test_extract_findings_preserves_existing_findings_attr(self) -> None:
        summary = ReviewSummary(
            dry_run=True,
            owner="omkar",
            repo="minishop",
            pr_number=1,
            head_sha="sha123",
        )
        custom_finding = AnalysisFinding(
            file_path="custom.py",
            line_number=99,
            category="Static Analysis",
            severity="Low",
            message="Custom msg",
            evidence="code",
            suggestion="fix",
        )
        summary.findings = [custom_finding]  # type: ignore[attr-defined]
        findings = _extract_findings(summary)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].file_path, "custom.py")


class TestDashboardCategoryClassification(unittest.TestCase):
    def test_category_by_index_threshold(self) -> None:
        summary = ReviewSummary(
            dry_run=True,
            owner="o",
            repo="r",
            pr_number=1,
            head_sha="s",
            static_findings=2,
        )
        self.assertEqual(_finding_category(0, summary), "Static Analysis")
        self.assertEqual(_finding_category(1, summary), "Static Analysis")
        self.assertEqual(_finding_category(2, summary), "Semantic Bug")

    def test_category_by_finding_attribute(self) -> None:
        summary = ReviewSummary(
            dry_run=True,
            owner="o",
            repo="r",
            pr_number=1,
            head_sha="s",
            static_findings=0,
        )
        f_static = AnalysisFinding("f.py", 1, "static_analysis", "Low", "msg", "", "")
        f_semantic = AnalysisFinding("f.py", 2, "Security", "High", "msg", "", "")
        self.assertEqual(_finding_category(0, summary, f_static), "Static Analysis")
        self.assertEqual(_finding_category(0, summary, f_semantic), "Semantic Bug")


class TestDashboardRenderers(unittest.TestCase):
    """Test that renderers execute without attribute errors or exceptions."""

    @patch("tests.dashboard.st")
    def test_render_finding_handles_all_fields(self, mock_st: MagicMock) -> None:
        finding = AnalysisFinding(
            file_path="src/inventory.py",
            line_number=42,
            category="Bug",
            severity="Critical",
            message="Stock overwrite defect.",
            evidence="counts[item] = qty",
            suggestion="counts[item] += qty",
        )
        _render_finding(finding, "Semantic Bug")
        mock_st.markdown.assert_called()
        mock_st.text.assert_any_call("File: src/inventory.py · Line: 42")
        mock_st.text.assert_any_call("Stock overwrite defect.")
        mock_st.code.assert_any_call("counts[item] = qty")
        mock_st.code.assert_any_call("counts[item] += qty", language="diff")

    @patch("tests.dashboard.st")
    def test_render_finding_handles_missing_optional_fields(self, mock_st: MagicMock) -> None:
        finding = AnalysisFinding(
            file_path="src/unknown.py",
            line_number=None,
            category="",
            severity="",
            message="",
            evidence="",
            suggestion="",
        )
        _render_finding(finding, "Semantic Bug")
        mock_st.text.assert_any_call("File: src/unknown.py · Line: N/A")
        mock_st.text.assert_any_call("No details provided.")

    @patch("tests.dashboard.st")
    def test_render_findings_real_review_summary_no_findings_attr(self, mock_st: MagicMock) -> None:
        """Verify the exact user failure: AttributeError: 'ReviewSummary' object has no attribute 'findings'."""
        summary = ReviewSummary(
            dry_run=True,
            owner="omkark0117",
            repo="minishop-review-trial01",
            pr_number=1,
            head_sha="0a4fdea800a7544fbd6b1490827cde2f548293fb",
            total_findings=1,
            findings_by_severity={"High": 1},
            intended_comments=[
                InlineComment(
                    path="src/tax.py",
                    line=10,
                    side="RIGHT",
                    body="**[High] Bug** in `src/tax.py:10`\n\nTax calc bug.\n\n**Evidence:** `tax = amt * (1+r)`\n\n**Suggestion:** tax = amt * r",
                    fingerprint="tax_fp",
                    finding_file="src/tax.py",
                    finding_line=10,
                )
            ],
        )

        mock_cols = [MagicMock(), MagicMock()]
        mock_st.columns.return_value = mock_cols
        mock_st.multiselect.side_effect = [
            ["High", "Critical", "Medium", "Low"],
            ["Static Analysis", "Semantic Bug"],
        ]

        # This call previously raised: AttributeError: 'ReviewSummary' object has no attribute 'findings'
        _render_findings(summary)
        mock_st.caption.assert_any_call("Showing 1 of 1 findings")

    @patch("tests.dashboard.st")
    def test_render_findings_empty_summary(self, mock_st: MagicMock) -> None:
        summary = ReviewSummary(
            dry_run=True,
            owner="omkark0117",
            repo="minishop",
            pr_number=1,
            head_sha="sha000",
        )
        mock_cols = [MagicMock(), MagicMock()]
        mock_st.columns.return_value = mock_cols
        mock_cols[0].multiselect.return_value = ["High", "Critical"]
        mock_cols[1].multiselect.return_value = ["Semantic Bug"]

        _render_findings(summary)
        mock_st.caption.assert_called_with("Showing 0 of 0 findings")
        mock_st.info.assert_called_with("No findings match the selected filters.")

    @patch("tests.dashboard.st")
    def test_render_comments(self, mock_st: MagicMock) -> None:
        comments = [
            InlineComment("src/a.py", 1, "RIGHT", "comment 1", "fp1", "src/a.py", 1),
            InlineComment("src/b.py", 2, "LEFT", "comment 2", "fp2", "src/b.py", 2),
        ]
        _render_comments(comments)
        self.assertEqual(mock_st.expander.call_count, 2)

    @patch("tests.dashboard.st")
    def test_render_coverage(self, mock_st: MagicMock) -> None:
        summary = ReviewSummary(
            dry_run=True,
            owner="o",
            repo="r",
            pr_number=1,
            head_sha="h",
            reviewed_files=["src/auth.py", "src/cart.py"],
            partially_reviewed_files=["src/tax.py"],
            unreviewed_files=["src/shipping.py"],
            skipped_files=["data.bin"],
            warnings=["Context bounded"],
            semantic_batches_total=3,
            semantic_batches_failed=1,
            semantic_elapsed_seconds=12.5,
        )
        mock_cols = [MagicMock(), MagicMock(), MagicMock()]
        mock_st.columns.return_value = mock_cols
        _render_coverage(summary)
        mock_cols[0].metric.assert_called_with("Batches", 3)
        mock_cols[1].metric.assert_called_with("Failed batches", 1)
        mock_cols[2].metric.assert_called_with("Elapsed", "12.50s")

    @patch("tests.dashboard.st")
    def test_render_summary_export(self, mock_st: MagicMock) -> None:
        summary = ReviewSummary(
            dry_run=True,
            owner="o",
            repo="r",
            pr_number=1,
            head_sha="h",
        )
        _render_summary(summary)
        mock_st.markdown.assert_called()
        self.assertEqual(mock_st.download_button.call_count, 2)


class TestOllamaModelCheck(unittest.TestCase):
    """Test _check_ollama_model availability detection."""

    @patch("tests.dashboard.requests.get")
    def test_model_found_exact_name(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = {
            "models": [{"name": "qwen2.5-coder:3b"}, {"name": "qwen2.5-coder:7b"}]
        }
        mock_get.return_value.raise_for_status = MagicMock()
        ok, msg = _check_ollama_model("qwen2.5-coder:3b")
        self.assertTrue(ok)
        self.assertIn("installed", msg)

    @patch("tests.dashboard.requests.get")
    def test_model_not_found_warns(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = {
            "models": [{"name": "qwen2.5-coder:7b"}]
        }
        mock_get.return_value.raise_for_status = MagicMock()
        ok, msg = _check_ollama_model("qwen2.5-coder:3b")
        self.assertFalse(ok)
        self.assertIn("not installed", msg)
        self.assertIn("ollama pull", msg)

    @patch("tests.dashboard.requests.get")
    def test_ollama_connection_error(self, mock_get: MagicMock) -> None:
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError("refused")
        ok, msg = _check_ollama_model("qwen2.5-coder:3b")
        self.assertFalse(ok)
        self.assertIn("not reachable", msg)

    @patch("tests.dashboard.requests.get")
    def test_ollama_timeout(self, mock_get: MagicMock) -> None:
        import requests as req
        mock_get.side_effect = req.exceptions.Timeout()
        ok, msg = _check_ollama_model("qwen2.5-coder:3b")
        self.assertFalse(ok)
        self.assertIn("timed out", msg)

    @patch("tests.dashboard.requests.get")
    def test_base_name_match(self, mock_get: MagicMock) -> None:
        """A bare base name query (no tag) matches any installed tag variant."""
        mock_get.return_value.json.return_value = {
            "models": [{"name": "qwen2.5-coder:3b-q4"}]
        }
        mock_get.return_value.raise_for_status = MagicMock()
        # Bare name without tag: should match any variant
        ok, _ = _check_ollama_model("qwen2.5-coder")
        self.assertTrue(ok)


class TestRenderModelInfo(unittest.TestCase):
    """Test _render_model_info displays provider, model, and timing."""

    @patch("tests.dashboard.st")
    def test_render_model_info_shows_provider_and_model(self, mock_st: MagicMock) -> None:
        summary = ReviewSummary(
            dry_run=True,
            owner="o",
            repo="r",
            pr_number=1,
            head_sha="h",
            llm_provider="ollama",
            llm_model="qwen2.5-coder:3b",
            semantic_elapsed_seconds=63.4,
            semantic_budget_seconds=240,
        )
        _render_model_info(summary)
        mock_st.caption.assert_called_once()
        call_text = mock_st.caption.call_args[0][0]
        self.assertIn("ollama", call_text)
        self.assertIn("qwen2.5-coder:3b", call_text)
        self.assertIn("63.4", call_text)
        self.assertIn("240", call_text)

    @patch("tests.dashboard.st")
    def test_render_model_info_handles_empty_fields(self, mock_st: MagicMock) -> None:
        summary = ReviewSummary(
            dry_run=True,
            owner="o",
            repo="r",
            pr_number=1,
            head_sha="h",
        )
        _render_model_info(summary)
        mock_st.caption.assert_called_once()
        call_text = mock_st.caption.call_args[0][0]
        # Should not raise even with empty fields; should mention provider/model keys
        self.assertIn("Provider", call_text)
        self.assertIn("Model", call_text)


if __name__ == "__main__":
    unittest.main()
