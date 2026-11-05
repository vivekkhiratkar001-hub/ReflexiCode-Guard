"""Tests for MiniShop semantic review and suggestion validation end-to-end.

Covers the five MiniShop regression bugs:
- Bug A (auth.py): exact equality vs prefix match
- Bug B (cart.py): percentage discount vs flat subtraction
- Bug C (inventory.py): restock accumulation vs overwrite
- Bug D (shipping.py): express surcharge vs division or doubling
- Bug E (tax.py): tax amount only vs tax-inclusive calculation
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.github.adapters import SharedContractReviewer, build_changed_line_content
from src.github.models import FileChange, ParsedPatch, PullRequestInfo
from src.github.orchestrator import ReviewOrchestrator
from src.llm.context import ReviewContext, ReviewContextBuilder
from src.llm.mock_provider import MockProvider
from src.llm.parser import ReviewResponseError, ReviewResponseParser, validate_suggestion
from src.llm.pipeline import ReviewPipeline
from src.llm.schemas import ReviewFinding


MINISHOP_REQUIREMENTS = [
    {
        "instructions": (
            "- Discounts are percentages: a 10% discount on 200 leaves 180.\n"
            "- Restocking adds new units to the current inventory count.\n"
            "- Password verification must require an exact match in this toy example.\n"
            "- Express shipping should add a surcharge; it must not reduce the standard fee.\n"
            "- calculate_tax(amount, rate) returns the tax amount only, not the total including tax.\n"
            "- Order transitions must follow the transition table in src/orders.py."
        )
    }
]

MINISHOP_FILES = {
    "src/auth.py": (
        '"""Toy password verification for review testing."""\n\n'
        "def verify_password(stored_password, supplied_password):\n"
        "    return supplied_password.startswith(stored_password)\n"
    ),
    "src/cart.py": (
        "def subtotal(items):\n"
        "    return sum(item['price'] * item['qty'] for item in items)\n\n\n"
        "def apply_discount(total, discount_percent):\n"
        "    if not 0 <= discount_percent <= 100:\n"
        '        raise ValueError("discount_percent must be between 0 and 100")\n\n'
        "    return total - discount_percent\n"
    ),
    "src/inventory.py": (
        "def restock(stock, sku, quantity):\n"
        "    if quantity < 0:\n"
        '        raise ValueError("quantity cannot be negative")\n\n'
        "    stock[sku] = quantity\n"
    ),
    "src/shipping.py": (
        "def shipping_fee(weight_kg, express=False):\n"
        "    fee = 50 if weight_kg <= 2 else 100\n\n"
        "    if express:\n"
        "        fee = fee / 2\n\n"
        "    return fee\n"
    ),
    "src/tax.py": (
        "def calculate_tax(amount, rate):\n"
        "    if not 0 <= rate <= 1:\n"
        '        raise ValueError("rate must be between 0 and 1")\n\n'
        "    return amount * (1 + rate)\n"
    ),
}

MINISHOP_CHANGED_LINES = {
    "src/auth.py": [4],
    "src/cart.py": [10],
    "src/inventory.py": [6],
    "src/shipping.py": [6],
    "src/tax.py": [6],
}

MINISHOP_CHANGED_LINE_CONTENT = {
    "src/auth.py": {4: "    return supplied_password.startswith(stored_password)"},
    "src/cart.py": {10: "    return total - discount_percent"},
    "src/inventory.py": {6: "    stock[sku] = quantity"},
    "src/shipping.py": {6: "        fee = fee / 2"},
    "src/tax.py": {6: "    return amount * (1 + rate)"},
}


def build_minishop_context() -> ReviewContext:
    return ReviewContextBuilder().build(
        pr_title="Introduce semantic bugs for PR review testing",
        pr_description="Trial PR introducing known bugs for semantic testing.",
        changed_code=MINISHOP_FILES,
        repository_rules=MINISHOP_REQUIREMENTS,
        changed_lines=MINISHOP_CHANGED_LINES,
        changed_line_content=MINISHOP_CHANGED_LINE_CONTENT,
    )


# ---------------------------------------------------------------------------
# Layer 1 & 2: Suggestion Validation Tests
# ---------------------------------------------------------------------------

class TestMiniShopSuggestionValidation:
    """Validate that invalid suggestions are detected and rejected or downgraded."""

    def test_bug_a_auth_exact_match_accepted(self) -> None:
        context = build_minishop_context()
        finding = ReviewFinding(
            file="src/auth.py",
            line=4,
            category="Security",
            severity="Critical",
            message="Password comparison uses prefix matching instead of exact equality.",
            evidence="return supplied_password.startswith(stored_password)",
            suggestion="return supplied_password == stored_password",
        )
        assert validate_suggestion(finding, context, strict=True) is None

    def test_bug_a_auth_prefix_match_suggestion_rejected(self) -> None:
        context = build_minishop_context()
        finding = ReviewFinding(
            file="src/auth.py",
            line=4,
            category="Security",
            severity="Critical",
            message="Prefix matching allows auth bypass.",
            evidence="return supplied_password.startswith(stored_password)",
            suggestion="return supplied_password.startswith(stored_password)",
        )
        with pytest.raises(ReviewResponseError, match="repeats the buggy code"):
            validate_suggestion(finding, context, strict=True)

    def test_bug_b_cart_percentage_discount_accepted(self) -> None:
        context = build_minishop_context()
        finding = ReviewFinding(
            file="src/cart.py",
            line=10,
            category="Bug",
            severity="High",
            message="Discount percentage is subtracted as a flat amount rather than a percentage.",
            evidence="return total - discount_percent",
            suggestion="return total * (1 - discount_percent / 100)",
        )
        assert validate_suggestion(finding, context, strict=True) is None

    def test_bug_b_cart_repeated_flat_discount_rejected(self) -> None:
        context = build_minishop_context()
        finding = ReviewFinding(
            file="src/cart.py",
            line=10,
            category="Bug",
            severity="High",
            message="Flat subtraction is incorrect.",
            evidence="return total - discount_percent",
            suggestion="return total - discount_percent",
        )
        with pytest.raises(ReviewResponseError, match="repeats the buggy code"):
            validate_suggestion(finding, context, strict=True)

    def test_bug_c_inventory_accumulation_accepted(self) -> None:
        context = build_minishop_context()
        finding = ReviewFinding(
            file="src/inventory.py",
            line=6,
            category="Bug",
            severity="High",
            message="Restocking overwrites existing inventory instead of accumulating it.",
            evidence="stock[sku] = quantity",
            suggestion="stock[sku] = stock.get(sku, 0) + quantity",
        )
        assert validate_suggestion(finding, context, strict=True) is None

    def test_bug_c_inventory_repeated_overwrite_rejected(self) -> None:
        context = build_minishop_context()
        finding = ReviewFinding(
            file="src/inventory.py",
            line=6,
            category="Bug",
            severity="High",
            message="Overwriting stock loses previous inventory.",
            evidence="stock[sku] = quantity",
            suggestion="stock[sku] = quantity",
        )
        with pytest.raises(ReviewResponseError, match="repeats the buggy code"):
            validate_suggestion(finding, context, strict=True)

    def test_bug_d_shipping_surcharge_accepted(self) -> None:
        context = build_minishop_context()
        finding = ReviewFinding(
            file="src/shipping.py",
            line=6,
            category="Bug",
            severity="High",
            message="Express shipping halves the fee instead of adding a surcharge.",
            evidence="fee = fee / 2",
            suggestion="fee += 50",
        )
        assert validate_suggestion(finding, context, strict=True) is None

    def test_bug_d_shipping_repeated_division_rejected(self) -> None:
        context = build_minishop_context()
        finding = ReviewFinding(
            file="src/shipping.py",
            line=6,
            category="Bug",
            severity="High",
            message="Express fee division is wrong.",
            evidence="fee = fee / 2",
            suggestion="fee = fee / 2",
        )
        with pytest.raises(ReviewResponseError, match="repeats the buggy code"):
            validate_suggestion(finding, context, strict=True)

    def test_bug_d_shipping_fee_doubling_violates_requirement(self) -> None:
        context = build_minishop_context()
        finding = ReviewFinding(
            file="src/shipping.py",
            line=6,
            category="Bug",
            severity="High",
            message="Express fee should add surcharge.",
            evidence="fee = fee / 2",
            suggestion="fee = fee * 2",
        )
        with pytest.raises(ReviewResponseError, match="conflicts with repository requirements"):
            validate_suggestion(finding, context, strict=True)

    def test_bug_e_tax_amount_only_accepted(self) -> None:
        context = build_minishop_context()
        finding = ReviewFinding(
            file="src/tax.py",
            line=6,
            category="Bug",
            severity="High",
            message="calculate_tax returns the total including tax instead of the tax amount only.",
            evidence="return amount * (1 + rate)",
            suggestion="return amount * rate",
        )
        assert validate_suggestion(finding, context, strict=True) is None

    def test_bug_e_tax_repeated_inclusive_expression_rejected(self) -> None:
        context = build_minishop_context()
        finding = ReviewFinding(
            file="src/tax.py",
            line=6,
            category="Bug",
            severity="High",
            message="Function should return tax amount only.",
            evidence="return amount * (1 + rate)",
            suggestion="return amount * (1 + rate)",
        )
        with pytest.raises(ReviewResponseError, match="repeats the buggy code"):
            validate_suggestion(finding, context, strict=True)

    def test_tax_repeated_expression_with_formatting_variation_rejected(self) -> None:
        context = build_minishop_context()
        finding = ReviewFinding(
            file="src/tax.py",
            line=6,
            category="Bug",
            severity="High",
            message="Function should return tax amount only.",
            evidence="return amount * (1 + rate)",
            suggestion="```python\nreturn (amount * (1 + rate))  # tax total\n```",
        )
        with pytest.raises(ReviewResponseError, match="repeats the buggy code"):
            validate_suggestion(finding, context, strict=True)


# ---------------------------------------------------------------------------
# Layer 3: Safe Failure and Production Downgrade Behavior
# ---------------------------------------------------------------------------

class TestMiniShopSafeFailureBehavior:
    """Test that invalid suggestions do not crash the pipeline and are safely downgraded."""

    def test_parser_downgrades_repeated_suggestion_to_none(self) -> None:
        context = build_minishop_context()
        raw_response = json.dumps(
            {
                "findings": [
                    {
                        "file": "src/tax.py",
                        "line": 6,
                        "category": "Bug",
                        "severity": "High",
                        "message": "calculate_tax returns total including tax instead of tax amount.",
                        "evidence": "return amount * (1 + rate)",
                        "suggestion": "return amount * (1 + rate)",
                    }
                ]
            }
        )
        parser = ReviewResponseParser()
        findings = parser.parse(raw_response, context)
        assert len(findings) == 1
        assert findings[0].file == "src/tax.py"
        assert findings[0].line == 6
        assert findings[0].evidence == "return amount * (1 + rate)"
        # Suggestion was rejected and safely downgraded to None
        assert findings[0].suggestion is None

    def test_parser_downgrades_conflicting_shipping_suggestion_to_none(self) -> None:
        context = build_minishop_context()
        raw_response = json.dumps(
            {
                "findings": [
                    {
                        "file": "src/shipping.py",
                        "line": 6,
                        "category": "Bug",
                        "severity": "High",
                        "message": "Express shipping fee division is incorrect.",
                        "evidence": "fee = fee / 2",
                        "suggestion": "fee = fee * 2",
                    }
                ]
            }
        )
        parser = ReviewResponseParser()
        findings = parser.parse(raw_response, context)
        assert len(findings) == 1
        assert findings[0].suggestion is None

    def test_pipeline_preserves_all_findings_even_when_suggestions_are_invalid(self) -> None:
        """When the model suggests invalid fixes for tax and shipping, findings are not dropped."""
        context = build_minishop_context()
        raw_response = json.dumps(
            {
                "findings": [
                    {
                        "file": "src/auth.py",
                        "line": 4,
                        "category": "Security",
                        "severity": "Critical",
                        "message": "Prefix matching allows authentication bypass.",
                        "evidence": "return supplied_password.startswith(stored_password)",
                        "suggestion": "return supplied_password == stored_password",
                    },
                    {
                        "file": "src/tax.py",
                        "line": 6,
                        "category": "Bug",
                        "severity": "High",
                        "message": "Returns tax-inclusive total.",
                        "evidence": "return amount * (1 + rate)",
                        "suggestion": "return amount * (1 + rate)",  # Invalid repeat
                    },
                    {
                        "file": "src/shipping.py",
                        "line": 6,
                        "category": "Bug",
                        "severity": "High",
                        "message": "Express fee reduced.",
                        "evidence": "fee = fee / 2",
                        "suggestion": "fee = fee * 2",  # Invalid doubling
                    },
                ]
            }
        )
        pipeline = ReviewPipeline(
            MockProvider(raw_response),
            max_requests=3,
            review_budget_seconds=240,
        )
        results = pipeline.review(context)
        assert len(results) == 3
        # auth.py valid suggestion preserved
        auth_f = next(f for f in results if f.file == "src/auth.py")
        assert auth_f.suggestion == "return supplied_password == stored_password"
        # tax.py invalid suggestion safely stripped
        tax_f = next(f for f in results if f.file == "src/tax.py")
        assert tax_f.suggestion is None
        # shipping.py invalid suggestion safely stripped
        ship_f = next(f for f in results if f.file == "src/shipping.py")
        assert ship_f.suggestion is None
        # Batch was successful, not failed
        assert pipeline.last_failed_batch_count == 0

    def test_end_to_end_orchestrator_minishop_all_five_bugs(self) -> None:
        """Test complete review flow for all five MiniShop bugs through ReviewOrchestrator."""
        all_five_response = json.dumps(
            {
                "findings": [
                    {
                        "file": "src/auth.py",
                        "line": 4,
                        "category": "Security",
                        "severity": "Critical",
                        "message": "Prefix matching allows password bypass.",
                        "evidence": "return supplied_password.startswith(stored_password)",
                        "suggestion": "return supplied_password == stored_password",
                    },
                    {
                        "file": "src/cart.py",
                        "line": 10,
                        "category": "Bug",
                        "severity": "High",
                        "message": "Flat subtraction instead of percentage calculation.",
                        "evidence": "return total - discount_percent",
                        "suggestion": "return total * (1 - discount_percent / 100)",
                    },
                    {
                        "file": "src/inventory.py",
                        "line": 6,
                        "category": "Bug",
                        "severity": "High",
                        "message": "Restock overwrites existing stock.",
                        "evidence": "stock[sku] = quantity",
                        "suggestion": "stock[sku] = stock.get(sku, 0) + quantity",
                    },
                ]
            }
        )
        pipeline = ReviewPipeline(
            MockProvider(all_five_response),
            max_requests=3,
            review_budget_seconds=240,
        )
        context = build_minishop_context()
        results = pipeline.review(context)
        assert len(results) == 3
        for finding in results:
            assert finding.suggestion is not None
            assert finding.suggestion != finding.evidence

    def test_minishop_diffs_produce_zero_unreachable_code_static_findings(self) -> None:
        """Diffs replacing return statements must not trigger DEF004 false positives."""
        from src.analysis.analyzer import analyze_diff

        auth_diff = (
            "--- a/src/auth.py\n"
            "+++ b/src/auth.py\n"
            "@@ -1,4 +1,4 @@\n"
            ' """Toy password verification for review testing."""\n\n'
            " def verify_password(stored_password, supplied_password):\n"
            "-    return supplied_password == stored_password\n"
            "+    return supplied_password.startswith(stored_password)\n"
        )
        cart_diff = (
            "--- a/src/cart.py\n"
            "+++ b/src/cart.py\n"
            "@@ -7,4 +7,4 @@\n"
            "     if not 0 <= discount_percent <= 100:\n"
            '         raise ValueError("discount_percent must be between 0 and 100")\n\n'
            "-    return total * (1 - discount_percent / 100)\n"
            "+    return total - discount_percent\n"
        )
        tax_diff = (
            "--- a/src/tax.py\n"
            "+++ b/src/tax.py\n"
            "@@ -7,4 +7,4 @@\n"
            "     if not 0 <= rate <= 1:\n"
            '         raise ValueError("rate must be between 0 and 1")\n\n'
            "-    return amount * rate\n"
            "+    return amount * (1 + rate)\n"
        )
        combined_diff = auth_diff + cart_diff + tax_diff
        findings = analyze_diff(combined_diff)
        unreachable = [f for f in findings if "unreachable" in f.message.lower()]
        assert unreachable == []

    def test_orchestrator_minishop_all_five_bugs_end_to_end_orchestration(self) -> None:
        """Complete orchestration for all 5 MiniShop bugs with GitHub client and reviewer."""
        from src.github.client import GitHubClient
        from src.github.orchestrator import ReviewOrchestrator
        from unittest.mock import Mock

        all_five_json = json.dumps(
            {
                "findings": [
                    {
                        "file": "src/auth.py",
                        "line": 4,
                        "category": "Security",
                        "severity": "Critical",
                        "message": "Prefix matching allows password bypass.",
                        "evidence": "return supplied_password.startswith(stored_password)",
                        "suggestion": "return supplied_password == stored_password",
                    },
                    {
                        "file": "src/cart.py",
                        "line": 10,
                        "category": "Bug",
                        "severity": "High",
                        "message": "Flat subtraction instead of percentage calculation.",
                        "evidence": "return total - discount_percent",
                        "suggestion": "return total * (1 - discount_percent / 100)",
                    },
                    {
                        "file": "src/inventory.py",
                        "line": 6,
                        "category": "Bug",
                        "severity": "High",
                        "message": "Restock overwrites existing stock.",
                        "evidence": "stock[sku] = quantity",
                        "suggestion": "stock[sku] = stock.get(sku, 0) + quantity",
                    },
                    {
                        "file": "src/shipping.py",
                        "line": 6,
                        "category": "Bug",
                        "severity": "High",
                        "message": "Express shipping halves fee instead of adding surcharge.",
                        "evidence": "fee = fee / 2",
                        "suggestion": "fee = fee + 50",
                    },
                    {
                        "file": "src/tax.py",
                        "line": 6,
                        "category": "Bug",
                        "severity": "Medium",
                        "message": "Tax calculation returns inclusive total instead of tax amount.",
                        "evidence": "return amount * (1 + rate)",
                        "suggestion": "return amount * rate",
                    },
                ]
            }
        )

        files = [
            {
                "filename": "src/auth.py",
                "status": "modified",
                "patch": "@@ -1,4 +1,4 @@\n \"\"\"Toy password verification for review testing.\"\"\"\n \n def verify_password(stored_password, supplied_password):\n-    return supplied_password == stored_password\n+    return supplied_password.startswith(stored_password)\n",
                "additions": 1,
                "deletions": 1,
                "changes": 2,
            },
            {
                "filename": "src/cart.py",
                "status": "modified",
                "patch": "@@ -7,4 +7,4 @@\n     if not 0 <= discount_percent <= 100:\n         raise ValueError(\"discount_percent must be between 0 and 100\")\n \n-    return total * (1 - discount_percent / 100)\n+    return total - discount_percent\n",
                "additions": 1,
                "deletions": 1,
                "changes": 2,
            },
            {
                "filename": "src/inventory.py",
                "status": "modified",
                "patch": "@@ -3,4 +3,4 @@\n     if quantity < 0:\n         raise ValueError(\"quantity cannot be negative\")\n \n-    stock[sku] = stock.get(sku, 0) + quantity\n+    stock[sku] = quantity\n",
                "additions": 1,
                "deletions": 1,
                "changes": 2,
            },
            {
                "filename": "src/shipping.py",
                "status": "modified",
                "patch": "@@ -3,4 +3,4 @@\n     fee = 50 if weight_kg <= 2 else 100\n \n     if express:\n-        fee = fee + 50\n+        fee = fee / 2\n",
                "additions": 1,
                "deletions": 1,
                "changes": 2,
            },
            {
                "filename": "src/tax.py",
                "status": "modified",
                "patch": "@@ -3,4 +3,4 @@\n     if not 0 <= rate <= 1:\n         raise ValueError(\"rate must be between 0 and 1\")\n \n-    return amount * rate\n+    return amount * (1 + rate)\n",
                "additions": 1,
                "deletions": 1,
                "changes": 2,
            },
        ]

        def _mock_request(method, url, **kwargs):
            resp = Mock()
            resp.status_code = 200
            resp.headers = {}
            if "pulls/1/files" in url:
                resp.json.return_value = files
                resp.links = {}
            elif "pulls/1/comments" in url:
                resp.json.return_value = []
                resp.links = {}
            elif "pulls/1" in url:
                resp.json.return_value = {
                    "number": 1,
                    "title": "MiniShop Buggy PR",
                    "body": "Trial PR with 5 intentional bugs",
                    "head": {"sha": "0a4fdea800a7544fbd6b1490827cde2f548293fb"},
                    "base": {"ref": "main"},
                }
            elif "contents" in url:
                resp.status_code = 404
                resp.json.return_value = {"message": "Not Found"}
            return resp

        session = Mock()
        session.headers = {}
        session.request.side_effect = _mock_request
        client = GitHubClient(token="mock-token", session=session, sleeper=lambda _: None)

        from src.llm.prompts import ReviewPromptBuilder
        prompt_builder = ReviewPromptBuilder(max_files=5, max_findings=5)
        pipeline = ReviewPipeline(
            MockProvider(all_five_json),
            prompt_builder=prompt_builder,
            parser=ReviewResponseParser(max_findings=5),
            max_requests=5,
            review_budget_seconds=240,
        )

        orchestrator = ReviewOrchestrator(
            client,
            pipeline,
            dry_run=True,
        )

        summary = orchestrator.run("omkark0117", "minishop-review-trial01", 1)

        assert summary.total_findings == 5
        assert summary.comments_intended == 5
        assert summary.semantic_batches_failed == 0
        assert len(summary.intended_comments) == 5

        comment_files = {c.path for c in summary.intended_comments}
        assert comment_files == {
            "src/auth.py",
            "src/cart.py",
            "src/inventory.py",
            "src/shipping.py",
            "src/tax.py",
        }
        for comment in summary.intended_comments:
            assert "suggestion" in comment.body.lower()

    def test_production_path_rejects_incorrect_suggestions_end_to_end(self) -> None:
        """Requirement 5 & 7: Send deliberately incorrect suggestions through the real production path

        and verify they are completely withheld from comments, markdown, and dashboard rendering.
        """
        from src.github.client import GitHubClient
        from src.github.orchestrator import ReviewOrchestrator
        from src.llm.prompts import ReviewPromptBuilder
        from tests.dashboard import _extract_findings
        from unittest.mock import Mock

        # Model outputs:
        # 1. Tax calculation: incorrect subtraction (amount * (1 - rate))
        # 2. Shipping calculation: doubling fee (fee = fee * 2) instead of adding surcharge
        # 3. Cart calculation: repeats the buggy code (return total - discount_percent)
        # 4. Auth: invalid Python syntax
        incorrect_json = json.dumps(
            {
                "findings": [
                    {
                        "file": "src/tax.py",
                        "line": 6,
                        "category": "Bug",
                        "severity": "High",
                        "message": "Incorrect tax calculation.",
                        "evidence": "return amount * (1 + rate)",
                        "suggestion": "return amount * (1 - rate)",
                    },
                    {
                        "file": "src/shipping.py",
                        "line": 6,
                        "category": "Bug",
                        "severity": "High",
                        "message": "Express fee doubling is wrong.",
                        "evidence": "fee = fee / 2",
                        "suggestion": "fee = fee * 2",
                    },
                    {
                        "file": "src/cart.py",
                        "line": 10,
                        "category": "Bug",
                        "severity": "High",
                        "message": "Flat subtraction is wrong.",
                        "evidence": "return total - discount_percent",
                        "suggestion": "return total - discount_percent",
                    },
                    {
                        "file": "src/auth.py",
                        "line": 4,
                        "category": "Bug",
                        "severity": "High",
                        "message": "Password comparison defect.",
                        "evidence": "return supplied_password.startswith(stored_password)",
                        "suggestion": "return supplied_password ==",
                    },
                ]
            }
        )

        files = [
            {
                "filename": "src/tax.py",
                "status": "modified",
                "patch": "@@ -3,4 +3,4 @@\n     if not 0 <= rate <= 1:\n         raise ValueError(\"rate must be between 0 and 1\")\n \n-    return amount * rate\n+    return amount * (1 + rate)\n",
                "additions": 1,
                "deletions": 1,
                "changes": 2,
            },
            {
                "filename": "src/shipping.py",
                "status": "modified",
                "patch": "@@ -3,4 +3,4 @@\n     fee = 50 if weight_kg <= 2 else 100\n \n     if express:\n-        fee = fee + 50\n+        fee = fee / 2\n",
                "additions": 1,
                "deletions": 1,
                "changes": 2,
            },
            {
                "filename": "src/cart.py",
                "status": "modified",
                "patch": "@@ -7,4 +7,4 @@\n     if not 0 <= discount_percent <= 100:\n         raise ValueError(\"discount_percent must be between 0 and 100\")\n \n-    return total * (1 - discount_percent / 100)\n+    return total - discount_percent\n",
                "additions": 1,
                "deletions": 1,
                "changes": 2,
            },
            {
                "filename": "src/auth.py",
                "status": "modified",
                "patch": "@@ -1,4 +1,4 @@\n \"\"\"Toy password verification for review testing.\"\"\"\n \n def verify_password(stored_password, supplied_password):\n-    return supplied_password == stored_password\n+    return supplied_password.startswith(stored_password)\n",
                "additions": 1,
                "deletions": 1,
                "changes": 2,
            },
        ]

        def _mock_req(method, url, **kwargs):
            resp = Mock()
            resp.status_code = 200
            resp.headers = {}
            if "pulls/1/files" in url:
                resp.json.return_value = files
                resp.links = {}
            elif "pulls/1/comments" in url:
                resp.json.return_value = []
                resp.links = {}
            elif "pulls/1" in url:
                resp.json.return_value = {
                    "number": 1,
                    "title": "MiniShop PR",
                    "body": "PR description",
                    "head": {"sha": "headsha"},
                    "base": {"ref": "main"},
                }
            elif "contents" in url:
                resp.status_code = 404
                resp.json.return_value = {}
            return resp

        session = Mock()
        session.headers = {}
        session.request.side_effect = _mock_req
        client = GitHubClient(token="mock", session=session, sleeper=lambda _: None)

        # Supply repository rules to the reviewer
        class MockRulesProvider:
            def get_rules(self, pr):
                return MINISHOP_REQUIREMENTS

        prompt_builder = ReviewPromptBuilder(max_files=5, max_findings=5)
        pipeline = ReviewPipeline(
            MockProvider(incorrect_json),
            prompt_builder=prompt_builder,
            parser=ReviewResponseParser(max_findings=5),
            max_requests=5,
            review_budget_seconds=240,
        )

        orchestrator = ReviewOrchestrator(
            client,
            pipeline,
            rules_provider=MockRulesProvider(),
            dry_run=True,
        )

        summary = orchestrator.run("omkark0117", "minishop-review-trial01", 1)

        # All 4 findings are retained
        assert summary.total_findings == 4
        assert len(summary.intended_comments) == 4

        # Crucial check: none of the comments contain the incorrect suggestions!
        for comment in summary.intended_comments:
            assert "**Suggestion:** No suggestion provided." in comment.body
            assert "**Suggestion:** return amount * (1 - rate)" not in comment.body
            assert "**Suggestion:** fee = fee * 2" not in comment.body
            assert "**Suggestion:** return total - discount_percent" not in comment.body
            assert "**Suggestion:** return supplied_password ==" not in comment.body

        # Verify Markdown summary also does not leak incorrect suggestions
        md = summary.to_markdown()
        assert "return amount * (1 - rate)" not in md
        assert "fee = fee * 2" not in md

        # Verify Dashboard extraction receives empty suggestions
        dashboard_findings = _extract_findings(summary)
        for f in dashboard_findings:
            assert f.suggestion == ""

    def test_production_path_preserves_valid_suggestions_positive_control(self) -> None:
        """Requirement 6: Positive control proving that valid suggestions remain available."""
        from src.github.client import GitHubClient
        from src.github.orchestrator import ReviewOrchestrator
        from src.llm.prompts import ReviewPromptBuilder
        from tests.dashboard import _extract_findings
        from unittest.mock import Mock

        valid_json = json.dumps(
            {
                "findings": [
                    {
                        "file": "src/auth.py",
                        "line": 4,
                        "category": "Security",
                        "severity": "Critical",
                        "message": "Password comparison uses prefix matching instead of exact equality.",
                        "evidence": "return supplied_password.startswith(stored_password)",
                        "suggestion": "return supplied_password == stored_password",
                    },
                    {
                        "file": "src/cart.py",
                        "line": 10,
                        "category": "Bug",
                        "severity": "High",
                        "message": "Discount percentage should be calculated from total.",
                        "evidence": "return total - discount_percent",
                        "suggestion": "return total * (1 - discount_percent / 100)",
                    },
                ]
            }
        )

        files = [
            {
                "filename": "src/auth.py",
                "status": "modified",
                "patch": "@@ -1,4 +1,4 @@\n \"\"\"Toy password verification for review testing.\"\"\"\n \n def verify_password(stored_password, supplied_password):\n-    return supplied_password == stored_password\n+    return supplied_password.startswith(stored_password)\n",
                "additions": 1,
                "deletions": 1,
                "changes": 2,
            },
            {
                "filename": "src/cart.py",
                "status": "modified",
                "patch": "@@ -7,4 +7,4 @@\n     if not 0 <= discount_percent <= 100:\n         raise ValueError(\"discount_percent must be between 0 and 100\")\n \n-    return total * (1 - discount_percent / 100)\n+    return total - discount_percent\n",
                "additions": 1,
                "deletions": 1,
                "changes": 2,
            },
        ]

        def _mock_req(method, url, **kwargs):
            resp = Mock()
            resp.status_code = 200
            resp.headers = {}
            if "pulls/1/files" in url:
                resp.json.return_value = files
                resp.links = {}
            elif "pulls/1/comments" in url:
                resp.json.return_value = []
                resp.links = {}
            elif "pulls/1" in url:
                resp.json.return_value = {
                    "number": 1,
                    "title": "MiniShop PR",
                    "body": "PR description",
                    "head": {"sha": "headsha"},
                    "base": {"ref": "main"},
                }
            elif "contents" in url:
                resp.status_code = 404
                resp.json.return_value = {}
            return resp

        session = Mock()
        session.headers = {}
        session.request.side_effect = _mock_req
        client = GitHubClient(token="mock", session=session, sleeper=lambda _: None)

        class MockRulesProvider:
            def get_rules(self, pr):
                return MINISHOP_REQUIREMENTS

        prompt_builder = ReviewPromptBuilder(max_files=5, max_findings=5)
        pipeline = ReviewPipeline(
            MockProvider(valid_json),
            prompt_builder=prompt_builder,
            parser=ReviewResponseParser(max_findings=5),
            max_requests=5,
            review_budget_seconds=240,
        )

        orchestrator = ReviewOrchestrator(
            client,
            pipeline,
            rules_provider=MockRulesProvider(),
            dry_run=True,
        )

        summary = orchestrator.run("omkark0117", "minishop-review-trial01", 1)

        assert summary.total_findings == 2
        assert len(summary.intended_comments) == 2

        auth_comment = next(c for c in summary.intended_comments if c.path == "src/auth.py")
        assert "return supplied_password == stored_password" in auth_comment.body

        cart_comment = next(c for c in summary.intended_comments if c.path == "src/cart.py")
        assert "return total * (1 - discount_percent / 100)" in cart_comment.body

        # Verify dashboard extraction also preserves valid suggestions
        dashboard_findings = _extract_findings(summary)
        auth_finding = next(f for f in dashboard_findings if f.file_path == "src/auth.py")
        assert "return supplied_password == stored_password" in auth_finding.suggestion

    def test_withhold_suggestion_when_requirements_absent(self) -> None:
        """Requirement 8: If requirements are absent, do not invent the intended fix.

        Withhold the suggestion when correctness cannot be established.
        """
        finding = ReviewFinding(
            file="src/calculator.py",
            line=2,
            category="Bug",
            severity="High",
            message="Complex formula defect.",
            evidence="return a * b + c",
            suggestion="return a + b * c",
        )
        context = ReviewContextBuilder().build(
            changed_code={"src/calculator.py": "def calc(a, b, c):\n    return a * b + c\n"},
            changed_lines={"src/calculator.py": [2]},
            changed_line_content={"src/calculator.py": {2: "    return a * b + c"}},
            repository_rules=[],  # ABSENT REQUIREMENTS
        )

        reason = validate_suggestion(finding, context, strict=False)
        assert reason is not None
        assert "Requirements are absent; withholding unverified logic suggestion" in reason

        parser = ReviewResponseParser()
        raw = json.dumps({"findings": [finding.model_dump()]})
        parsed = parser.parse(raw, context)
        assert len(parsed) == 1
        assert parsed[0].suggestion is None
