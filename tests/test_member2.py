"""Comprehensive test suite for Member 2: AST, diff localization, static, security, complexity, and evaluation."""

import ast
import base64
import unittest
from typing import List

from src.analysis import (
    ComplexityConfig,
    FileDiff,
    analyze_diff,
    analyze_pr,
    calculate_cyclomatic_complexity,
    count_parameters,
    deduplicate_and_sort_findings,
    parse_source_or_fragment,
    parse_unified_diff,
    tokenize_source,
)
from src.analysis.evaluation import (
    GroundTruthDefect,
    evaluate_findings,
    run_benchmark_suite,
)
from src.analysis.rules.base import RuleFinding
from src.github import review_pr
from src.shared.enums import FindingCategory, ReviewStatus, Severity
from src.shared.models import AnalysisFinding, PRContext

from tests.fixtures import (
    get_changed_vs_unchanged_pr,
    get_clean_code_pr,
    get_complexity_violation_pr,
    get_deterministic_defect_pr,
    get_duplicate_defect_pr,
    get_file_lifecycle_pr,
    get_ground_truth_map,
    get_security_defect_pr,
)


class ASTParserTests(unittest.TestCase):
    def test_parses_valid_module(self) -> None:
        tree, offset, diag = parse_source_or_fragment(["def foo():", "    return 42"])
        self.assertIsNotNone(tree)
        self.assertEqual(offset, 0)
        self.assertIsNone(diag)

    def test_parses_indented_fragment_without_error(self) -> None:
        tree, offset, diag = parse_source_or_fragment(["    return value + 1"])
        self.assertIsNotNone(tree)
        self.assertIsNone(diag)

    def test_parses_try_except_fragment(self) -> None:
        tree, offset, diag = parse_source_or_fragment(["except ValueError:", "    pass"])
        self.assertIsNotNone(tree)
        self.assertEqual(offset, 2)

    def test_parses_elif_fragment_with_if_wrapper(self) -> None:
        tree, offset, diag = parse_source_or_fragment(["elif count > 0:", "    total += count"])
        self.assertIsNotNone(tree)
        self.assertEqual(offset, 2)

    def test_tokens_filters_trivia(self) -> None:
        tokens = tokenize_source("# comment\nx = 10\n")
        token_strings = [t.string for t in tokens]
        self.assertEqual(token_strings, ["x", "=", "10"])


class DiffParserTests(unittest.TestCase):
    def test_empty_diff_returns_empty_list(self) -> None:
        self.assertEqual(parse_unified_diff(""), [])
        self.assertEqual(parse_unified_diff("   \n"), [])

    def test_parses_single_file_with_multiple_hunks(self) -> None:
        diff = (
            "diff --git a/app.py b/app.py\n"
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -5,2 +5,3 @@\n"
            " line1\n"
            "+added1\n"
            " line2\n"
            "@@ -20,2 +21,3 @@\n"
            " line20\n"
            "+added2\n"
            " line21\n"
        )
        file_diffs = parse_unified_diff(diff)
        self.assertEqual(len(file_diffs), 1)
        self.assertEqual(file_diffs[0].target_path, "app.py")
        self.assertEqual(len(file_diffs[0].hunks), 2)
        self.assertEqual(file_diffs[0].added_line_numbers, {6, 22})

    def test_parses_renamed_file(self) -> None:
        diff = (
            "diff --git a/old.py b/new.py\n"
            "rename from old.py\n"
            "rename to new.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-old_line\n"
            "+new_line\n"
        )
        file_diffs = parse_unified_diff(diff)
        self.assertEqual(len(file_diffs), 1)
        self.assertEqual(file_diffs[0].change_type, "renamed")
        self.assertEqual(file_diffs[0].target_path, "new.py")

    def test_parses_deleted_file(self) -> None:
        diff = (
            "diff --git a/obsolete.py b/obsolete.py\n"
            "deleted file mode 100644\n"
            "--- a/obsolete.py\n"
            "+++ /dev/null\n"
            "@@ -1,2 +0,0 @@\n"
            "-line1\n"
            "-line2\n"
        )
        file_diffs = parse_unified_diff(diff)
        self.assertEqual(len(file_diffs), 1)
        self.assertEqual(file_diffs[0].change_type, "deleted")
        self.assertEqual(len(file_diffs[0].added_line_numbers), 0)

    def test_handles_no_newline_marker(self) -> None:
        diff = (
            "diff --git a/one.py b/one.py\n"
            "--- a/one.py\n"
            "+++ b/one.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "\\ No newline at end of file\n"
            "+new\n"
            "\\ No newline at end of file\n"
        )
        file_diffs = parse_unified_diff(diff)
        self.assertEqual(len(file_diffs), 1)
        self.assertEqual(file_diffs[0].added_line_numbers, {1})


class DeterministicStaticRulesTests(unittest.TestCase):
    def test_mutable_default_argument_detected(self) -> None:
        diff = (
            "diff --git a/mod.py b/mod.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/mod.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+def add_item(val, container=[]):\n"
            "+    container.append(val)\n"
        )
        pr = PRContext("org", "repo", 1, "test", "", "b", "h", diff=diff)
        findings = analyze_pr(pr)
        mutable_findings = [f for f in findings if "mutable default" in f.message.lower()]
        self.assertEqual(len(mutable_findings), 1)
        self.assertEqual(mutable_findings[0].line_number, 1)

    def test_unreachable_code_detected(self) -> None:
        diff = (
            "diff --git a/mod.py b/mod.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/mod.py\n"
            "@@ -0,0 +1,4 @@\n"
            "+def compute():\n"
            "+    return 1\n"
            "+    dead_code = 2\n"
            "+    return dead_code\n"
        )
        pr = PRContext("org", "repo", 2, "test", "", "b", "h", diff=diff)
        findings = analyze_pr(pr)
        unreachable = [f for f in findings if "unreachable" in f.message.lower()]
        self.assertTrue(len(unreachable) >= 1)
        self.assertEqual(unreachable[0].line_number, 3)

    def test_singleton_equality_comparison(self) -> None:
        diff = (
            "diff --git a/mod.py b/mod.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/mod.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+def check(x):\n"
            "+    if x == None: pass\n"
        )
        pr = PRContext("org", "repo", 3, "test", "", "b", "h", diff=diff)
        findings = analyze_pr(pr)
        singleton = [f for f in findings if "singleton" in f.message.lower()]
        self.assertEqual(len(singleton), 1)
        self.assertEqual(singleton[0].line_number, 2)

    def test_numeric_comparisons_are_not_mistaken_for_singleton_checks(self) -> None:
        diff = (
            "diff --git a/mod.py b/mod.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/mod.py\n"
            "@@ -0,0 +1,4 @@\n"
            "+def check(value):\n"
            "+    if value == 0:\n"
            "+        return value == 1\n"
            "+    return value == False\n"
        )
        pr = PRContext("org", "repo", 30, "test", "", "b", "h", diff=diff)

        findings = analyze_pr(pr)

        singleton = [f for f in findings if "singleton" in f.message.lower()]
        self.assertEqual(len(singleton), 1)
        self.assertEqual(singleton[0].line_number, 4)

    def test_singleton_comparisons_detect_constants_on_either_side(self) -> None:
        diff = (
            "diff --git a/mod.py b/mod.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/mod.py\n"
            "@@ -0,0 +1,4 @@\n"
            "+def check(value):\n"
            "+    if False == value: pass\n"
            "+    if value is True: pass\n"
            "+    if 0 is value: pass\n"
        )
        findings = analyze_pr(PRContext("org", "repo", 31, "test", "", "b", "h", diff=diff))
        self.assertTrue(any("singleton" in finding.message.lower() and finding.line_number == 2 for finding in findings))
        identity = [finding for finding in findings if "using 'is'" in finding.message.lower()]
        self.assertEqual([finding.line_number for finding in identity], [4])

    def test_duplicate_dict_key_detected(self) -> None:
        diff = (
            "diff --git a/mod.py b/mod.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/mod.py\n"
            "@@ -0,0 +1,1 @@\n"
            "+cfg = {'timeout': 10, 'timeout': 20}\n"
        )
        pr = PRContext("org", "repo", 4, "test", "", "b", "h", diff=diff)
        findings = analyze_pr(pr)
        dup = [f for f in findings if "duplicate dictionary key" in f.message.lower()]
        self.assertEqual(len(dup), 1)
        self.assertEqual(dup[0].line_number, 1)

    def test_assert_tuple_detected(self) -> None:
        diff = (
            "diff --git a/mod.py b/mod.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/mod.py\n"
            "@@ -0,0 +1,1 @@\n"
            "+assert (x > 0, 'Must be positive')\n"
        )
        pr = PRContext("org", "repo", 5, "test", "", "b", "h", diff=diff)
        findings = analyze_pr(pr)
        assert_f = [f for f in findings if "assert statement called with a non-empty tuple" in f.message.lower()]
        self.assertEqual(len(assert_f), 1)


class SecurityRulesTests(unittest.TestCase):
    def test_insecure_deserialization_detected(self) -> None:
        diff = (
            "diff --git a/ser.py b/ser.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/ser.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+import pickle\n"
            "+obj = pickle.loads(user_payload)\n"
        )
        pr = PRContext("org", "repo", 6, "test", "", "b", "h", diff=diff)
        findings = analyze_pr(pr)
        pickle_f = [f for f in findings if "pickle" in f.message.lower()]
        self.assertEqual(len(pickle_f), 1)
        self.assertEqual(pickle_f[0].line_number, 2)
        self.assertEqual(pickle_f[0].severity, Severity.HIGH.value)

    def test_insecure_mktemp_detected(self) -> None:
        diff = (
            "diff --git a/f.py b/f.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/f.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+import tempfile\n"
            "+path = tempfile.mktemp()\n"
        )
        pr = PRContext("org", "repo", 7, "test", "", "b", "h", diff=diff)
        findings = analyze_pr(pr)
        mktemp_f = [f for f in findings if "mktemp" in f.message.lower()]
        self.assertEqual(len(mktemp_f), 1)
        self.assertEqual(mktemp_f[0].line_number, 2)

    def test_sql_injection_detected(self) -> None:
        diff = (
            "diff --git a/sql.py b/sql.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/sql.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+def fetch(cursor, name):\n"
            "+    cursor.execute(f'SELECT * FROM items WHERE name = {name}')\n"
        )
        pr = PRContext("org", "repo", 8, "test", "", "b", "h", diff=diff)
        findings = analyze_pr(pr)
        sql_f = [f for f in findings if "sql injection" in f.message.lower()]
        self.assertEqual(len(sql_f), 1)
        self.assertEqual(sql_f[0].line_number, 2)

    def test_hardcoded_aws_key_masked_in_evidence(self) -> None:
        synthetic_key = "AKIA" + "A1B2C3D4E5F6G7H8"
        diff = (
            "diff --git a/auth.py b/auth.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/auth.py\n"
            "@@ -0,0 +1,1 @@\n"
            f"+AWS_KEY = '{synthetic_key}'\n"
        )
        pr = PRContext("org", "repo", 9, "test", "", "b", "h", diff=diff)
        findings = analyze_pr(pr)
        aws_f = [f for f in findings if "aws" in f.message.lower()]
        self.assertEqual(len(aws_f), 1)
        self.assertEqual(aws_f[0].severity, Severity.CRITICAL.value)
        # Verify secret is masked in evidence!
        self.assertNotIn(synthetic_key, aws_f[0].evidence)
        self.assertIn("AKIA****", aws_f[0].evidence)

    def test_clearly_synthetic_aws_fixture_is_not_reported_even_in_tests(self) -> None:
        synthetic_key = "AKIA" + "1234567890ABCDEF"
        diff = (
            "diff --git a/tests/fixtures.py b/tests/fixtures.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/tests/fixtures.py\n"
            "@@ -0,0 +1,1 @@\n"
            f"+AWS_KEY = '{synthetic_key}'\n"
        )
        self.assertEqual(analyze_pr(PRContext("org", "repo", 10, "test", "", "b", "h", diff=diff)), [])

    def test_aws_key_ending_in_example_is_not_broadly_exempted(self) -> None:
        key = "AKIA" + "A1B2C3D4EEXAMPLE"
        diff = (
            "diff --git a/auth.py b/auth.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/auth.py\n"
            "@@ -0,0 +1,1 @@\n"
            f"+AWS_KEY = '{key}'\n"
        )
        findings = analyze_pr(PRContext("org", "repo", 15, "test", "", "b", "h", diff=diff))
        self.assertTrue(any("aws access key" in finding.message.lower() for finding in findings))

    def test_complete_private_key_block_is_detected_with_redacted_evidence(self) -> None:
        def der_value(tag: int, content: bytes) -> bytes:
            return bytes((tag, len(content))) + content

        algorithm = der_value(
            0x30,
            der_value(0x06, bytes.fromhex("2a864886f70d010101")) + b"\x05\x00",
        )
        rsa_private_key = der_value(
            0x30,
            b"".join(der_value(0x02, b"\x01") for _ in range(9)),
        )
        der = der_value(
            0x30,
            der_value(0x02, b"\x00")
            + algorithm
            + der_value(0x04, rsa_private_key),
        )
        body = base64.b64encode(der).decode("ascii")
        diff = (
            "diff --git a/keys.py b/keys.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/keys.py\n"
            "@@ -0,0 +1,5 @@\n"
            "+PRIVATE_KEY = '''-----BEGIN PRIVATE KEY-----\n"
            f"+{body}\n"
            "+-----END PRIVATE KEY-----\n"
            "+'''\n"
        )
        findings = analyze_pr(PRContext("org", "repo", 11, "test", "", "b", "h", diff=diff))
        private_findings = [f for f in findings if "private key block" in f.message.lower()]
        self.assertEqual(len(private_findings), 1)
        self.assertEqual(private_findings[0].line_number, 1)
        self.assertEqual(private_findings[0].evidence, "[REDACTED PRIVATE KEY BLOCK]")
        self.assertNotIn(body, private_findings[0].evidence)

    def test_unrelated_der_sequence_is_not_reported_as_a_private_key(self) -> None:
        encoded_data = base64.b64encode(
            b"\x30\x12" + b"\x02\x01\x00" * 6
        ).decode("ascii")
        diff = (
            "diff --git a/data.py b/data.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/data.py\n"
            "@@ -0,0 +1,3 @@\n"
            "+DATA = '''-----BEGIN PRIVATE KEY-----\n"
            f"+{encoded_data}\n"
            "+-----END PRIVATE KEY-----'''\n"
        )
        findings = analyze_pr(PRContext("org", "repo", 14, "test", "", "b", "h", diff=diff))
        self.assertFalse(any("private key block" in f.message.lower() for f in findings))

    def test_private_key_rule_marker_without_a_complete_block_is_not_reported(self) -> None:
        diff = (
            "diff --git a/security_rules.py b/security_rules.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/security_rules.py\n"
            "@@ -0,0 +1,1 @@\n"
            "+PRIVATE_KEY_MARKER = '-----BEGIN PRIVATE KEY-----'\n"
        )
        findings = analyze_pr(PRContext("org", "repo", 12, "test", "", "b", "h", diff=diff))
        self.assertFalse(any("private key block" in f.message.lower() for f in findings))

    def test_security_scanning_remains_enabled_in_test_paths(self) -> None:
        diff = (
            "diff --git a/tests/test_security.py b/tests/test_security.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/tests/test_security.py\n"
            "@@ -0,0 +1,1 @@\n"
            "+result = eval(user_input)\n"
        )
        findings = analyze_pr(PRContext("org", "repo", 13, "test", "", "b", "h", diff=diff))
        self.assertTrue(any(f.category == "security" for f in findings))

    def test_placeholder_token_is_not_flagged(self) -> None:
        diff = (
            "diff --git a/config.py b/config.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/config.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+api_key = 'your_api_key_here'\n"
            "+dummy_token = 'example_dummy_token'\n"
        )
        pr = PRContext("org", "repo", 10, "test", "", "b", "h", diff=diff)
        findings = analyze_pr(pr)
        self.assertEqual(findings, [])


class ComplexityRulesTests(unittest.TestCase):
    def test_calculates_cyclomatic_complexity(self) -> None:
        code = (
            "def sample(x, y):\n"
            "    if x > 0:\n"
            "        if y > 0:\n"
            "            return 1\n"
            "    return 0\n"
        )
        tree = ast.parse(code)
        func = tree.body[0]
        # Base 1 + 2 ifs = 3
        self.assertEqual(calculate_cyclomatic_complexity(func), 3)

    def test_nested_function_branches_are_not_counted_for_outer_function(self) -> None:
        tree = ast.parse(
            "def outer(value):\n"
            "    def inner(item):\n"
            "        if item:\n"
            "            if item > 1:\n"
            "                return 1\n"
            "        return 0\n"
            "    return inner(value)\n"
        )
        self.assertEqual(calculate_cyclomatic_complexity(tree.body[0]), 1)
        self.assertEqual(calculate_cyclomatic_complexity(tree.body[0].body[0]), 3)

    def test_parameter_count_calculation(self) -> None:
        code = "def sample(a, b, c, d, e, *args, **kwargs): pass\n"
        tree = ast.parse(code)
        self.assertEqual(count_parameters(tree.body[0]), 7)

    def test_high_complexity_function_reported(self) -> None:
        pr = get_complexity_violation_pr()
        findings = analyze_pr(pr)
        cmp_findings = [f for f in findings if "complexity" in f.message.lower()]
        self.assertEqual(len(cmp_findings), 1)
        self.assertIn("Function 'complex_decision_engine'", cmp_findings[0].message)


class NormalizationAndDeduplicationTests(unittest.TestCase):
    def test_deduplicates_repeated_findings(self) -> None:
        raw = [
            RuleFinding("SEC001", "security", "high", "eval() unsafe", "fix", 5, "eval(x)"),
            RuleFinding("SEC001", "security", "high", "eval() unsafe", "fix", 5, "eval(x)"),
        ]
        result = deduplicate_and_sort_findings("test.py", raw)
        self.assertEqual(len(result), 1)

    def test_preserves_distinct_findings_on_same_line(self) -> None:
        raw = [
            RuleFinding("DEF008", "static_analysis", "low", "print() used", "fix", 10, "print(x)"),
            RuleFinding("SEC001", "security", "high", "eval() unsafe", "fix", 10, "eval(print(x))"),
        ]
        result = deduplicate_and_sort_findings("test.py", raw)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].category, FindingCategory.SECURITY.value)
        self.assertEqual(result[1].category, FindingCategory.STATIC_ANALYSIS.value)


class FixturesAndEvaluationTests(unittest.TestCase):
    def test_clean_fixture_produces_no_findings(self) -> None:
        pr = get_clean_code_pr()
        findings = analyze_pr(pr)
        self.assertEqual(findings, [])

    def test_changed_vs_unchanged_fixture_does_not_flag_unchanged_lines(self) -> None:
        pr = get_changed_vs_unchanged_pr()
        findings = analyze_pr(pr)
        self.assertEqual(findings, [])

    def test_duplicate_nested_call_is_deduplicated(self) -> None:
        pr = get_duplicate_defect_pr()
        findings = analyze_pr(pr)
        self.assertEqual(len(findings), 1)

    def test_file_lifecycle_handles_added_deleted_and_renamed(self) -> None:
        pr = get_file_lifecycle_pr()
        findings = analyze_pr(pr)
        self.assertEqual(findings, [])

    def test_benchmark_evaluation_suite(self) -> None:
        fixtures = [
            get_clean_code_pr(),
            get_deterministic_defect_pr(),
            get_security_defect_pr(),
            get_complexity_violation_pr(),
            get_changed_vs_unchanged_pr(),
            get_duplicate_defect_pr(),
        ]
        gt_map = get_ground_truth_map()
        eval_report = run_benchmark_suite(fixtures, gt_map, line_tolerance=0)

        overall = eval_report["overall"]
        self.assertGreater(overall["total_expected"], 0)
        self.assertEqual(overall["false_negatives"], 0)
        self.assertGreaterEqual(overall["precision"], 0.9)
        self.assertGreaterEqual(overall["recall"], 0.9)
        self.assertGreaterEqual(overall["localization_accuracy"], 0.9)


class SharedContractIntegrationTests(unittest.TestCase):
    def test_full_pipeline_with_all_member2_rules(self) -> None:
        pr = get_security_defect_pr()
        result = review_pr(pr)

        self.assertEqual(result.pr_number, 103)
        self.assertEqual(result.status, ReviewStatus.COMPLETE)
        self.assertEqual(len(result.findings), 3)

        # Verify all findings satisfy AnalysisFinding contract
        for f in result.findings:
            self.assertIsInstance(f, AnalysisFinding)
            self.assertTrue(f.file_path)
            self.assertIsInstance(f.line_number, int)
            self.assertIn(f.category, ["security", "static_analysis", "semantic"])
            self.assertIn(f.severity, ["low", "medium", "high", "critical"])
            self.assertTrue(f.message)
            self.assertTrue(f.suggestion)


if __name__ == "__main__":
    unittest.main()
