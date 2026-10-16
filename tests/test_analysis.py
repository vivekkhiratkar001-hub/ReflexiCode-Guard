import unittest

from src.analysis import analyze_pr
from src.shared.enums import FindingCategory, Severity
from src.shared.models import PRContext


def context(diff: str) -> PRContext:
    return PRContext(
        repository_owner="team",
        repository_name="demo",
        pr_number=12,
        title="Analysis test",
        description="",
        base_sha="base",
        head_sha="head",
        diff=diff,
    )


def single_file_diff(path: str, hunk: str) -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        f"{hunk}"
    )


class DiffParsingTests(unittest.TestCase):
    def test_empty_diff_returns_no_findings(self) -> None:
        self.assertEqual(analyze_pr(context("")), [])

    def test_added_line_has_new_file_line_number(self) -> None:
        diff = single_file_diff(
            "app.py",
            "@@ -10,3 +10,5 @@\n"
            " before()\n"
            "+eval(value)\n"
            " middle()\n"
            "+exec(other)\n"
            " after()\n",
        )

        findings = analyze_pr(context(diff))

        self.assertEqual([item.line_number for item in findings], [11, 13])

    def test_multiple_files_and_hunks_are_localized(self) -> None:
        diff = (
            single_file_diff("one.py", "@@ -1 +1,2 @@\n first()\n+eval(x)\n")
            + single_file_diff(
                "two.py",
                "@@ -4 +4,2 @@\n safe()\n+exec(y)\n"
                "@@ -20 +21,2 @@\n old()\n+os.system(cmd)\n",
            )
        )

        findings = analyze_pr(context(diff))

        self.assertEqual(
            [(item.file_path, item.line_number) for item in findings],
            [("one.py", 2), ("two.py", 5), ("two.py", 22)],
        )

    def test_new_file_is_analyzed(self) -> None:
        diff = (
            "diff --git a/new.py b/new.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/new.py\n"
            "@@ -0,0 +1,1 @@\n"
            "+eval(data)\n"
        )

        findings = analyze_pr(context(diff))

        self.assertEqual(
            [(item.file_path, item.line_number) for item in findings],
            [("new.py", 1)],
        )

    def test_deleted_file_is_ignored(self) -> None:
        diff = (
            "diff --git a/removed.py b/removed.py\n"
            "deleted file mode 100644\n"
            "--- a/removed.py\n"
            "+++ /dev/null\n"
            "@@ -1,1 +0,0 @@\n"
            "-eval(data)\n"
        )

        self.assertEqual(analyze_pr(context(diff)), [])

    def test_unusual_incomplete_hunk_does_not_crash(self) -> None:
        diff = (
            "diff --git a/app.py b/app.py\n"
            "+++ b/app.py\n"
            "@@ malformed @@\n"
            "+eval(data)\n"
            "@@ -2 +2,2 @@\n"
            " safe()\n"
            "+exec(data)\n"
        )

        findings = analyze_pr(context(diff))

        self.assertEqual([(item.line_number, item.evidence) for item in findings], [(3, "exec(data)")])


class PythonRuleTests(unittest.TestCase):
    def test_eval_and_exec_are_security_findings(self) -> None:
        diff = single_file_diff(
            "app.py", "@@ -0,0 +1,2 @@\n+eval(value)\n+exec(code)\n"
        )

        findings = analyze_pr(context(diff))

        self.assertEqual(len(findings), 2)
        self.assertTrue(
            all(item.category == FindingCategory.SECURITY.value for item in findings)
        )
        self.assertTrue(all(item.severity == Severity.HIGH.value for item in findings))

    def test_os_system_is_detected(self) -> None:
        diff = single_file_diff("app.py", "@@ -0,0 +1,1 @@\n+os.system(command)\n")

        finding = analyze_pr(context(diff))[0]

        self.assertEqual(finding.category, "security")
        self.assertIn("shell", finding.message)
        self.assertEqual(finding.evidence, "os.system(command)")

    def test_subprocess_shell_true_is_localized_to_keyword(self) -> None:
        diff = single_file_diff(
            "app.py",
            "@@ -0,0 +1,4 @@\n"
            "+import subprocess\n"
            "+subprocess.run(\n"
            "+    command,\n"
            "+    shell=True,\n"
            "+)\n",
        )

        finding = analyze_pr(context(diff))[0]

        self.assertEqual(finding.line_number, 4)
        self.assertEqual(finding.evidence, "shell=True,")

    def test_bare_except_is_reported(self) -> None:
        diff = single_file_diff(
            "app.py",
            "@@ -0,0 +1,3 @@\n"
            "+try:\n"
            "+    run()\n"
            "+except:\n",
        )

        finding = analyze_pr(context(diff))[0]

        self.assertEqual(finding.line_number, 3)
        self.assertEqual(finding.category, "static_analysis")
        self.assertIn("bare except", finding.message)

    def test_print_is_reported(self) -> None:
        diff = single_file_diff("app.py", "@@ -0,0 +1,1 @@\n+print(result)\n")

        finding = analyze_pr(context(diff))[0]

        self.assertEqual(finding.line_number, 1)
        self.assertEqual(finding.severity, "low")

    def test_safe_code_has_no_findings(self) -> None:
        diff = single_file_diff(
            "app.py",
            "@@ -0,0 +1,3 @@\n"
            "+def add(left, right):\n"
            "+    return left + right\n"
            "+logger.info('ready')\n",
        )

        self.assertEqual(analyze_pr(context(diff)), [])

    def test_deleted_or_unchanged_context_lines_are_not_reported(self) -> None:
        diff = single_file_diff(
            "app.py",
            "@@ -1,3 +1,2 @@\n"
            "safe()\n"
            "-eval(deleted_value)\n"
            "exec(unchanged_value)\n",
        )

        self.assertEqual(analyze_pr(context(diff)), [])

    def test_malformed_python_uses_lexical_fallback(self) -> None:
        diff = single_file_diff(
            "app.py", "@@ -0,0 +1,2 @@\n+if :\n+    eval(value)\n"
        )

        findings = analyze_pr(context(diff))

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].line_number, 2)

    def test_lexical_fallback_ignores_comments_and_strings(self) -> None:
        diff = single_file_diff(
            "app.py",
            "@@ -0,0 +1,3 @@\n"
            "+# eval(value)\n"
            "+message = 'exec(code)'\n"
            "+actual = eval(value\n",
        )

        findings = analyze_pr(context(diff))

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].line_number, 3)

    def test_nested_duplicate_rule_on_one_line_is_deduplicated(self) -> None:
        diff = single_file_diff(
            "app.py", "@@ -0,0 +1,1 @@\n+eval(eval(value))\n"
        )

        findings = analyze_pr(context(diff))

        self.assertEqual(len(findings), 1)

    def test_finding_has_all_shared_contract_fields(self) -> None:
        diff = single_file_diff("app.py", "@@ -0,0 +1,1 @@\n+eval(value)\n")

        finding = analyze_pr(context(diff))[0]

        self.assertEqual(finding.file_path, "app.py")
        self.assertEqual(finding.line_number, 1)
        self.assertEqual(finding.category, "security")
        self.assertEqual(finding.severity, "high")
        self.assertTrue(finding.message)
        self.assertEqual(finding.evidence, "eval(value)")
        self.assertTrue(finding.suggestion)

    def test_non_python_file_does_not_run_python_rules(self) -> None:
        diff = single_file_diff("app.js", "@@ -0,0 +1,1 @@\n+eval(value)\n")

        self.assertEqual(analyze_pr(context(diff)), [])


if __name__ == "__main__":
    unittest.main()
