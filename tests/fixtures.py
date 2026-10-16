"""Synthetic PR test fixtures for Member 2 static analysis verification."""

from typing import Dict, List
from src.shared.models import PRContext
from src.analysis.evaluation import GroundTruthDefect


def get_clean_code_pr() -> PRContext:
    """Fixture: Clean Python code with no expected findings."""
    diff = (
        "diff --git a/src/clean.py b/src/clean.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/src/clean.py\n"
        "@@ -0,0 +1,7 @@\n"
        "+import logging\n"
        "+\n"
        "+logger = logging.getLogger(__name__)\n"
        "+\n"
        "+def compute_sum(a: int, b: int) -> int:\n"
        "+    logger.debug('Adding %d and %d', a, b)\n"
        "+    return a + b\n"
    )
    return PRContext(
        repository_owner="team",
        repository_name="demo",
        pr_number=101,
        title="Add clean calculation utility",
        description="Clean helper function with proper logging.",
        base_sha="base101",
        head_sha="head101",
        changed_files=["src/clean.py"],
        diff=diff,
    )


def get_deterministic_defect_pr() -> PRContext:
    """Fixture: PR introducing mutable default argument and duplicate dictionary key."""
    diff = (
        "diff --git a/src/data.py b/src/data.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/src/data.py\n"
        "@@ -0,0 +1,6 @@\n"
        "+def register_items(items=[]):\n"
        "+    config = {'env': 'dev', 'env': 'prod'}\n"
        "+    if items == None:\n"
        "+        return None\n"
        "+    items.append('new')\n"
        "+    return items\n"
    )
    return PRContext(
        repository_owner="team",
        repository_name="demo",
        pr_number=102,
        title="Add data registration helper",
        description="Introduces mutable default argument, duplicate dict keys, and == None.",
        base_sha="base102",
        head_sha="head102",
        changed_files=["src/data.py"],
        diff=diff,
    )


def get_security_defect_pr() -> PRContext:
    """Fixture: PR introducing SQL injection, insecure tempfile, and synthetic token."""
    diff = (
        "diff --git a/src/db.py b/src/db.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/src/db.py\n"
        "@@ -0,0 +1,7 @@\n"
        "+import tempfile\n"
        "+\n"
        "+def query_user(cursor, user_id):\n"
        "+    tmp_path = tempfile.mktemp()\n"
        "+    cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')\n"
        "+    api_key = 'abcdef1234567890abcdef1234567890'\n"
        "+    return tmp_path\n"
    )
    return PRContext(
        repository_owner="team",
        repository_name="demo",
        pr_number=103,
        title="Add database lookup",
        description="Introduces SQL injection via f-string, mktemp, and synthetic credential.",
        base_sha="base103",
        head_sha="head103",
        changed_files=["src/db.py"],
        diff=diff,
    )


def get_complexity_violation_pr() -> PRContext:
    """Fixture: PR introducing high cyclomatic complexity and excessive parameters."""
    diff = (
        "diff --git a/src/complex.py b/src/complex.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/src/complex.py\n"
        "@@ -0,0 +1,24 @@\n"
        "+def complex_decision_engine(a, b, c, d, e, f, g, h):\n"
        "+    if a > 0 and b > 0 and c > 0:\n"
        "+        if d:\n"
        "+            return 1\n"
        "+        elif e:\n"
        "+            return 2\n"
        "+    elif f:\n"
        "+        while g > 0:\n"
        "+            g -= 1\n"
        "+            if h:\n"
        "+                break\n"
        "+    for i in range(10):\n"
        "+        if i % 2 == 0:\n"
        "+            try:\n"
        "+                pass\n"
        "+            except ValueError:\n"
        "+                pass\n"
        "+            except KeyError:\n"
        "+                pass\n"
        "+    return 0\n"
    )
    return PRContext(
        repository_owner="team",
        repository_name="demo",
        pr_number=104,
        title="Add complex decision engine",
        description="Function exceeding cyclomatic complexity and parameter thresholds.",
        base_sha="base104",
        head_sha="head104",
        changed_files=["src/complex.py"],
        diff=diff,
    )


def get_changed_vs_unchanged_pr() -> PRContext:
    """Fixture: Unchanged context line contains defect; changed line is clean."""
    diff = (
        "diff --git a/src/service.py b/src/service.py\n"
        "--- a/src/service.py\n"
        "+++ b/src/service.py\n"
        "@@ -1,4 +1,4 @@\n"
        " def handle(req):\n"
        "     eval(req.payload)\n"
        "-    x = 1\n"
        "+    x = 2\n"
        "     return x\n"
    )
    return PRContext(
        repository_owner="team",
        repository_name="demo",
        pr_number=105,
        title="Tweak variable assignment",
        description="Modifies clean line; existing pre-change eval line should NOT be reported.",
        base_sha="base105",
        head_sha="head105",
        changed_files=["src/service.py"],
        diff=diff,
    )


def get_duplicate_defect_pr() -> PRContext:
    """Fixture: Nested duplicate call on the same line."""
    diff = (
        "diff --git a/src/dup.py b/src/dup.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/src/dup.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def run(cmd):\n"
        "+    eval(eval(cmd))\n"
    )
    return PRContext(
        repository_owner="team",
        repository_name="demo",
        pr_number=106,
        title="Add nested call",
        description="Nested eval on single line should be deduplicated.",
        base_sha="base106",
        head_sha="head106",
        changed_files=["src/dup.py"],
        diff=diff,
    )


def get_file_lifecycle_pr() -> PRContext:
    """Fixture: PR with added, modified, deleted, and renamed files."""
    diff = (
        "diff --git a/added.py b/added.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/added.py\n"
        "@@ -0,0 +1,1 @@\n"
        "+def new_func(): pass\n"
        "diff --git a/deleted.py b/deleted.py\n"
        "deleted file mode 100644\n"
        "--- a/deleted.py\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-eval(bad)\n"
        "-exec(bad)\n"
        "diff --git a/old_name.py b/new_name.py\n"
        "similarity index 100%\n"
        "rename from old_name.py\n"
        "rename to new_name.py\n"
    )
    return PRContext(
        repository_owner="team",
        repository_name="demo",
        pr_number=107,
        title="Refactor repository file layout",
        description="Covers added, deleted, and renamed files.",
        base_sha="base107",
        head_sha="head107",
        changed_files=["added.py", "deleted.py", "new_name.py"],
        diff=diff,
    )


def get_ground_truth_map() -> Dict[int, List[GroundTruthDefect]]:
    """Ground truth mapping for benchmark evaluation."""
    return {
        101: [],
        102: [
            GroundTruthDefect(file_path="src/data.py", line_number=1, category="static_analysis", rule_id="DEF003"),
            GroundTruthDefect(file_path="src/data.py", line_number=2, category="static_analysis", rule_id="DEF006"),
            GroundTruthDefect(file_path="src/data.py", line_number=3, category="static_analysis", rule_id="DEF005"),
        ],
        103: [
            GroundTruthDefect(file_path="src/db.py", line_number=4, category="security", rule_id="SEC004"),
            GroundTruthDefect(file_path="src/db.py", line_number=5, category="security", rule_id="SEC006"),
            GroundTruthDefect(file_path="src/db.py", line_number=6, category="security", rule_id="SEC005"),
        ],
        104: [
            GroundTruthDefect(file_path="src/complex.py", line_number=1, category="static_analysis", rule_id="CMP001"),
            GroundTruthDefect(file_path="src/complex.py", line_number=1, category="static_analysis", rule_id="CMP002"),
        ],
        105: [],  # Defect was on unchanged line, so 0 findings expected on changed lines
        106: [
            GroundTruthDefect(file_path="src/dup.py", line_number=2, category="security", rule_id="SEC001"),
        ],
        107: [],
    }
