"""Builds the controlled-PR benchmark. Line numbers of ground-truth defects are COMPUTED from the
generated diff (by locating a unique needle), never typed by hand.

Run:  python benchmarks/build_benchmark.py
Out:  benchmarks/controlled_prs/<case>/{pr_event.json, sample.diff, expected.json}

NOTE: these 13 PRs are synthetic and written for testing the M4 pipeline/evaluation harness.
They are NOT the project's real ground truth (valid PRs + human review targets). The runner accepts
any folder in this same layout, so the real dataset can be dropped in later.
"""
import difflib, json, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "controlled_prs")

def make_diff(files):
    chunks, new_texts = [], {}
    for path, old, new in files:
        o = old.splitlines() if old else []
        n = new.splitlines()
        new_texts[path] = n
        a = "/dev/null" if not old else f"a/{path}"
        hunks = list(difflib.unified_diff(o, n, fromfile=a, tofile=f"b/{path}", n=3, lineterm=""))
        chunks.append(f"diff --git a/{path} b/{path}\n" + "\n".join(hunks))
    return "\n".join(chunks) + "\n", new_texts

def locate(new_texts, path, needle):
    hits = [i + 1 for i, l in enumerate(new_texts[path]) if needle in l]
    assert len(hits) == 1, f"needle {needle!r} must match exactly once in {path}, got {hits}"
    return hits[0]

CASES = []
def case(name, title, files, defects=(), body=""):
    CASES.append((name, title, body, files, list(defects)))

# ---- cases ------------------------------------------------------------------------------------
case("01_hardcoded_secret", "Add DB config",
     [("src/config.py", "import os\n\nDEBUG = False\nHOST = os.environ.get('HOST', 'localhost')\nPORT = 5432\n",
       "import os\n\nDEBUG = False\nHOST = os.environ.get('HOST', 'localhost')\nPORT = 5432\nDB_PASSWORD = \"hunter2-prod\"\n")],
     [("src/config.py", 'DB_PASSWORD = "hunter2', "SEC001", "detectable")])

case("02_eval_in_handler", "Add calculator endpoint",
     [("src/api.py", "from flask import request\n\ndef health():\n    return 'ok'\n",
       "from flask import request\n\ndef health():\n    return 'ok'\n\ndef calc():\n    result = eval(request.args['expr'])\n    return str(result)\n")],
     [("src/api.py", "eval(request.args", "SEC002", "detectable")], body="Fixes #11")

case("03_todo_left", "Add input validation stub",
     [("src/validate.py", "def check(x):\n    return x is not None\n",
       "def check(x):\n    # FIXME: validate input length\n    return x is not None\n")],
     [("src/validate.py", "FIXME", "STY001", "detectable")])

case("04_shell_true", "Add backup helper",
     [("src/backup.py", "import subprocess\n\ndef noop():\n    pass\n",
       "import subprocess\n\ndef noop():\n    pass\n\ndef backup(path):\n    subprocess.run('tar czf b.tgz ' + path, shell=True)\n")],
     [("src/backup.py", "shell=True", "CMD-INJ", "detectable")])

case("05_sql_fstring", "Add user lookup",
     [("src/db.py", "def conn():\n    return None\n",
       "def conn():\n    return None\n\ndef find(cur, uid):\n    cur.execute(f\"SELECT * FROM users WHERE id = {uid}\")\n    return cur.fetchone()\n")],
     [("src/db.py", "cur.execute(f", "SQL-INJ", "detectable")])

case("06_bare_except", "Make loader resilient",
     [("src/loader.py", "import json\n\ndef load(p):\n    return json.load(open(p))\n",
       "import json\n\ndef load(p):\n    try:\n        return json.load(open(p))\n    except:\n        return {}\n")],
     [("src/loader.py", "except:", "BARE-EXCEPT", "detectable")])

case("07_off_by_one_logic", "Process all items",
     [("src/proc.py", "def run(items):\n    return len(items)\n",
       "def run(items):\n    out = []\n    for i in range(len(items) + 1):\n        out.append(items[i])\n    return out\n")],
     [("src/proc.py", "range(len(items) + 1)", "LOGIC-OFF-BY-ONE", "semantic")])

case("08_clean_refactor", "Rename helpers (no behaviour change)",
     [("src/util.py", "def evaluate_score(x):\n    return eval(x)\n\ndef get_token():\n    return 'a'\n",
       "def evaluate_score(x):\n    return int(x)\n\ndef get_token():\n    return 'a'\n\ntoken = get_token()\npassword_hash = hash(token)\n")],
     [])

case("09_preexisting_secret_in_context", "Add helper next to old config",
     [("src/settings.py", "A = 1\nB = 2\nAPI_KEY = \"old-prod-key-123\"\nC = 3\nD = 4\n",
       "A = 1\nB = 2\nAPI_KEY = \"old-prod-key-123\"\nC = 3\nD = 4\nE = 5\n")],
     [])   # pre-existing secret is out of scope for THIS PR

case("10_multi_file_mixed", "Add exporter and stats",
     [("src/export.py", "def export():\n    return 1\n", "def export():\n    api_key = \"sk-test-abcdef123\"\n    return 1\n"),
      ("src/reader.py", "def read(f):\n    return f.read()\n", "def read(f):\n    try:\n        return f.read()\n    except:\n        return ''\n"),
      ("src/stats.py", "def mean(xs):\n    return 0\n", "def mean(xs):\n    return sum(xs) / len(xs)\n")],
     [("src/export.py", "sk-test-abcdef123", "SEC001", "detectable"),
      ("src/reader.py", "except:", "BARE-EXCEPT", "detectable"),
      ("src/stats.py", "sum(xs) / len(xs)", "LOGIC-DIV-ZERO", "semantic")])

case("11_hardcoded_token_js", "Add GitHub client (JS)",
     [("web/gh.js", "export const base = 'https://api.github.com';\n",
       "export const base = 'https://api.github.com';\nconst token = \"ghp_abcdefghijklmnop\";\n")],
     [("web/gh.js", "ghp_abcdefghijklmnop", "SEC001", "detectable")])

case("12_two_issues_one_hunk", "Quick admin tool",
     [("src/admin.py", "def run():\n    pass\n",
       "def run():\n    cmd = input()\n    eval(cmd)\n    # TODO: add auth check\n    pass\n")],
     [("src/admin.py", "eval(cmd)", "SEC002", "detectable"), ("src/admin.py", "TODO", "STY001", "detectable")])

case("13_comment_mentions_eval", "Document security policy in code",
     [("src/policy.py", "def policy():\n    return 'strict'\n",
       "def policy():\n    # never call eval( on user input\n    return 'strict'\n")],
     [])   # a comment is not a defect -> regex analyzers produce a false positive here

# ---- write ------------------------------------------------------------------------------------
EVENT = {"action": "opened", "repository": {"full_name": "demo-org/demo-repo"},
         "pull_request": {"number": 0, "title": "", "body": "", "draft": False, "user": {"login": "student1"},
           "head": {"ref": "feature/x", "sha": "", "repo": {"full_name": "demo-org/demo-repo"}},
           "base": {"ref": "main", "sha": "1" * 40, "repo": {"full_name": "demo-org/demo-repo"}}}}

if os.path.isdir(OUT): shutil.rmtree(OUT)
for i, (name, title, body, files, defects) in enumerate(CASES, start=1):
    d = os.path.join(OUT, name); os.makedirs(d)
    diff, new_texts = make_diff(files)
    ev = json.loads(json.dumps(EVENT)); pr = ev["pull_request"]
    pr.update(number=100 + i, title=title, body=body)
    pr["head"]["sha"] = f"{i:02d}" + "a" * 38
    json.dump(ev, open(os.path.join(d, "pr_event.json"), "w"), indent=2)
    open(os.path.join(d, "sample.diff"), "w").write(diff)
    exp = [{"file": f, "line": locate(new_texts, f, needle), "rule": rule, "kind": kind}
           for f, needle, rule, kind in defects]
    json.dump({"case": name, "defects": exp}, open(os.path.join(d, "expected.json"), "w"), indent=2)
print(f"wrote {len(CASES)} cases to {OUT}")
