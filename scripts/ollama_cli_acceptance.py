"""Exercise the production GitHub CLI with a real local Ollama model.

The temporary HTTP server only supplies read-only GitHub pull-request fixtures;
it does not implement or accept comment publication requests.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import monotonic
from urllib.parse import urlparse


class FixtureHandler(BaseHTTPRequestHandler):
    """Serve a tiny deterministic pull request without external GitHub access."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path == "/repos/fixture/semantic-review/pulls/1":
            payload: object = {
                "number": 1,
                "title": "Correct addition implementation",
                "body": "The add function must return the sum of its two arguments.",
                "html_url": "http://fixture.invalid/pull/1",
                "state": "open",
                "base": {"ref": "main", "sha": "base-sha"},
                "head": {"ref": "fix-add", "sha": "fixture-head-sha"},
            }
        elif path == "/repos/fixture/semantic-review/pulls/1/files":
            payload = [
                {
                    "filename": "src/math.py",
                    "status": "modified",
                    "patch": (
                        "@@ -1,2 +1,2 @@\n"
                        " def add(left, right):\n"
                        "-    return left + right\n"
                        "+    return left - right\n"
                    ),
                    "additions": 1,
                    "deletions": 1,
                    "changes": 2,
                }
            ]
        elif path == "/repos/fixture/semantic-review/pulls/1/comments":
            payload = []
        else:
            self.send_error(404)
            return

        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.send_error(405, "This acceptance fixture is read-only; publishing is disabled.")

    def log_message(self, _format: str, *args: object) -> None:
        return


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    environment = os.environ.copy()
    environment["REFLEXICODE_USE_MOCK_LLM"] = "0"
    environment.pop("REFLEXICODE_MOCK_LLM_RESPONSE", None)
    started = monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="reflexicode-acceptance-") as tmpdir:
            analysis_path = os.path.join(tmpdir, "analysis.json")
            with open(analysis_path, "w", encoding="utf-8") as fixture:
                json.dump([], fixture)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.github",
                    "--owner",
                    "fixture",
                    "--repo",
                    "semantic-review",
                    "--pr-number",
                    "1",
                    "--api-url",
                    base_url,
                    "--analysis-file",
                    analysis_path,
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=False,
                env=environment,
            )
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)

    elapsed = monotonic() - started
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    expected = (
        "Total findings: 1",
        "Semantic status: success_with_findings",
        "src/math.py:2",
        "left - right",
        "Failed batches: 0",
        "Not reviewed files\n- none",
        "Comments that would be published: 1",
        "DRY RUN (no GitHub comments were created)",
    )
    missing = [item for item in expected if not result.stdout.strip() or item not in result.stdout]
    if result.returncode != 0 or missing:
        print(
            "Real-model CLI acceptance failed: "
            f"exit_code={result.returncode}, missing_checks={missing}, "
            f"elapsed_seconds={elapsed:.2f}",
            file=sys.stderr,
        )
        return 1
    print(
        "Real-model CLI acceptance passed: controlled arithmetic defect detected "
        f"with changed-line evidence; elapsed_seconds={elapsed:.2f}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
