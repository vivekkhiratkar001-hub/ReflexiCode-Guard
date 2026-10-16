"""Test that a timed-out semantic review is reported as PARTIAL/Error, not clean."""

import os
import subprocess
import sys
import tempfile
import threading
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

class FixtureHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/repos/fixture/timeout-test/pulls/1":
            payload = {
                "number": 1,
                "title": "Timeout Test",
                "body": "Testing timeout",
                "head": {"sha": "head-sha"},
            }
        elif path == "/repos/fixture/timeout-test/pulls/1/files":
            payload = [{
                "filename": "test.py",
                "status": "modified",
                "patch": "@@ -1,1 +1,1 @@\n-print(1)\n+print(2)\n",
            }]
        elif path == "/repos/fixture/timeout-test/pulls/1/comments":
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

    def do_POST(self) -> None:
        self.send_error(405)

    def log_message(self, _format, *args):
        return

def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            analysis_path = os.path.join(tmpdir, "analysis.json")
            with open(analysis_path, "w") as f:
                json.dump([], f)

            # Use a ridiculously low timeout (1 second) to force a timeout
            result = subprocess.run(
                [
                    sys.executable, "-m", "src.github",
                    "--owner", "fixture",
                    "--repo", "timeout-test",
                    "--pr-number", "1",
                    "--api-url", base_url,
                    "--analysis-file", analysis_path,
                    "--ollama-timeout", "1",
                    "--dry-run",
                ],
                capture_output=True, text=True, env={**os.environ, "REFLEXICODE_USE_MOCK_LLM": "0"}
            )

            stdout = result.stdout
            print(stdout)

            if "Errors" in stdout and "Ollama request timed out" in stdout:
                print("SUCCESS: Timeout was correctly reported as an error.")
                return 0
            else:
                print("FAILURE: Timeout was not reported as an error. Output:")
                print(stdout)
                return 1
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)

if __name__ == "__main__":
    sys.exit(main())
