"""Local stand-in for api.github.com (stdlib http.server) for 8 Oct / 14-15 Oct workflow tests.

Behaves like GitHub where it matters to us:
  * 401 without a Bearer token
  * POST /reviews returns 422 if any comment targets a line outside the diff
  * list-comments returns what earlier reviews posted (so re-run dedupe can be tested)
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from reflexicode_guard.diff_utils import commentable_lines


class FakeGitHub:
    def __init__(self, pr: dict, files: list[dict], require_token=True, flaky_first_n=0):
        self.pr, self.files, self.require_token = pr, files, require_token
        self.flaky_left = flaky_first_n          # first N requests return 502 (retry test)
        self.reviews, self.issue_comments, self.review_comments, self.requests = [], [], [], []
        self._srv = None

    # --- lifecycle -------------------------------------------------------
    def start(self) -> str:
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            def _route(self, method):
                path = self.path.split("?")[0]
                outer.requests.append((method, self.path))
                if outer.flaky_left > 0:
                    outer.flaky_left -= 1; return self._send(502, {"message": "Bad Gateway"})
                if outer.require_token and not self.headers.get("Authorization", "").startswith("Bearer "):
                    return self._send(401, {"message": "Requires authentication"})
                body = {}
                n = int(self.headers.get("Content-Length") or 0)
                if n: body = json.loads(self.rfile.read(n))
                if method == "GET" and re.fullmatch(r"/repos/[^/]+/[^/]+/pulls/\d+", path):
                    return self._send(200, outer.pr)
                if method == "GET" and path.endswith("/files"):
                    return self._send(200, outer.files)
                if method == "GET" and path.endswith("/pulls/%d/comments" % outer.pr["number"]):
                    return self._send(200, outer.review_comments)
                if method == "POST" and path.endswith("/reviews"):
                    valid = {f["filename"]: commentable_lines(f.get("patch")) for f in outer.files}
                    for c in body.get("comments", []):
                        if c["line"] not in valid.get(c["path"], set()):
                            return self._send(422, {"message": "Unprocessable Entity",
                                                    "errors": [f"line {c['line']} is not part of the diff"]})
                    outer.reviews.append(body)
                    outer.review_comments += [{"body": c["body"], "path": c["path"], "line": c["line"]}
                                              for c in body.get("comments", [])]
                    return self._send(200, {"id": len(outer.reviews)})
                if method == "POST" and "/issues/" in path and path.endswith("/comments"):
                    outer.issue_comments.append(body); return self._send(201, {"id": 1})
                return self._send(404, {"message": "Not Found"})
            def do_GET(self): self._route("GET")
            def do_POST(self): self._route("POST")

        self._srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self._srv.server_port}"

    def stop(self):
        if self._srv:
            self._srv.shutdown(); self._srv.server_close()
