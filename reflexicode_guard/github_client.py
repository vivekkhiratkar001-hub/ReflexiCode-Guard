"""Day 1 (30 Sep): thin GitHub REST client (stdlib only, injectable transport)."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

from .events import extract_linked_issues
from .models import ChangedFile, PRContext

API = "https://api.github.com"


class GitHubError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"GitHub API {status}: {message}")
        self.status = status


def _urllib_transport(method, url, headers, body):
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


class GitHubClient:
    def __init__(self, token: Optional[str], api_url: str = API,
                 transport: Callable = _urllib_transport, retries: int = 3, sleep=time.sleep):
        self.token, self.api_url = token, api_url.rstrip("/")
        self.transport, self.retries, self.sleep = transport, retries, sleep

    def _request(self, method: str, path: str, payload=None, accept="application/vnd.github+json"):
        url = path if path.startswith("http") else f"{self.api_url}{path}"
        headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28",
                   "User-Agent": "reflexicode-guard"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        body = None
        if payload is not None:
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        for attempt in range(self.retries):
            status, text = self.transport(method, url, headers, body)
            retryable = status >= 500 or status == 429 or (status == 403 and "rate limit" in text.lower())
            if retryable and attempt < self.retries - 1:
                self.sleep(2 ** attempt)
                continue
            if status >= 400:
                raise GitHubError(status, text[:300])
            return text if accept.endswith(".diff") else (json.loads(text) if text else {})

    # ---- read -----------------------------------------------------------
    def get_pull(self, repo: str, number: int) -> dict:
        return self._request("GET", f"/repos/{repo}/pulls/{number}")

    def list_pull_files(self, repo: str, number: int) -> list[dict]:
        files, page = [], 1
        while True:
            chunk = self._request("GET", f"/repos/{repo}/pulls/{number}/files?per_page=100&page={page}")
            files += chunk
            if len(chunk) < 100:
                return files
            page += 1

    def list_review_comments(self, repo: str, number: int) -> list[dict]:
        out, page = [], 1
        while True:
            chunk = self._request("GET", f"/repos/{repo}/pulls/{number}/comments?per_page=100&page={page}")
            out += chunk
            if len(chunk) < 100:
                return out
            page += 1

    def get_diff(self, repo: str, number: int) -> str:
        return self._request("GET", f"/repos/{repo}/pulls/{number}", accept="application/vnd.github.v3.diff")

    # ---- write ----------------------------------------------------------
    def create_review(self, repo, number, commit_id, body, comments, event="COMMENT") -> dict:
        return self._request("POST", f"/repos/{repo}/pulls/{number}/reviews",
                             {"commit_id": commit_id, "body": body, "event": event, "comments": comments})

    def create_issue_comment(self, repo, number, body) -> dict:
        return self._request("POST", f"/repos/{repo}/issues/{number}/comments", {"body": body})

    # ---- metadata -> PRContext -----------------------------------------
    def build_context(self, repo: str, number: int) -> PRContext:
        pr = self.get_pull(repo, number)
        files = [ChangedFile(path=f["filename"], status=f["status"], patch=f.get("patch"),
                             additions=f.get("additions", 0), deletions=f.get("deletions", 0))
                 for f in self.list_pull_files(repo, number)]
        return PRContext(
            repo=repo, number=number, title=pr["title"], body=pr.get("body") or "",
            author=pr["user"]["login"], base_ref=pr["base"]["ref"], head_ref=pr["head"]["ref"],
            base_sha=pr["base"]["sha"], head_sha=pr["head"]["sha"], draft=pr.get("draft", False),
            labels=[l["name"] for l in pr.get("labels", [])], files=files,
            linked_issues=extract_linked_issues(pr.get("body")))
